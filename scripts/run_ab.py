"""Evaluate the cloud base model (A) or a LoRA adapter (B) on ChartQA val."""

import argparse
import csv
import hashlib
import json
import platform
import socket
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import pyarrow.parquet as pq
import torch
from filelock import FileLock
from huggingface_hub import hf_hub_download
from peft import PeftModel
from transformers import AutoProcessor, GenerationConfig

import evaluate as scoring
from inspect_chartqa import DATASET_ID, DATASET_REVISION, PARQUET_FILE
from run_baseline import DATA_DIR, HUB_CACHE, MODEL_ID, MODEL_REVISION, PROJECT_ROOT
from run_baseline_batch import append_record, chart_image, completed_ids, predict, read_json, write_json
from run_lora_smoke import INFERENCE_SETTINGS, MODEL_ARGS, load_base


P0_MANIFEST_SHA256 = "6e7c7b7ba23acacf41817904a48c47319b88d9726e4a150d75c2452d4d6a0a10"
P0_VAL_SHA256 = "46ee830d03a1f288ef21be3d8d149d307cbf09404f9e056906169e617679fb09"
P1_DIR = PROJECT_ROOT / "outputs" / "chartqa_p1"
DEFAULT_ADAPTER = P1_DIR / "experiment_b" / "adapter"
BAD_CASE_FIELDS = ["sample_id", "row_index", "imgname", "type", "question", "reference_answer", "raw_prediction"]


def file_digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def encode_manifest(manifest: dict) -> bytes:
    return (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def remaining_val_manifest(p0_manifest: dict, metadata) -> dict:
    excluded_rows = {sample["row_index"] for sample in p0_manifest["samples"]}
    samples = [
        {
            "sample_id": f"val:{row_index}", "row_index": row_index,
            "imgname": row["imgname"], "type": row["type"],
            "question": row["query"], "reference_answer": row["label"],
        }
        for row_index, row in enumerate(metadata.to_pylist())
        if row_index not in excluded_rows
    ]
    if metadata.num_rows != 1920 or Counter(sample["type"] for sample in samples) != {
        "human": 910, "augmented": 910,
    }:
        raise ValueError("Expected 910 human and 910 augmented rows after excluding the P0 selection.")
    return {
        "dataset_id": p0_manifest["dataset_id"], "dataset_revision": p0_manifest["dataset_revision"],
        "split": p0_manifest["split"], "parquet_file": p0_manifest["parquet_file"],
        "selection": "remaining-val", "excluded_p0_manifest_sha256": P0_MANIFEST_SHA256,
        "samples": samples,
    }


def load_reference(eval_set: str = "p0") -> tuple[dict, dict, Path]:
    settings = read_json(DATA_DIR / "batch_bf16_512" / "config.json")
    manifest_bytes = (DATA_DIR / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    digest = hashlib.sha256(manifest_bytes).hexdigest()
    if digest != P0_MANIFEST_SHA256 or settings["manifest_sha256"] != digest:
        raise ValueError("Expected the byte-for-byte P0 validation manifest and its original config.")
    inference = {
        **INFERENCE_SETTINGS, "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "dataset_revision": DATASET_REVISION, "dtype": "bfloat16", "attention": "sdpa",
    }
    if any(settings[key] != value for key, value in inference.items()):
        raise ValueError("The uploaded config does not match the fixed P0 inference settings.")
    if (manifest["dataset_id"], manifest["dataset_revision"], manifest["split"], manifest["parquet_file"]) != (
        DATASET_ID, DATASET_REVISION, "val", PARQUET_FILE,
    ):
        raise ValueError("Expected the pinned official ChartQA val data.")
    samples = manifest["samples"]
    if len({sample["sample_id"] for sample in samples}) != 100 or Counter(
        sample["type"] for sample in samples
    ) != {"human": 50, "augmented": 50}:
        raise ValueError("Expected 100 unique samples, 50 human and 50 augmented.")

    path = Path(hf_hub_download(
        DATASET_ID, PARQUET_FILE, repo_type="dataset", revision=DATASET_REVISION,
        cache_dir=HUB_CACHE, local_files_only=True,
    ))
    val_digest = file_digest(path)
    if val_digest != P0_VAL_SHA256:
        raise ValueError("Cached val Parquet differs from the P0 file.")
    metadata = pq.read_table(path, columns=["imgname", "query", "label", "type"])
    for sample in samples:
        row = metadata.slice(sample["row_index"], 1).to_pylist()
        expected = {
            "imgname": sample["imgname"], "query": sample["question"],
            "label": sample["reference_answer"], "type": sample["type"],
        }
        if row != [expected]:
            raise ValueError(f"Cached val row does not match manifest: {sample['sample_id']}")
    scoring.check_scoring_rules()
    shared = {
        **inference, "manifest_sha256": digest, "val_sha256": val_digest,
        "scoring_sha256": file_digest(Path(scoring.__file__)),
    }
    if eval_set == "remaining-val":
        manifest = remaining_val_manifest(manifest, metadata)
        shared["manifest_sha256"] = hashlib.sha256(encode_manifest(manifest)).hexdigest()
    return manifest, shared, path


def adapter_identity(path: Path) -> dict:
    config = path / "adapter_config.json"
    weights = path / "adapter_model.safetensors"
    if not config.is_file() or not weights.is_file():
        raise ValueError(f"Expected a complete local LoRA adapter in {path}")
    settings = read_json(config)
    if (settings.get("base_model_name_or_path"), settings.get("revision"), settings.get("peft_type")) != (
        MODEL_ID, MODEL_REVISION, "LORA",
    ):
        raise ValueError("Adapter must be LoRA for the pinned P0 base model and revision.")
    return {"path": str(path), "config_sha256": file_digest(config), "weights_sha256": file_digest(weights)}


def run_predictions(manifest, shared, parquet_path, predictions_path, adapter_path) -> None:
    samples = manifest["samples"]
    seen = completed_ids(predictions_path, {sample["sample_id"] for sample in samples})
    if len(seen) == len(samples):
        return

    processor = AutoProcessor.from_pretrained(MODEL_ID, **MODEL_ARGS)
    processor.image_processor.size = {"shortest_edge": 4 * 32**2, "longest_edge": shared["max_image_tokens"] * 32**2}
    device = torch.device("cuda:0")
    model = load_base(device).eval()
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False, local_files_only=True).eval().get_base_model()
    defaults = model.generation_config
    generation = GenerationConfig(
        bos_token_id=defaults.bos_token_id, eos_token_id=defaults.eos_token_id, pad_token_id=defaults.pad_token_id,
        max_new_tokens=shared["max_new_tokens"], do_sample=shared["do_sample"],
    )
    images = pq.read_table(parquet_path, columns=["image"])["image"]
    progress_interval = 10 if len(samples) <= 100 else 100
    for sample in samples:
        sample_id = sample["sample_id"]
        if sample_id in seen:
            continue
        started = perf_counter()
        try:
            image = chart_image(images[sample["row_index"]].as_py())
            result = predict(model, processor, generation, device, image, sample["question"], shared)
            torch.cuda.synchronize(device)
        except Exception as error:
            raise RuntimeError(f"Prediction failed for {sample_id}. Completed predictions can be resumed.") from error
        append_record(predictions_path, {
            **sample, **result, "status": "ok", "latency_seconds": round(perf_counter() - started, 3),
        })
        seen.add(sample_id)
        if len(seen) % progress_interval == 0 or len(seen) == len(samples):
            print(f"Predictions saved: {len(seen)}/{len(samples)}", flush=True)


def score_group(manifest: dict, config: dict, group_dir: Path) -> None:
    samples = manifest["samples"]
    rows = scoring.evaluate(samples, scoring.read_predictions(group_dir / "predictions.jsonl"))
    overall = scoring.summarize(rows)
    if overall["missing_predictions"]:
        raise RuntimeError("Finish the fixed evaluation set before scoring the group.")
    summary = {
        "group": config["group"], "manifest_sha256": config["shared"]["manifest_sha256"],
        "relative_numeric_tolerance": str(scoring.TOLERANCE), "overall": overall,
        "by_type": {kind: scoring.summarize([row for row in rows if row["type"] == kind])
                    for kind in ("human", "augmented")},
    }
    with (group_dir / "scored_predictions.jsonl").open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (group_dir / "bad_cases.csv").open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=BAD_CASE_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(row for row in rows if not row["relaxed_match"])
    write_json(group_dir / "summary.json", summary)
    print(f"{config['group']}: {overall}", flush=True)


def compare_groups(output_root: Path) -> None:
    paths = [output_root / group / "summary.json" for group in ("A", "B")]
    if not all(path.is_file() for path in paths):
        return
    a, b = (read_json(path) for path in paths)
    metrics = ("exact_accuracy_pct", "relaxed_accuracy_pct")
    a_scores = {"overall": a["overall"], **a["by_type"]}
    b_scores = {"overall": b["overall"], **b["by_type"]}
    comparison = {
        "A": a, "B": b,
        "b_minus_a_percentage_points": {
            category: {metric: round(b_scores[category][metric] - scores[metric], 2) for metric in metrics}
            for category, scores in a_scores.items()
        },
    }
    write_json(output_root / "comparison.json", comparison)
    print(f"A/B comparison saved: {output_root / 'comparison.json'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="Check the validation selection and cached inputs without a GPU.")
    parser.add_argument("--group", choices=("A", "B"))
    parser.add_argument("--eval-set", choices=("p0", "remaining-val"), default="p0",
                        help="Evaluate the fixed P0 100 or the other 1820 ChartQA val rows.")
    parser.add_argument("--adapter", type=Path, help="B adapter directory (defaults to experiment_b/adapter).")
    parser.add_argument("--output-root", type=Path, help="Defaults to ab or ab_remaining_val for the selected set.")
    args = parser.parse_args()
    if not args.check_only and args.group is None:
        parser.error("Specify --group A or --group B for inference.")
    if args.adapter is not None and args.group != "B":
        parser.error("--adapter is only used with --group B.")

    manifest, shared, parquet = load_reference(args.eval_set)
    hf_hub_download(MODEL_ID, "model.safetensors", **MODEL_ARGS)
    generation = GenerationConfig.from_pretrained(MODEL_ID, **MODEL_ARGS)
    if generation.eos_token_id != [151645, 151643]:
        raise ValueError("Unexpected generation stop tokens for the pinned P0 model.")
    if args.check_only:
        print(f"Offline A/B preflight PASS | matched val rows: {len(manifest['samples'])}")
        return
    if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported(including_emulation=False):
        raise SystemExit("A/B inference needs exactly one visible CUDA GPU with native BF16 support.")

    shared.update(
        gpu=torch.cuda.get_device_name(0), compute_capability=list(torch.cuda.get_device_capability(0)),
        host=socket.gethostname(), platform=platform.system(), python=platform.python_version(),
        cuda=torch.version.cuda, dependencies={name: version(name) for name in (
            "torch", "torchvision", "transformers", "peft", "pillow", "tokenizers", "pyarrow",
        )},
    )
    adapter_path = (args.adapter or DEFAULT_ADAPTER).resolve() if args.group == "B" else None
    config = {"group": args.group, "shared": shared,
              "adapter": adapter_identity(adapter_path) if adapter_path is not None else None}
    default_run_name = "ab_remaining_val" if args.eval_set == "remaining-val" else "ab"
    output_root = (args.output_root or P1_DIR / default_run_name).resolve()
    group_dir = output_root / args.group
    group_dir.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output_root / ".ab.lock"), timeout=0):
        other_config = output_root / ("B" if args.group == "A" else "A") / "config.json"
        if other_config.is_file() and read_json(other_config)["shared"] != shared:
            raise ValueError("A and B must use the same environment, inference settings, val data, and scoring code.")
        config_path = group_dir / "config.json"
        if config_path.is_file():
            if read_json(config_path) != config:
                raise ValueError("Group configuration changed. Use the original settings or a new --output-root.")
        else:
            if (group_dir / "predictions.jsonl").exists():
                raise ValueError("Existing predictions have no configuration. Use a new --output-root.")
            write_json(config_path, config)
        if args.eval_set == "remaining-val":
            manifest_path = output_root / "manifest.json"
            payload = encode_manifest(manifest)
            if not manifest_path.is_file():
                manifest_path.write_bytes(payload)
            elif manifest_path.read_bytes() != payload:
                raise ValueError("Saved validation selection differs. Use a new --output-root.")
        run_predictions(manifest, shared, parquet, group_dir / "predictions.jsonl", adapter_path)
        score_group(manifest, config, group_dir)
        compare_groups(output_root)


if __name__ == "__main__":
    main()
