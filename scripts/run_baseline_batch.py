"""Run the fixed ChartQA P0 sample with the validated single-image settings."""

import hashlib
import json
from io import BytesIO
from pathlib import Path
from time import perf_counter

import pyarrow.parquet as pq
import torch
import transformers
from filelock import FileLock
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import AutoProcessor, GenerationConfig, Qwen3VLForConditionalGeneration

from run_baseline import DATA_DIR, HUB_CACHE


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def append_record(path: Path, record: dict) -> None:
    with path.open("ab") as output:
        output.write((json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8"))


def completed_ids(path: Path, sample_ids: set[str]) -> set[str]:
    if not path.exists():
        return set()
    completed = set()
    with path.open("r+b") as output:
        while line := output.readline():
            if not line.endswith(b"\n"):
                # An interrupted append can leave a partial last line. Retry that sample.
                output.truncate(output.tell() - len(line))
                break
            sample_id = json.loads(line)["sample_id"]
            if sample_id not in sample_ids or sample_id in completed:
                raise ValueError(f"Invalid or duplicate prediction ID: {sample_id}")
            completed.add(sample_id)
    return completed


def chart_image(value) -> Image.Image:
    image_bytes = value.get("bytes") if isinstance(value, dict) else value
    if not isinstance(image_bytes, (bytes, bytearray, list)) or not image_bytes:
        raise ValueError("Missing embedded image bytes")
    with Image.open(BytesIO(bytes(image_bytes))) as image:
        return image.convert("RGB")


def predict(model, processor, generation, device, image, question, settings) -> dict:
    prompt = f"{question}\n{settings['prompt_suffix']}"
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[chat_text], images=[image], return_tensors="pt").to(device)
    input_tokens = inputs.input_ids.shape[1]
    image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    image_tokens = int((inputs.input_ids == image_token_id).sum().item())
    if not 0 < image_tokens <= settings["max_image_tokens"]:
        raise ValueError(f"Unexpected image token count: {image_tokens}")

    with torch.inference_mode():
        output = model.generate(**inputs, generation_config=generation, use_model_defaults=False)
    new_tokens = output[:, input_tokens:]
    raw_prediction = processor.batch_decode(
        new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return {
        "raw_prediction": raw_prediction,
        "input_tokens": input_tokens,
        "image_tokens": image_tokens,
        "generated_tokens": new_tokens.shape[1],
    }


def run_batch(manifest: dict, settings: dict, output_dir: Path) -> int:
    samples = manifest["samples"]
    predictions_path = output_dir / "predictions.jsonl"
    failures_path = output_dir / "failures.jsonl"
    sample_ids = {sample["sample_id"] for sample in samples}
    if len(samples) != 100 or len(sample_ids) != len(samples):
        raise ValueError("Expected the fixed manifest with 100 unique sample IDs.")
    completed = completed_ids(predictions_path, sample_ids)
    if completed == sample_ids:
        print(f"All {len(samples)} predictions are already saved: {predictions_path}")
        return 0

    data_path = hf_hub_download(
        manifest["dataset_id"], manifest["parquet_file"], repo_type="dataset",
        revision=manifest["dataset_revision"], cache_dir=HUB_CACHE, local_files_only=True,
    )
    charts = pq.read_table(data_path, columns=["image"])["image"]
    model_args = {
        "revision": settings["model_revision"],
        "cache_dir": HUB_CACHE,
        "local_files_only": True,
    }
    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats(device)
    print(f"Loading cached model on {settings['gpu']} (BF16)", flush=True)
    processor = AutoProcessor.from_pretrained(settings["model_id"], **model_args)
    processor.image_processor.size = {
        "shortest_edge": 4 * 32**2,
        "longest_edge": settings["max_image_tokens"] * 32**2,
    }
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        settings["model_id"], dtype=torch.bfloat16, attn_implementation=settings["attention"],
        device_map={"": 0}, **model_args,
    ).eval()
    defaults = model.generation_config
    generation = GenerationConfig(
        bos_token_id=defaults.bos_token_id,
        eos_token_id=defaults.eos_token_id,
        pad_token_id=defaults.pad_token_id,
        max_new_tokens=settings["max_new_tokens"],
        do_sample=settings["do_sample"],
    )

    print(f"Ready: {len(completed)}/{len(samples)} predictions already saved", flush=True)
    for sample in samples:
        sample_id = sample["sample_id"]
        if sample_id in completed:
            continue
        started = perf_counter()
        try:
            image = chart_image(charts[sample["row_index"]].as_py())
            prediction = predict(model, processor, generation, device, image, sample["question"], settings)
            torch.cuda.synchronize(device)
        except (OSError, ValueError, torch.cuda.OutOfMemoryError) as error:
            append_record(failures_path, {**sample, "error": type(error).__name__, "message": str(error)})
            print(f"Failed {sample_id}: {error}", flush=True)
            continue
        except Exception as error:
            append_record(failures_path, {**sample, "error": type(error).__name__, "message": str(error)})
            raise RuntimeError(f"Unexpected inference error for {sample_id}") from error
        append_record(
            predictions_path,
            {**sample, **prediction, "status": "ok", "latency_seconds": round(perf_counter() - started, 3)},
        )
        completed.add(sample_id)
        if len(completed) % 10 == 0:
            print(f"Saved {len(completed)}/{len(samples)} predictions", flush=True)

    missing = [sample["sample_id"] for sample in samples if sample["sample_id"] not in completed]
    report = {
        "predictions_saved": len(completed),
        "planned_samples": len(samples),
        "missing_sample_ids": missing,
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated(device) / 1024**2, 1),
        "peak_reserved_mib": round(torch.cuda.max_memory_reserved(device) / 1024**2, 1),
    }
    write_json(output_dir / "last_run.json", report)
    print(json.dumps(report, indent=2), flush=True)
    print(f"Predictions: {predictions_path}", flush=True)
    return int(bool(missing))


def main() -> int:
    smoke = read_json(DATA_DIR / "smoke_bf16_512.json")
    manifest_path = DATA_DIR / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if smoke["dataset_revision"] != manifest["dataset_revision"]:
        raise ValueError("Smoke test and manifest refer to different dataset revisions.")
    if smoke["dtype"] != "bfloat16" or smoke["max_image_tokens"] != 512 or smoke["do_sample"]:
        raise ValueError("Expected the validated 512-token BF16 smoke configuration.")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("This baseline requires a CUDA GPU with BF16 support.")

    settings = {
        key: smoke[key] for key in (
            "model_id", "model_revision", "dataset_revision", "dtype", "attention", "prompt_suffix",
            "do_sample", "max_image_tokens", "max_new_tokens",
        )
    }
    settings.update(
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        gpu=torch.cuda.get_device_name(0),
        torch_version=str(torch.__version__),
        cuda_version=torch.version.cuda,
        transformers_version=transformers.__version__,
    )

    output_dir = DATA_DIR / "batch_bf16_512"
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "config.json"
    with FileLock(str(output_dir / ".run.lock"), timeout=0):
        if config_path.exists():
            if read_json(config_path) != settings:
                raise ValueError("Batch settings or manifest changed. Existing predictions cannot be resumed.")
        else:
            write_json(config_path, settings)
        return run_batch(manifest, settings, output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
