"""Train experiment B on the official ChartQA train split using cached files."""

import argparse
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import accelerate
import peft
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import transformers
from filelock import FileLock
from huggingface_hub import hf_hub_download
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, Trainer, TrainerCallback, TrainingArguments, set_seed

from inspect_chartqa import DATASET_ID, DATASET_REVISION, SEED
from inspect_sft_sample import MAX_IMAGE_TOKENS, encode_training_sample
from run_baseline import HUB_CACHE, MODEL_ID, MODEL_REVISION, PROJECT_ROOT
from run_baseline_batch import chart_image, read_json, write_json
from run_lora_smoke import LEARNING_RATE, LORA_ALPHA, LORA_RANK, LORA_TARGET, MODEL_ARGS, load_base


TRAIN_SHARDS = tuple(f"data/train-{index:05d}-of-00003.parquet" for index in range(3))
DATA_COLUMNS = ["image", "query", "label"]
CHECKPOINT_FILES = (
    "adapter_config.json", "adapter_model.safetensors", "optimizer.pt",
    "scheduler.pt", "rng_state.pth", "trainer_state.json",
)


def cached_train_shards() -> list[Path]:
    return [
        Path(hf_hub_download(
            DATASET_ID, shard, repo_type="dataset", revision=DATASET_REVISION,
            cache_dir=HUB_CACHE, local_files_only=True,
        ))
        for shard in TRAIN_SHARDS
    ]


def check_shards(paths: list[Path]) -> list[int]:
    counts = []
    for path in paths:
        with pq.ParquetFile(path) as parquet:
            count = parquet.metadata.num_rows
            if count == 0 or not set(DATA_COLUMNS).issubset(parquet.schema_arrow.names):
                raise ValueError(f"Missing training rows or columns: {path.name}")
            for batch in parquet.iter_batches(columns=["query", "label"]):
                for row in batch.to_pylist():
                    if not all(isinstance(row[key], str) and row[key].strip() for key in ("query", "label")):
                        raise ValueError(f"Found a missing question or answer in {path.name}")
        counts.append(count)
        print(f"{path.name}: {count} rows", flush=True)
    return counts


class ChartQATrainDataset(torch.utils.data.Dataset):
    def __init__(self, paths: list[Path]):
        # Keep compressed image bytes in Arrow, decoding only the sampled chart per step.
        self.rows = pa.concat_tables([pq.read_table(path, columns=DATA_COLUMNS) for path in paths])

    def __len__(self) -> int:
        return self.rows.num_rows

    def __getitem__(self, index: int) -> dict:
        return {**self.rows.slice(index, 1).to_pylist()[0], "sample_id": f"train:{index}"}


def collate_one(processor, samples: list[dict]) -> dict:
    sample, = samples
    try:
        batch, _, _ = encode_training_sample(
            processor, chart_image(sample["image"]), sample["query"].strip(), sample["label"].strip(),
        )
    except ValueError as error:
        raise ValueError(f"Failed to encode {sample['sample_id']}: {error}") from error
    return {**batch, "use_cache": False}


class CheckTrainingMetrics(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        for key in ("loss", "grad_norm"):
            if logs is not None and key in logs and not math.isfinite(logs[key]):
                raise RuntimeError(f"Non-finite {key} at training step {state.global_step}: {logs[key]}")


def latest_checkpoint(output_dir: Path) -> str | None:
    latest_step, latest_path = 0, None
    for path in output_dir.glob("checkpoint-*"):
        step = path.name.removeprefix("checkpoint-")
        if not step.isdecimal() or int(step) <= latest_step:
            continue
        if all((path / name).is_file() and (path / name).stat().st_size > 0 for name in CHECKPOINT_FILES):
            try:
                state = read_json(path / "trainer_state.json")
            except ValueError:
                continue
            if state.get("global_step") == int(step):
                latest_step, latest_path = int(step), str(path)
    return latest_path


def train(args, paths: list[Path], counts: list[int], processor, output_dir: Path) -> None:
    settings = {
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION, "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION, "train_shards": list(TRAIN_SHARDS), "train_rows_by_shard": counts,
        "seed": SEED, "max_steps": args.max_steps, "save_steps": args.save_steps, "micro_batch_size": 1,
        "max_image_tokens": MAX_IMAGE_TOKENS, "dtype": "bfloat16", "attention": "sdpa",
        "weight_decay": 0.0, "learning_rate": LEARNING_RATE, "lr_scheduler": "constant", "max_grad_norm": 1.0,
        "lora_rank": LORA_RANK, "lora_alpha": LORA_ALPHA, "lora_dropout": 0.0, "lora_target": LORA_TARGET.pattern,
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda": torch.version.cuda,
        "transformers": transformers.__version__, "peft": peft.__version__, "accelerate": accelerate.__version__,
    }
    config_path = output_dir / "config.json"
    adapter_path = output_dir / "adapter"
    if adapter_path.exists():
        raise SystemExit(f"Final adapter already exists: {adapter_path}")
    checkpoint = None
    if args.resume:
        if not config_path.is_file() or read_json(config_path) != settings:
            raise SystemExit("Resume requires the original training configuration and environment.")
        checkpoint = latest_checkpoint(output_dir)
        print(f"Resume point: {checkpoint or 'step 0 (no complete checkpoint)'}", flush=True)
    elif any(path.name != ".train.lock" for path in output_dir.iterdir()):
        raise SystemExit("Output directory is in use. Use --resume or a new --output-dir.")
    else:
        write_json(config_path, settings)

    set_seed(SEED)
    torch.cuda.reset_peak_memory_stats(0)
    base = load_base(torch.device("cuda:0"))
    targets = [name for name, module in base.named_modules()
               if LORA_TARGET.fullmatch(name) and isinstance(module, torch.nn.Linear)]
    if len(targets) != 2 * base.config.text_config.num_hidden_layers:
        raise RuntimeError("Expected q_proj and v_proj in every language decoder layer.")
    model = get_peft_model(base, LoraConfig(
        task_type="CAUSAL_LM", r=LORA_RANK, lora_alpha=LORA_ALPHA,
        lora_dropout=0.0, bias="none", target_modules=targets, revision=MODEL_REVISION,
    ))
    model.print_trainable_parameters()

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(output_dir), max_steps=args.max_steps, per_device_train_batch_size=1,
            learning_rate=LEARNING_RATE, lr_scheduler_type="constant", optim="adamw_torch",
            weight_decay=0.0, max_grad_norm=1.0, logging_steps=1, logging_nan_inf_filter=False,
            save_strategy="steps", save_steps=args.save_steps, save_total_limit=2,
            seed=SEED, data_seed=SEED, dataloader_num_workers=0, remove_unused_columns=False,
            label_names=["labels"], report_to=[], accelerator_config={"use_seedable_sampler": True},
        ),
        train_dataset=ChartQATrainDataset(paths),
        data_collator=lambda samples: collate_one(processor, samples),
        callbacks=[CheckTrainingMetrics()],
    )
    trainer.model_accepts_loss_kwargs = False
    result = trainer.train(resume_from_checkpoint=checkpoint)
    if result.global_step != args.max_steps or not math.isfinite(result.training_loss):
        raise RuntimeError(f"Training ended at step {result.global_step} with loss {result.training_loss}.")
    trainer.save_state()
    report = {
        "steps": result.global_step, "metrics": result.metrics,
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated(0) / 1024**2, 1),
        "peak_reserved_mib": round(torch.cuda.max_memory_reserved(0) / 1024**2, 1),
        "adapter_dir": str(adapter_path),
    }
    with TemporaryDirectory(dir=output_dir) as staging:
        staged_adapter = Path(staging) / "adapter"
        model.save_pretrained(staged_adapter, safe_serialization=True, save_embedding_layers=False)
        write_json(output_dir / "summary.json", report)
        staged_adapter.rename(adapter_path)
    print(f"Experiment B adapter saved: {adapter_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="Check cached inputs without loading the model.")
    parser.add_argument("--max-steps", type=int, help="Optimizer steps for experiment B, chosen after the smoke test.")
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "chartqa_p1" / "experiment_b")
    parser.add_argument("--resume", action="store_true", help="Continue the same run from a complete checkpoint.")
    args = parser.parse_args()
    if not args.check_only and (args.max_steps is None or args.max_steps < 1 or args.save_steps < 1):
        parser.error("Training needs positive --max-steps and --save-steps.")

    paths = cached_train_shards()
    counts = check_shards(paths)
    hf_hub_download(MODEL_ID, "model.safetensors", **MODEL_ARGS)
    processor = AutoProcessor.from_pretrained(MODEL_ID, **MODEL_ARGS)
    processor.image_processor.size = {"shortest_edge": 4 * 32**2, "longest_edge": MAX_IMAGE_TOKENS * 32**2}
    if args.check_only:
        print(f"Offline training preflight PASS | train rows: {sum(counts)}")
        return
    if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported(including_emulation=False):
        raise SystemExit("Experiment B needs exactly one visible CUDA GPU with native BF16 support.")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output_dir / ".train.lock"), timeout=0):
        train(args, paths, counts, processor, output_dir)


if __name__ == "__main__":
    main()
