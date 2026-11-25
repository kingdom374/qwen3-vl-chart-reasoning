"""Check ChartQA causal loss and run a 10-step Qwen3-VL LoRA smoke test."""

import argparse
import json
import math
import re
from pathlib import Path
from time import perf_counter

import peft
import pyarrow.parquet as pq
import torch
import transformers
from huggingface_hub import hf_hub_download
from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict
from transformers import AutoProcessor, GenerationConfig, Qwen3VLForConditionalGeneration

from inspect_chartqa import DATASET_ID, DATASET_REVISION, SEED
from inspect_sft_sample import IGNORE_LABEL, MAX_IMAGE_TOKENS, TRAIN_SHARD, encode_training_sample
from run_baseline import HUB_CACHE, MAX_NEW_TOKENS, MODEL_ID, MODEL_REVISION, PROJECT_ROOT, PROMPT_SUFFIX
from run_baseline_batch import chart_image, predict


SAMPLE_COUNT = 8
STEPS = 10
LEARNING_RATE = 1e-4
LORA_RANK = 8
LORA_ALPHA = 16
LORA_TARGET = re.compile(r"model\.language_model\.layers\.\d+\.self_attn\.(q_proj|v_proj)")
MODEL_ARGS = {"revision": MODEL_REVISION, "cache_dir": HUB_CACHE, "local_files_only": True}
INFERENCE_SETTINGS = {
    "prompt_suffix": PROMPT_SUFFIX,
    "max_image_tokens": MAX_IMAGE_TOKENS,
    "max_new_tokens": MAX_NEW_TOKENS,
    "do_sample": False,
}


def read_training_samples() -> list[dict]:
    path = hf_hub_download(
        repo_id=DATASET_ID, repo_type="dataset", filename=TRAIN_SHARD,
        revision=DATASET_REVISION, cache_dir=HUB_CACHE, local_files_only=True,
    )
    with pq.ParquetFile(path) as parquet:
        batches = parquet.iter_batches(batch_size=SAMPLE_COUNT, columns=["image", "query", "label"])
        batch = next(batches, None)
        if batch is None or batch.num_rows != SAMPLE_COUNT:
            raise ValueError(f"Expected the first {SAMPLE_COUNT} rows of {TRAIN_SHARD}.")
        return batch.to_pylist()


def load_base(device: torch.device):
    return Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map={"": device.index}, **MODEL_ARGS,
    )


def check_first_loss(output, labels, tokenizer) -> dict:
    target_positions = (labels[0] != IGNORE_LABEL).nonzero(as_tuple=True)[0]
    target_ids = labels[0, target_positions]
    # Each answer label at position t is predicted by the logits at t - 1.
    logits = output.logits[0, target_positions - 1].detach().float()
    token_losses = torch.logsumexp(logits, dim=-1) - logits.gather(1, target_ids[:, None]).squeeze(1)
    manual_loss = token_losses.mean().item()
    model_loss = output.loss.item()
    if not math.isclose(manual_loss, model_loss, rel_tol=1e-4, abs_tol=1e-4):
        raise RuntimeError(f"Shifted cross-entropy mismatch: manual={manual_loss}, model={model_loss}.")
    return {
        "sample_id": "train:0",
        "manual_loss": manual_loss,
        "model_loss": model_loss,
        "tokens": [
            {"label_position": position, "logit_position": position - 1, "token_id": token_id,
             "token": tokenizer.decode([token_id], clean_up_tokenization_spaces=False), "nll": nll}
            for position, token_id, nll in zip(target_positions.tolist(), target_ids.tolist(), token_losses.tolist())
        ],
    }


def adapter_state(model) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in get_peft_model_state_dict(model).items()}


def predict_training_example(model, processor, device, sample) -> dict:
    base = model.get_base_model()  # The LoRA layers remain installed in the underlying Qwen model.
    defaults = base.generation_config
    generation = GenerationConfig(
        bos_token_id=defaults.bos_token_id, eos_token_id=defaults.eos_token_id, pad_token_id=defaults.pad_token_id,
        max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
    )
    return predict(base, processor, generation, device, chart_image(sample["image"]), sample["query"], INFERENCE_SETTINGS)


def train_adapter(processor, samples, device: torch.device, run_dir: Path) -> tuple[dict, dict[str, torch.Tensor]]:
    torch.cuda.reset_peak_memory_stats(device)
    base = load_base(device)
    targets = [
        name for name, module in base.named_modules()
        if LORA_TARGET.fullmatch(name) and isinstance(module, torch.nn.Linear)
    ]
    expected_count = 2 * base.config.text_config.num_hidden_layers
    if len(targets) != expected_count:
        raise RuntimeError(f"Expected {expected_count} text attention projections, found {len(targets)}.")

    model = get_peft_model(base, LoraConfig(
        task_type="CAUSAL_LM", r=LORA_RANK, lora_alpha=LORA_ALPHA,
        lora_dropout=0.0, bias="none", target_modules=targets, revision=MODEL_REVISION,
    )).train()
    trainable = {name: param for name, param in model.named_parameters() if param.requires_grad}
    expected_trainable = {
        f"{target}.lora_{matrix}.default.weight" for target in targets for matrix in ("A", "B")
    }
    if {name.removeprefix("base_model.model.") for name in trainable} != expected_trainable:
        raise RuntimeError("Expected only the LoRA A and B matrices of the selected text projections to train.")
    model.print_trainable_parameters()
    initial_state = adapter_state(model)
    optimizer = torch.optim.AdamW(trainable.values(), lr=LEARNING_RATE, weight_decay=0.0)

    steps = []
    with (run_dir / "steps.jsonl").open("x", encoding="utf-8") as log:
        for step in range(STEPS):
            started = perf_counter()
            row_index = step % len(samples)
            sample = samples[row_index]
            image = chart_image(sample["image"])
            batch, answer_start, answer_end = encode_training_sample(processor, image, sample["query"], sample["label"])
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(**batch, use_cache=False)
            loss = output.loss.item()
            if not math.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at step {step + 1}: {loss}.")
            if step == 0:
                loss_check = check_first_loss(output, batch["labels"], processor.tokenizer)
                print(f"First loss check: {json.dumps(loss_check, ensure_ascii=False)}", flush=True)
                (run_dir / "first_loss.json").write_text(json.dumps(loss_check, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

            output.loss.backward()
            norms = [param.grad.detach().float().norm() for param in trainable.values() if param.grad is not None]
            gradient_norm = torch.stack(norms).norm().item() if norms else 0.0
            if not math.isfinite(gradient_norm) or (step == 0 and gradient_norm == 0):
                raise RuntimeError(f"Invalid LoRA gradient norm at step {step + 1}: {gradient_norm}.")
            optimizer.step()
            torch.cuda.synchronize(device)
            record = {
                "step": step + 1, "sample_id": f"train:{row_index}", "loss": loss,
                "supervised_tokens": answer_end - answer_start + 1, "gradient_norm": gradient_norm,
                "step_seconds": round(perf_counter() - started, 3),
            }
            steps.append(record)
            line = json.dumps(record)
            log.write(line + "\n")
            log.flush()
            print(line, flush=True)
            del output, batch

    trained_state = adapter_state(model)
    changed_tensors = sum(not torch.equal(initial_state[name], tensor) for name, tensor in trained_state.items())
    if changed_tensors == 0 or not all(torch.isfinite(tensor).all().item() for tensor in trained_state.values()):
        raise RuntimeError("Expected updated, finite LoRA weights after optimizer steps.")
    model.save_pretrained(run_dir / "adapter", safe_serialization=True, save_embedding_layers=False)

    report = {
        "first_loss": loss_check,
        "trainable_parameters": sum(param.numel() for param in trainable.values()),
        "target_modules": targets,
        "updated_adapter_tensors": changed_tensors,
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated(device) / 1024**2, 1),
        "peak_reserved_mib": round(torch.cuda.max_memory_reserved(device) / 1024**2, 1),
        "step_seconds_total": round(sum(record["step_seconds"] for record in steps), 3),
        "prediction_before_reload": predict_training_example(model.eval(), processor, device, samples[0]),
    }
    return report, trained_state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "chartqa_p1" / "lora_smoke")
    run_dir = parser.parse_args().output_dir
    if run_dir.exists():
        raise SystemExit(f"Output directory already exists. Choose a new --output-dir: {run_dir}")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False):
        raise SystemExit("This offline smoke test needs a CUDA GPU with native BF16 support.")

    torch.manual_seed(SEED)
    device = torch.device("cuda:0")
    print(f"GPU: {torch.cuda.get_device_name(device)} | PyTorch {torch.__version__} | CUDA {torch.version.cuda}", flush=True)
    samples = read_training_samples()
    processor = AutoProcessor.from_pretrained(MODEL_ID, **MODEL_ARGS)
    processor.image_processor.size = {"shortest_edge": 4 * 32**2, "longest_edge": MAX_IMAGE_TOKENS * 32**2}
    run_dir.mkdir(parents=True, exist_ok=False)

    settings = {
        **INFERENCE_SETTINGS, "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "dataset_id": DATASET_ID, "dataset_revision": DATASET_REVISION, "train_shard": TRAIN_SHARD,
        "seed": SEED, "sample_count": SAMPLE_COUNT, "steps": STEPS, "micro_batch_size": 1,
        "dtype": "bfloat16", "attention": "sdpa", "optimizer": "AdamW", "learning_rate": LEARNING_RATE,
        "weight_decay": 0.0, "lora_rank": LORA_RANK, "lora_alpha": LORA_ALPHA, "lora_dropout": 0.0,
        "gpu": torch.cuda.get_device_name(device), "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda, "transformers_version": transformers.__version__,
        "peft_version": peft.__version__,
    }
    (run_dir / "config.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    report, saved_state = train_adapter(processor, samples, device, run_dir)
    torch.cuda.empty_cache()

    print("Reloading the adapter onto a fresh BF16 base model.", flush=True)
    reloaded = PeftModel.from_pretrained(load_base(device), run_dir / "adapter", is_trainable=False).eval()
    reloaded_state = adapter_state(reloaded)
    if reloaded_state.keys() != saved_state.keys() or any(
        not torch.equal(reloaded_state[name], tensor) for name, tensor in saved_state.items()
    ):
        raise RuntimeError("Reloaded adapter weights differ from the trained weights.")
    prediction = predict_training_example(reloaded, processor, device, samples[0])
    if prediction != report["prediction_before_reload"]:
        raise RuntimeError(f"Prediction changed after adapter reload: {report['prediction_before_reload']} -> {prediction}.")
    report.update(prediction_after_reload=prediction, adapter_reload_verified=True)
    (run_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: loss, optimizer updates, adapter reload and inference verified. Outputs: {run_dir}")


if __name__ == "__main__":
    main()
