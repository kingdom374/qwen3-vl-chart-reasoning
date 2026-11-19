"""Run the Qwen3-VL baseline on the exported ChartQA example."""

import argparse
import json
from pathlib import Path

import torch
from huggingface_hub import HfApi
from PIL import Image
from transformers import AutoProcessor, GenerationConfig, Qwen3VLForConditionalGeneration


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HUB_CACHE = PROJECT_ROOT / ".cache" / "huggingface" / "hub"
DATA_DIR = PROJECT_ROOT / "outputs" / "chartqa_p0"
MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
MODEL_REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"
PROMPT_SUFFIX = "Answer with only the final answer."
MAX_NEW_TOKENS = 32


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-image-tokens", type=int, choices=(256, 512), default=512)
    args = parser.parse_args()

    example_path = DATA_DIR / "example.json"
    if not example_path.is_file():
        raise SystemExit("Run scripts/inspect_chartqa.py to export the ChartQA example first.")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("This baseline requires a CUDA GPU with BF16 support.")

    example = json.loads(example_path.read_text(encoding="utf-8"))
    manifest = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
    with Image.open(DATA_DIR / example["image_file"]) as source:
        image = source.convert("RGB")

    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats(device)
    revision = HfApi().model_info(MODEL_ID, revision=MODEL_REVISION).sha
    if not revision:
        raise SystemExit(f"Could not resolve a commit for {MODEL_ID}.")
    print(f"Loading {MODEL_ID} at {revision} on {torch.cuda.get_device_name(device)}", flush=True)

    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=revision, cache_dir=HUB_CACHE)
    # Qwen3-VL merges 16-pixel patches into 32-pixel spatial units for the LLM.
    processor.image_processor.size = {
        "shortest_edge": 4 * 32**2,
        "longest_edge": args.max_image_tokens * 32**2,
    }
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        revision=revision,
        cache_dir=HUB_CACHE,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": 0},
    ).eval()

    prompt = f"{example['question']}\n{PROMPT_SUFFIX}"
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[chat_text], images=[image], return_tensors="pt").to(device)
    input_tokens = inputs.input_ids.shape[1]
    image_grid = inputs.image_grid_thw[0].tolist()
    image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    image_tokens = int((inputs.input_ids == image_token_id).sum().item())
    if not 0 < image_tokens <= args.max_image_tokens:
        raise RuntimeError(f"Unexpected image token count: {image_tokens}")
    print(f"Input: {input_tokens} tokens, including {image_tokens} image tokens", flush=True)

    defaults = model.generation_config
    generation = GenerationConfig(
        bos_token_id=defaults.bos_token_id,
        eos_token_id=defaults.eos_token_id,
        pad_token_id=defaults.pad_token_id,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
    )
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs, generation_config=generation, use_model_defaults=False
        )

    # A decoder-only model returns the input prefix followed by new tokens.
    new_ids = output_ids[:, input_tokens:]
    prediction = processor.batch_decode(
        new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    torch.cuda.synchronize(device)

    result = {
        "sample_id": example["sample_id"],
        "imgname": example["imgname"],
        "dataset_revision": manifest["dataset_revision"],
        "question": example["question"],
        "reference_answer": example["reference_answer"],
        "raw_prediction": prediction,
        "model_id": MODEL_ID,
        "model_revision": revision,
        "dtype": "bfloat16",
        "attention": "sdpa",
        "prompt_suffix": PROMPT_SUFFIX,
        "do_sample": False,
        "max_image_tokens": args.max_image_tokens,
        "max_new_tokens": MAX_NEW_TOKENS,
        "input_ids_shape": list(inputs.input_ids.shape),
        "pixel_values_shape": list(inputs.pixel_values.shape),
        "image_grid_thw": image_grid,
        "image_tokens": image_tokens,
        "generated_tokens": new_ids.shape[1],
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated(device) / 1024**2, 1),
        "peak_reserved_mib": round(torch.cuda.max_memory_reserved(device) / 1024**2, 1),
    }
    result_path = DATA_DIR / f"smoke_bf16_{args.max_image_tokens}.json"
    report = json.dumps(result, ensure_ascii=False, indent=2)
    result_path.write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"Saved: {result_path}")


if __name__ == "__main__":
    main()
