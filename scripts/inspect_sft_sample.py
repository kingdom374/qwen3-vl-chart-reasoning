"""Inspect the Qwen3-VL training inputs and loss mask for ChartQA train:0."""

import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoProcessor

from inspect_chartqa import DATASET_ID, DATASET_REVISION
from run_baseline import HUB_CACHE, MODEL_ID, MODEL_REVISION, PROMPT_SUFFIX
from run_baseline_batch import chart_image


TRAIN_SHARD = "data/train-00000-of-00003.parquet"
MAX_IMAGE_TOKENS = 512
IGNORE_LABEL = -100
END_OF_TURN = "<|im_end|>"


def read_training_sample() -> dict:
    print(f"Reading {DATASET_ID}@{DATASET_REVISION} | {TRAIN_SHARD}", flush=True)
    path = hf_hub_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        filename=TRAIN_SHARD,
        revision=DATASET_REVISION,
        cache_dir=HUB_CACHE,
    )
    with pq.ParquetFile(path) as parquet:
        batches = parquet.iter_batches(batch_size=1, columns=["image", "imgname", "query", "label", "type"])
        first_batch = next(batches, None)
        if first_batch is None:
            raise ValueError(f"Empty training shard: {TRAIN_SHARD}")
        return first_batch.to_pylist()[0]


def encode_training_sample(processor, image, question: str, answer: str):
    user_messages = [{
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": f"{question}\n{PROMPT_SUFFIX}"}],
    }]
    full_messages = [*user_messages, {"role": "assistant", "content": answer}]
    prefix_text = processor.apply_chat_template(user_messages, tokenize=False, add_generation_prompt=True)
    full_text = processor.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)
    prefix = processor(text=[prefix_text], images=[image], return_tensors="pt")
    batch = processor(text=[full_text], images=[image], return_tensors="pt")

    input_ids = batch["input_ids"]
    answer_start = prefix["input_ids"].shape[1]
    # Compare after multimodal processing, since it expands the image placeholder.
    if not torch.equal(input_ids[:, :answer_start], prefix["input_ids"]):
        raise ValueError("The encoded inference prefix does not match the training sequence.")

    tokenizer = processor.tokenizer
    end_id = tokenizer.convert_tokens_to_ids(END_OF_TURN)
    end_positions = (input_ids[0, answer_start:] == end_id).nonzero(as_tuple=True)[0]
    if end_positions.numel() != 1:
        raise ValueError("Expected exactly one assistant end-of-turn token after the prefix.")
    answer_end = answer_start + end_positions.item()
    answer_ids = input_ids[0, answer_start:answer_end]
    decoded_answer = tokenizer.decode(answer_ids.tolist(), clean_up_tokenization_spaces=False)
    if answer_ids.numel() == 0 or decoded_answer != answer:
        raise ValueError(f"Decoded answer {decoded_answer!r} differs from ChartQA label {answer!r}.")

    labels = torch.full_like(input_ids, IGNORE_LABEL)
    labels[:, answer_start:answer_end + 1] = input_ids[:, answer_start:answer_end + 1]
    labels.masked_fill_(batch["attention_mask"] == 0, IGNORE_LABEL)
    batch["labels"] = labels
    return batch, answer_start, answer_end


def main() -> None:
    sample = read_training_sample()
    image = chart_image(sample["image"])
    print(f"Loading processor files for {MODEL_ID}@{MODEL_REVISION}", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION, cache_dir=HUB_CACHE)
    processor.image_processor.size = {
        "shortest_edge": 4 * 32**2,
        "longest_edge": MAX_IMAGE_TOKENS * 32**2,
    }
    batch, answer_start, answer_end = encode_training_sample(processor, image, sample["query"], sample["label"])

    input_ids = batch["input_ids"][0]
    labels = batch["labels"][0]
    image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    image_positions = input_ids == image_token_id
    image_token_count = int(image_positions.sum().item())
    grid = batch["image_grid_thw"]
    merge_size = processor.image_processor.merge_size
    expected_image_tokens = grid.prod(dim=1).sum().item() // merge_size**2
    if image_token_count != expected_image_tokens or not 0 < image_token_count <= MAX_IMAGE_TOKENS:
        raise ValueError(f"Unexpected visual token count: {image_token_count}, grid implies {expected_image_tokens}.")
    if torch.any(labels[image_positions] != IGNORE_LABEL):
        raise ValueError("Image placeholders must be masked from the loss.")
    supervised = (labels != IGNORE_LABEL).nonzero(as_tuple=True)[0].tolist()
    if supervised != list(range(answer_start, answer_end + 1)):
        raise ValueError("Expected supervision on every answer token and its end-of-turn token only.")

    print(f"\nSample: train:0 | {sample['type']} | {sample['imgname']} | image size: {image.size}")
    print(f"Question: {sample['query']}")
    print(f"Answer: {sample['label']!r}")
    print(f"input_ids.shape: {tuple(batch['input_ids'].shape)}")
    print(f"pixel_values.shape: {tuple(batch['pixel_values'].shape)}")
    print(f"image_grid_thw: {grid.tolist()} | visual tokens: {image_token_count}")
    print(f"input_ids: {input_ids.tolist()}")
    print(f"attention_mask: {batch['attention_mask'][0].tolist()}")
    print(f"labels: {labels.tolist()}")
    print(f"Supervised label positions (0-based): {supervised}")
    print(f"First answer token uses logits at position {answer_start - 1} to predict label at {answer_start}.")
    print("\nPosition | Token ID | Label | Token text")
    for index in range(max(0, answer_start - 4), min(input_ids.numel(), answer_end + 3)):
        token_id = input_ids[index].item()
        token_text = processor.tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
        print(f"{index:8} | {token_id:8} | {labels[index].item():5} | {token_text!r}")
    print("PASS: multimodal inputs, prompt boundary, and answer-only loss mask verified.")


if __name__ == "__main__":
    main()
