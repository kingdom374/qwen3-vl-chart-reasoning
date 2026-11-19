"""Export a ChartQA example and freeze the P0 validation sample."""

import json
import random
from collections import Counter
from io import BytesIO
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HUB_CACHE = PROJECT_ROOT / ".cache" / "huggingface" / "hub"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "chartqa_p0"
DATASET_ID = "ahmed-masry/ChartQA"
DATASET_REVISION = "af8b6f5c08c95085271561c2a3f9d15f2b5a9031"
SPLIT = "val"
PARQUET_FILE = "data/val-00000-of-00001.parquet"
SEED = 42
SAMPLES_PER_TYPE = 50
QUESTION_TYPES = ("human", "augmented")


def build_samples(parquet: pq.ParquetFile) -> tuple[list[dict], Counter]:
    # Reading only metadata columns avoids decoding every chart for sampling.
    columns = ["imgname", "query", "label", "type"]
    rows = parquet.read(columns=columns).to_pylist()
    counts = Counter(row["type"] for row in rows)
    rng = random.Random(SEED)
    selected_indices = []
    for question_type in QUESTION_TYPES:
        indices = [i for i, row in enumerate(rows) if row["type"] == question_type]
        if len(indices) < SAMPLES_PER_TYPE:
            raise ValueError(f"Too few {question_type!r} rows to sample: {counts}")
        selected_indices.extend(rng.sample(indices, SAMPLES_PER_TYPE))

    samples = []
    for index in sorted(selected_indices):
        row = rows[index]
        samples.append(
            {
                "sample_id": f"{SPLIT}:{index}",
                "row_index": index,
                "imgname": row["imgname"],
                "question": row["query"],
                "reference_answer": row["label"],
                "type": row["type"],
            }
        )
    return samples, counts


def export_chart(parquet: pq.ParquetFile, row_index: int, output: Path) -> tuple[int, int]:
    # Translate the split-wide index into a row-group offset before reading images.
    offset = row_index
    for group_index in range(parquet.num_row_groups):
        row_count = parquet.metadata.row_group(group_index).num_rows
        if offset < row_count:
            image_column = parquet.read_row_group(group_index, columns=["image"])["image"]
            image_value = image_column[offset].as_py()
            image_bytes = image_value.get("bytes") if isinstance(image_value, dict) else image_value
            if not isinstance(image_bytes, (bytes, bytearray, list)) or not image_bytes:
                raise ValueError(f"Missing embedded image bytes for {SPLIT}:{row_index}")
            with Image.open(BytesIO(bytes(image_bytes))) as image:
                rgb = image.convert("RGB")
                rgb.save(output)
                return rgb.size
        offset -= row_count
    raise IndexError(f"Row {row_index} is outside split {SPLIT!r}")


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    revision = HfApi().dataset_info(DATASET_ID, revision=DATASET_REVISION).sha
    if not revision:
        raise RuntimeError(f"Could not resolve a commit for {DATASET_ID}")
    parquet_path = hf_hub_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        filename=PARQUET_FILE,
        revision=revision,
        cache_dir=HUB_CACHE,
    )
    parquet = pq.ParquetFile(parquet_path)
    samples, counts = build_samples(parquet)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    example_image = OUTPUT_DIR / "example.png"
    example = samples[0]
    width, height = export_chart(parquet, example["row_index"], example_image)
    example_info = {**example, "image_file": example_image.name, "image_size": [width, height]}
    write_json(OUTPUT_DIR / "example.json", example_info)
    write_json(
        OUTPUT_DIR / "manifest.json",
        {
            "dataset_id": DATASET_ID,
            "dataset_revision": revision,
            "split": SPLIT,
            "parquet_file": PARQUET_FILE,
            "split_type_counts": dict(counts),
            "seed": SEED,
            "samples_per_type": SAMPLES_PER_TYPE,
            "samples": samples,
        },
    )

    print(f"ChartQA {SPLIT} | commit {revision} | {sum(counts.values())} rows")
    print(f"Source counts: {dict(counts)}")
    print(f"Sample counts: {dict(Counter(row['type'] for row in samples))}")
    print(json.dumps(example_info, ensure_ascii=False, indent=2))
    print(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
