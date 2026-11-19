"""Score the saved ChartQA P0 predictions without loading the model."""

import csv
import hashlib
import json
import re
from decimal import Decimal
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parents[1] / "outputs" / "chartqa_p0"
RUN_DIR = DATA_DIR / "batch_bf16_512"
TOLERANCE = Decimal("0.05")
NUMERIC_ANSWER = re.compile(
    r"([+-]?(?:(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?|\.[0-9]+)) *(%)?"
)


def normalize(answer: str) -> str:
    return " ".join(answer.casefold().split())


def parse_number(answer: str) -> tuple[Decimal, bool] | None:
    match = NUMERIC_ANSWER.fullmatch(answer.replace("−", "-"))
    if match is None:
        return None
    value, percent = match.groups()
    return Decimal(value.replace(",", "")), percent is not None


def score(prediction: str, reference: str) -> tuple[bool, bool]:
    exact_match = bool(prediction) and prediction == reference
    predicted_number = parse_number(prediction)
    reference_number = parse_number(reference)
    if predicted_number is None or reference_number is None:
        return exact_match, exact_match

    predicted_value, predicted_percent = predicted_number
    reference_value, reference_percent = reference_number
    same_unit = predicted_percent == reference_percent
    within_tolerance = abs(predicted_value - reference_value) <= abs(reference_value) * TOLERANCE
    return exact_match, same_unit and within_tolerance


def check_scoring_rules() -> None:
    cases = [
        ("   New   York ", "new york", (True, True)),
        ("105", "100", (False, True)),
        ("105.01", "100", (False, False)),
        ("-95", "-100", (False, True)),
        ("0.00", "0", (False, True)),
        ("0.01", "0", (False, False)),
        ("50%", "50", (False, False)),
        ("1,050", "1000", (False, True)),
        ("", "64", (False, False)),
        ("The answer is 64", "64", (False, False)),
    ]
    for prediction, reference, expected in cases:
        if score(normalize(prediction), normalize(reference)) != expected:
            raise RuntimeError(f"Scoring rule check failed for {prediction!r} and {reference!r}.")


def read_predictions(path: Path) -> dict[str, dict]:
    predictions = {}
    if not path.exists():
        raise FileNotFoundError(f"Run scripts/run_baseline_batch.py first. Missing {path}")
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            record = json.loads(line)
            sample_id = record["sample_id"]
            if sample_id in predictions:
                raise ValueError(f"Duplicate sample ID {sample_id} at line {line_number}.")
            predictions[sample_id] = record
    return predictions


def evaluate(samples: list[dict], predictions: dict[str, dict]) -> list[dict]:
    unknown_ids = predictions.keys() - {sample["sample_id"] for sample in samples}
    if unknown_ids:
        raise ValueError(f"Prediction IDs absent from the manifest: {sorted(unknown_ids)}")

    rows = []
    for sample in samples:
        sample_id = sample["sample_id"]
        prediction = predictions.get(sample_id)
        reference = normalize(sample["reference_answer"])
        if prediction is None:
            rows.append({
                **sample, "status": "missing", "raw_prediction": None,
                "normalized_prediction": None, "normalized_reference": reference,
                "exact_match": False, "relaxed_match": False,
            })
            continue
        if prediction["status"] != "ok" or any(prediction[key] != value for key, value in sample.items()):
            raise ValueError(f"Prediction metadata does not match the manifest for {sample_id}.")
        normalized = normalize(prediction["raw_prediction"])
        exact, relaxed = score(normalized, reference)
        rows.append({
            **sample, "status": "ok", "raw_prediction": prediction["raw_prediction"],
            "normalized_prediction": normalized, "normalized_reference": reference,
            "exact_match": exact, "relaxed_match": relaxed,
        })
    return rows


def summarize(rows: list[dict]) -> dict:
    total = len(rows)
    exact_correct = sum(row["exact_match"] for row in rows)
    relaxed_correct = sum(row["relaxed_match"] for row in rows)
    return {
        "planned": total,
        "missing_predictions": sum(row["status"] == "missing" for row in rows),
        "exact_correct": exact_correct,
        "exact_accuracy_pct": round(100 * exact_correct / total, 2),
        "relaxed_correct": relaxed_correct,
        "relaxed_accuracy_pct": round(100 * relaxed_correct / total, 2),
    }


def main() -> None:
    check_scoring_rules()
    manifest_bytes = (DATA_DIR / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    settings = json.loads((RUN_DIR / "config.json").read_text(encoding="utf-8"))
    if hashlib.sha256(manifest_bytes).hexdigest() != settings["manifest_sha256"]:
        raise ValueError("The current manifest differs from the one used for inference.")
    samples = manifest["samples"]
    if len(samples) != 100 or len({sample["sample_id"] for sample in samples}) != 100:
        raise ValueError("Expected the fixed manifest with 100 unique sample IDs.")
    rows = evaluate(samples, read_predictions(RUN_DIR / "predictions.jsonl"))

    summary = {
        "model_id": settings["model_id"],
        "model_revision": settings["model_revision"],
        "dataset_revision": manifest["dataset_revision"],
        "split": manifest["split"],
        "seed": manifest["seed"],
        "relative_numeric_tolerance": str(TOLERANCE),
        "overall": summarize(rows),
        "by_type": {
            kind: summarize([row for row in rows if row["type"] == kind])
            for kind in sorted({row["type"] for row in rows})
        },
    }
    with (RUN_DIR / "scored_predictions.jsonl").open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")

    bad_cases = [row for row in rows if row["status"] == "ok" and not row["relaxed_match"]]
    columns = ["sample_id", "row_index", "imgname", "type", "question", "reference_answer", "raw_prediction"]
    with (RUN_DIR / "bad_cases.csv").open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(bad_cases)

    report = json.dumps(summary, ensure_ascii=False, indent=2)
    (RUN_DIR / "summary.json").write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"Bad cases: {len(bad_cases)} | Outputs: {RUN_DIR}")


if __name__ == "__main__":
    main()
