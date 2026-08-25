#!/usr/bin/env python
"""Normalize common benchmark JSON schemas without fabricating preference labels."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _read_records(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        value = (
            [json.loads(line) for line in handle if line.strip()]
            if path.suffix.lower() == ".jsonl"
            else json.load(handle)
        )
    if isinstance(value, dict):
        value = value.get("data", value.get("records", value.get("examples", [])))
    if not isinstance(value, list):
        raise ValueError("Input must be a JSON list, JSONL stream, or wrapped data/records root")
    return [record for record in value if isinstance(record, dict)]


def _first(record: Dict[str, Any], keys: List[str]) -> Optional[Any]:
    for key in keys:
        if record.get(key) is not None:
            return record[key]
    return None


def normalize_records(records: List[Dict[str, Any]], dataset_name: str = "custom") -> List[Dict[str, Any]]:
    """Map common QA/preference schemas to the repository's canonical fields."""
    normalized = []
    for index, record in enumerate(records):
        prompt = _first(record, ["prompt", "question", "instruction", "input", "query"])
        if prompt is None:
            raise ValueError(f"Record {index} has no prompt/question/instruction/input/query field")
        output: Dict[str, Any] = {
            "id": record.get("id", f"{dataset_name}_{index}"),
            "prompt": str(prompt),
            "metadata": dict(record.get("metadata", {})) if isinstance(record.get("metadata"), dict) else {},
        }
        output["metadata"]["dataset"] = dataset_name
        candidates = _first(record, ["candidates", "responses", "outputs", "completions"])
        if isinstance(candidates, list) and len(candidates) >= 2:
            output["candidates"] = candidates
        chosen = _first(record, ["chosen", "preferred", "chosen_response", "answer", "response", "solution"])
        rejected = _first(record, ["rejected", "dispreferred", "rejected_response"])
        if chosen is not None:
            output["chosen"] = str(chosen)
        if rejected is not None:
            output["rejected"] = str(rejected)
        if "candidates" not in output and not {"chosen", "rejected"}.issubset(output):
            raise ValueError(
                f"Record {index} has no preference pair or at least two candidates; "
                "normalization will not invent a negative response"
            )
        for key in ("task_score_chosen", "task_score_rejected", "calibration_target_chosen", "calibration_target_rejected"):
            if key in record:
                output[key] = record[key]
        normalized.append(output)
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize benchmark records for TUR-DPO")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset_name", default="custom")
    args = parser.parse_args()
    records = normalize_records(_read_records(Path(args.input)), args.dataset_name)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".jsonl":
        output.write_text("\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n", encoding="utf-8")
    else:
        output.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"count": len(records), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
