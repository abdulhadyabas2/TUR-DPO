#!/usr/bin/env python
"""Aggregate human/LLM pair judgments and agreement diagnostics."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def aggregate_judgments(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_pair: Dict[str, List[float]] = defaultdict(list)
    for record in records:
        pair_id = str(record.get("pair_id", record.get("id", "")))
        label = record.get("label", record.get("winner"))
        if not pair_id or label is None:
            continue
        if isinstance(label, str):
            label = {"chosen": 1.0, "rejected": 0.0, "tie": 0.5}.get(label.lower())
        if label is None:
            continue
        by_pair[pair_id].append(float(label))
    consensus = {
        pair_id: float(np.mean(labels))
        for pair_id, labels in by_pair.items()
        if labels
    }
    agreement = []
    for labels in by_pair.values():
        if len(labels) > 1:
            agreement.append(float(np.std(labels) == 0.0))
    return {
        "pairs": float(len(consensus)),
        "judgments": float(sum(len(labels) for labels in by_pair.values())),
        "unanimous_pair_fraction": float(np.mean(agreement)) if agreement else 0.0,
        "consensus": consensus,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate human or LLM judgments")
    parser.add_argument("--input", required=True, help="JSON/JSONL records with pair_id and label/winner")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    path = Path(args.input)
    with path.open("r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()] if path.suffix.lower() == ".jsonl" else json.load(handle)
    if isinstance(records, dict):
        records = records.get("data", records.get("judgments", []))
    result = aggregate_judgments(records)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
