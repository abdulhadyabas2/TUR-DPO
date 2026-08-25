#!/usr/bin/env python
"""Aggregate scalar metrics across experiment runs and random seeds."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def _collect(root: Path) -> Dict[str, List[float]]:
    grouped: Dict[str, List[float]] = {}
    for path in root.rglob("training_results.json"):
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(result, dict):
            continue
        variant = path.parent.parent.name if path.parent.name.startswith("seed_") else path.parent.name
        for key, value in result.items():
            if isinstance(value, (int, float)) and np.isfinite(value):
                grouped.setdefault(f"{variant}.{key}", []).append(float(value))
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate TUR-DPO training results")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    grouped = _collect(Path(args.input_dir))
    summary: Dict[str, Any] = {}
    for key, values in sorted(grouped.items()):
        array = np.asarray(values, dtype=float)
        summary[key] = {
            "n": int(len(array)),
            "mean": float(array.mean()),
            "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
            "values": [float(value) for value in array],
        }
    Path(args.output).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
