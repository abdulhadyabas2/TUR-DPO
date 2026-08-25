#!/usr/bin/env python
"""Small, reproducible text-evaluation and answer-guardrail utility.

This script evaluates saved predictions; it does not claim to reproduce human
or LLM-judge scores.  Input records need ``prediction`` and ``reference`` and
may optionally include ``prompt``.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from turdpo.utils import compute_exact_match, compute_f1, compute_rouge_l


def extract_final_answer(text: str) -> str:
    """Extract a conservative final answer for arithmetic/short-answer tasks."""
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxed:
        return boxed[-1].strip()
    answer_marked = re.findall(r"(?:final answer|answer|therefore)\s*[:=]?\s*([^\n.]+)", text, re.I)
    if answer_marked:
        return answer_marked[-1].strip()
    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    return numbers[-1] if numbers else text.strip()


def _calibration_metrics(probabilities: List[float], labels: List[float], num_bins: int = 10) -> Dict[str, float]:
    """Compute ECE and Brier score for records carrying probabilities."""
    probs = np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)
    targets = np.asarray(labels, dtype=float)
    if len(probs) == 0:
        return {}
    ece = 0.0
    for index in range(num_bins):
        lower = index / num_bins
        upper = (index + 1) / num_bins
        mask = (probs >= lower) & (probs <= upper if index == num_bins - 1 else probs < upper)
        if mask.any():
            ece += float(mask.mean()) * abs(float(probs[mask].mean()) - float(targets[mask].mean()))
    return {
        "ece": float(ece),
        "brier": float(np.mean((probs - targets) ** 2)),
        "calibration_count": float(len(probs)),
    }


def _bootstrap_mean_ci(values: List[float], samples: int, seed: int) -> Dict[str, float]:
    """Return percentile confidence intervals for a metric vector."""
    if samples <= 0 or len(values) < 2:
        return {}
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.asarray([
        rng.choice(array, size=len(array), replace=True).mean() for _ in range(samples)
    ])
    return {
        "bootstrap_mean": float(array.mean()),
        "bootstrap_ci_low": float(np.percentile(means, 2.5)),
        "bootstrap_ci_high": float(np.percentile(means, 97.5)),
    }


def evaluate_records(
    records: List[Dict[str, Any]],
    postprocess: bool = False,
    bootstrap_samples: int = 0,
    seed: int = 42,
) -> Dict[str, float]:
    predictions = []
    references = []
    correctness = []
    probabilities = []
    structural: Dict[str, List[float]] = {
        "path_coverage": [],
        "edge_path_coverage": [],
        "contradiction_score": [],
        "peer_pressure_rate": [],
    }
    for record in records:
        prediction = str(record.get("prediction", record.get("response", "")))
        reference = str(record.get("reference", record.get("target", "")))
        if postprocess:
            prediction = extract_final_answer(prediction)
            reference = extract_final_answer(reference)
        predictions.append(prediction)
        references.append(reference)
        correct = float(compute_exact_match([prediction], [reference]))
        correctness.append(correct)
        probability = record.get("probability", record.get("confidence"))
        if probability is not None:
            try:
                probabilities.append(float(probability))
            except (TypeError, ValueError):
                pass
        features = record.get("topology_features", record.get("features", {}))
        if isinstance(features, dict):
            for key in structural:
                value = features.get(key, record.get(key))
                if value is not None:
                    try:
                        structural[key].append(float(value))
                    except (TypeError, ValueError):
                        pass
    if not predictions:
        return {"count": 0.0, "exact_match": 0.0, "f1": 0.0, "rouge_l": 0.0}
    f1_values = [compute_f1(p, r) for p, r in zip(predictions, references)]
    rouge_values = [compute_rouge_l(p, r) for p, r in zip(predictions, references)]
    metrics = {
        "count": float(len(predictions)),
        "exact_match": float(compute_exact_match(predictions, references)),
        "f1": float(np.mean(f1_values)),
        "rouge_l": float(np.mean(rouge_values)),
    }
    if len(probabilities) == len(correctness):
        metrics.update(_calibration_metrics(probabilities, correctness))
    for key, values in structural.items():
        if values:
            metrics["mean_" + key] = float(np.mean(values))
    metrics.update({
        "exact_match_ci_" + suffix: value
        for suffix, value in _bootstrap_mean_ci(correctness, bootstrap_samples, seed).items()
    })
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate TUR-DPO predictions")
    parser.add_argument("--input", required=True, help="JSON list or JSONL predictions")
    parser.add_argument("--output", default=None)
    parser.add_argument("--postprocess", action="store_true")
    parser.add_argument("--bootstrap_samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    path = Path(args.input)
    with path.open("r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()] if path.suffix == ".jsonl" else json.load(handle)
    if isinstance(records, dict):
        records = records.get("data", records.get("predictions", []))
    metrics = evaluate_records(
        records,
        postprocess=args.postprocess,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    serialized = json.dumps(metrics, indent=2)
    if args.output:
        Path(args.output).write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
