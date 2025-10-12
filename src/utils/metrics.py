from __future__ import annotations
from typing import List, Dict, Optional
import numpy as np


def mean_std(xs: List[float]):
    if not xs:
        return 0.0, 0.0
    a = np.array(xs, dtype=float)
    return float(a.mean()), float(a.std(ddof=1) if len(a) > 1 else 0.0)


def win_rate(margins: List[float]) -> float:
    if not margins:
        return 0.0
    wins = sum(1 for m in margins if m > 0)
    return float(wins / len(margins))


def expected_calibration_error(probs: List[float], labels: List[int], n_bins: int = 10) -> float:
    if not probs:
        return 0.0
    probs_np = np.clip(np.array(probs, dtype=float), 1e-8, 1 - 1e-8)
    labels_np = np.array(labels, dtype=int)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    N = len(probs_np)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (probs_np >= lo) & (probs_np < hi if i < n_bins - 1 else probs_np <= hi)
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        avg_p = float(probs_np[idx].mean())
        avg_a = float((labels_np[idx] == 1).mean())
        ece += (len(idx) / N) * abs(avg_p - avg_a)
    return float(ece)


def brier_score(probs: List[float], labels: List[int]) -> float:
    if not probs:
        return 0.0
    p = np.array(probs, dtype=float)
    y = np.array(labels, dtype=float)
    return float(np.mean((p - y) ** 2))


def summarize_pairwise_metrics(margins: List[float], seq_kl_pos: Optional[List[float]] = None, seq_kl_neg: Optional[List[float]] = None) -> Dict[str, float]:
    # probs for chosen being preferred
    import math
    probs = [1.0 / (1.0 + math.exp(-m)) for m in margins]
    labels = [1] * len(margins)
    wr = win_rate(margins)
    ece = expected_calibration_error(probs, labels, n_bins=10)
    brier = brier_score(probs, labels)
    out = {"win_rate": wr, "ece": ece, "brier": brier}
    if seq_kl_pos is not None and seq_kl_neg is not None and len(seq_kl_pos) == len(margins):
        # Report simple averages of sequence-level KL estimates (lp_pol - lp_ref)
        out["avg_seq_kl_pos"] = float(np.mean(seq_kl_pos))
        out["avg_seq_kl_neg"] = float(np.mean(seq_kl_neg))
        out["avg_seq_kl_mean"] = float(np.mean(0.5 * (np.array(seq_kl_pos) + np.array(seq_kl_neg))))
    return out
