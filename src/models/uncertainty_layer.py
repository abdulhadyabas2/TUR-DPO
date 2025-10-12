from __future__ import annotations
from typing import Dict


def combine_uncertainty(u_epi: float, u_ale: float, w_epi: float = 0.5, w_ale: float = 0.5) -> float:
    return float(max(0.0, w_epi * u_epi + w_ale * u_ale))


def weight_from_uncertainty(u_pos: float, u_neg: float, tau_w: float = 1.2, w_min: float = 0.05) -> float:
    u_bar = 0.5 * (u_pos + u_neg)
    w = tau_w / (1.0 + u_bar)
    return float(max(w_min, min(1.0, w)))
