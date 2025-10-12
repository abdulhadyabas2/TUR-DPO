from __future__ import annotations
from typing import Dict


def shaped_reward(s_sem: float, s_topo: float, u_total: float,
                  a: float = 0.6,
                  gamma_sem: float = 1.0,
                  gamma_topo: float = 1.0,
                  b_sem: float = 0.0,
                  b_topo: float = 0.0,
                  lam: float = 0.5) -> Dict[str, float]:
    f_sem = gamma_sem * s_sem + b_sem
    f_top = gamma_topo * s_topo + b_topo
    r = a * f_sem + (1 - a) * f_top - lam * u_total
    return {"f_sem": float(f_sem), "f_top": float(f_top), "r": float(r)}


def tur_dpo_margin(delta_logp: float, delta_logp_ref: float, delta_r: float, beta: float = 2.0, gamma: float = 1.0) -> float:
    return float(beta * (delta_logp - delta_logp_ref) + gamma * delta_r)
