from __future__ import annotations
from typing import Dict
import math


def sigmoid(z: float) -> float:
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    else:
        ez = math.exp(z)
        return ez / (1.0 + ez)


def tur_dpo_loss(delta_logp: float,
                 delta_logp_ref: float,
                 delta_r: float,
                 beta: float = 2.0,
                 gamma: float = 1.0,
                 w: float = 1.0) -> Dict[str, float]:
    """
    L = - w * log sigma( beta*(delta_logp - delta_logp_ref) + gamma*delta_r )
    Returns dict with loss and margin.
    """
    margin = beta * (delta_logp - delta_logp_ref) + gamma * delta_r
    sig = sigmoid(margin)
    loss = -w * math.log(max(1e-12, sig))
    return {"loss": float(loss), "margin": float(margin), "sigmoid": float(sig)}
