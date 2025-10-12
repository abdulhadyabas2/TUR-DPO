from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any
import math

from .topology_graph import extract_graph, topology_score, aleatoric_uncertainty, epistemic_uncertainty
from .uncertainty_layer import combine_uncertainty, weight_from_uncertainty
from .reward_estimator import shaped_reward, tur_dpo_margin


def _sigmoid(z: float) -> float:
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


@dataclass
class Pair:
    prompt: str
    chosen: str
    rejected: str
    ref_answer: str | None = None


class TURDPOTrainer:
    def __init__(self, beta=2.0, gamma=1.0, a=0.6, gamma_sem=1.0, gamma_topo=1.0, b_sem=0.0, b_topo=0.0, lam=0.5,
                 tau_w=1.2, w_min=0.05):
        self.beta = beta
        self.gamma = gamma
        self.a = a
        self.gamma_sem = gamma_sem
        self.gamma_topo = gamma_topo
        self.b_sem = b_sem
        self.b_topo = b_topo
        self.lam = lam
        self.tau_w = tau_w
        self.w_min = w_min

    def semantic_score(self, prompt: str, response: str, ref_answer: str | None) -> float:
        # placeholder: +0.3 if contains numbers, +0.2 if cites a cue, -0.3 if many uppercase words
        import re
        has_num = bool(re.search(r"\d", response))
        has_cue = bool(re.search(r"\b(because|therefore|hence|accordingly)\b", response, flags=re.I))
        caps = sum(1 for t in re.findall(r"[A-Za-z]+", response) if t[:1].isupper())
        q_fact = 0.5 + 0.3 * has_num + 0.2 * has_cue
        q_task = 1.0 if (ref_answer and response.strip().lower().find(ref_answer.strip().lower()) != -1) else 0.0
        q_hall = min(1.0, max(0.0, (caps - 3) / 10.0))
        return float(max(0.0, min(1.0, 0.7 * q_fact + 0.3 * q_task - 0.4 * q_hall)))

    def compute_pair(self, pair: Pair, logp_pos: float, logp_neg: float, logp_ref_pos: float, logp_ref_neg: float) -> Dict[str, Any]:
        Gp = extract_graph(pair.chosen)
        Gn = extract_graph(pair.rejected)
        s_top_p = topology_score(Gp)["s_topo"]
        s_top_n = topology_score(Gn)["s_topo"]
        u_p = combine_uncertainty(epistemic_uncertainty([Gp]), aleatoric_uncertainty(Gp))
        u_n = combine_uncertainty(epistemic_uncertainty([Gn]), aleatoric_uncertainty(Gn))
        w = weight_from_uncertainty(u_p, u_n, self.tau_w, self.w_min)
        s_sem_p = self.semantic_score(pair.prompt, pair.chosen, pair.ref_answer)
        s_sem_n = self.semantic_score(pair.prompt, pair.rejected, pair.ref_answer)
        r_p = shaped_reward(s_sem_p, s_top_p, u_p, self.a, self.gamma_sem, self.gamma_topo, self.b_sem, self.b_topo, self.lam)["r"]
        r_n = shaped_reward(s_sem_n, s_top_n, u_n, self.a, self.gamma_sem, self.gamma_topo, self.b_sem, self.b_topo, self.lam)["r"]
        margin = tur_dpo_margin(logp_pos - logp_neg, logp_ref_pos - logp_ref_neg, r_p - r_n, self.beta, self.gamma)
        prob = _sigmoid(margin)
        loss = -w * math.log(max(1e-12, prob))
        return {"loss": float(loss), "w": float(w), "margin": float(margin), "prob": float(prob)}
