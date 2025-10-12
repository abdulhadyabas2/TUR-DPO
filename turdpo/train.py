from __future__ import annotations
import argparse
import json
from typing import Dict, Any
from dataclasses import dataclass

from .topology import extract_topology, topology_score, epistemic_uncertainty, aleatoric_uncertainty
from .semantic import semantic_score
from .reward import shaped_reward
from .losses import tur_dpo_loss
from .models import load_causal_lm, batch_logprobs


@dataclass
class Pair:
    x: str
    y_pos: str
    y_neg: str
    ref_answer: str | None = None


def compute_weight(u_pos: float, u_neg: float, tau_w: float = 1.2, w_min: float = 0.05) -> float:
    u_bar = 0.5 * (u_pos + u_neg)
    w = tau_w / (1.0 + u_bar)
    return float(max(w_min, min(1.0, w)))


def run_epoch(pairs, args) -> Dict[str, Any]:
    tok_pol, pol, device = load_causal_lm(args.policy)
    if args.reference:
        tok_ref, ref, _ = load_causal_lm(args.reference, device=device)
    else:
        tok_ref, ref = tok_pol, pol  # fallback to same as policy for demo

    prompts = [p.x for p in pairs for _ in (0, 1)]
    responses = [p.y_pos for p in pairs] + [p.y_neg for p in pairs]

    lp_pol = batch_logprobs(tok_pol, pol, device, prompts, responses)
    lp_ref = batch_logprobs(tok_ref, ref, device, prompts, responses)

    losses = []
    for i, pr in enumerate(pairs):
        # extract topologies (single sample per side for demo; could do K>1 and aggregate)
        Gp = extract_topology(pr.y_pos)
        Gn = extract_topology(pr.y_neg)
        topo_p = topology_score(Gp)
        topo_n = topology_score(Gn)

        # uncertainties
        u_p = 0.5 * epistemic_uncertainty([Gp]) + 0.5 * aleatoric_uncertainty(Gp)
        u_n = 0.5 * epistemic_uncertainty([Gn]) + 0.5 * aleatoric_uncertainty(Gn)
        w = compute_weight(u_p, u_n, tau_w=args.tau_w, w_min=args.w_min)

        # semantics
        sem_p = semantic_score(pr.x, pr.y_pos, pr.ref_answer)
        sem_n = semantic_score(pr.x, pr.y_neg, pr.ref_answer)

        # shaped reward
        rp = shaped_reward(sem_p["s_sem"], topo_p["s_topo"], (u_p), a=args.a, gamma_sem=args.gamma_sem,
                           gamma_topo=args.gamma_topo, b_sem=args.b_sem, b_topo=args.b_topo, lam=args.lam)
        rn = shaped_reward(sem_n["s_sem"], topo_n["s_topo"], (u_n), a=args.a, gamma_sem=args.gamma_sem,
                           gamma_topo=args.gamma_topo, b_sem=args.b_sem, b_topo=args.b_topo, lam=args.lam)

        d_lp = lp_pol[i] - lp_pol[i + len(pairs)]
        d_lr = lp_ref[i] - lp_ref[i + len(pairs)]
        d_r = rp["r"] - rn["r"]

        out = tur_dpo_loss(d_lp, d_lr, d_r, beta=args.beta, gamma=args.gamma, w=w)
        losses.append(out["loss"])

    return {"mean_loss": float(sum(losses) / max(1, len(losses)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/sample_pairs.jsonl")
    ap.add_argument("--policy", type=str, default="sshleifer/tiny-gpt2")
    ap.add_argument("--reference", type=str, default="sshleifer/tiny-gpt2")

    # reward and loss params
    ap.add_argument("--beta", type=float, default=2.0)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--a", type=float, default=0.6)
    ap.add_argument("--gamma_sem", type=float, default=1.0)
    ap.add_argument("--gamma_topo", type=float, default=1.0)
    ap.add_argument("--b_sem", type=float, default=0.0)
    ap.add_argument("--b_topo", type=float, default=0.0)
    ap.add_argument("--lam", type=float, default=0.5)

    ap.add_argument("--tau_w", type=float, default=1.2)
    ap.add_argument("--w_min", type=float, default=0.05)

    args = ap.parse_args()

    pairs = []
    with open(args.data, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            pairs.append(Pair(x=obj["prompt"], y_pos=obj["chosen"], y_neg=obj["rejected"], ref_answer=obj.get("ref_answer")))

    stats = run_epoch(pairs, args)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
