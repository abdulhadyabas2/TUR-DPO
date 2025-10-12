from __future__ import annotations
from typing import Tuple
import torch
import torch.nn.functional as F

from src.models.tur_dpo_trainer import Pair, TURDPOTrainer
from src.models.reward_estimator import shaped_reward
from src.models.topology_graph import extract_graph, topology_score, aleatoric_uncertainty, epistemic_uncertainty


def _build_io(tok, prompt: str, response: str, device: str):
    enc = tok(prompt, return_tensors="pt")
    dec = tok(response, return_tensors="pt", add_special_tokens=False)
    input_ids = torch.cat([enc.input_ids, dec.input_ids], dim=1).to(device)
    P = enc.input_ids.shape[1]
    return input_ids, P


def _sum_response_logprob(model, input_ids, P):
    out = model(input_ids=input_ids)
    logits = out.logits[:, :-1, :]
    labels = input_ids[:, 1:]
    mask = torch.zeros_like(labels)
    mask[:, P - 1 :] = 1
    logprobs = F.log_softmax(logits, dim=-1)
    token_lp = (logprobs.gather(-1, labels.unsqueeze(-1)).squeeze(-1) * mask).sum()  # scalar tensor
    return token_lp, logits, labels, mask


def _token_kl(logits_pol, logits_ref, mask) -> float:
    # KL(P||Q) averaged over masked positions
    with torch.no_grad():
        logP = F.log_softmax(logits_pol, dim=-1)
        logQ = F.log_softmax(logits_ref, dim=-1)
        P = logP.exp()
        kl_pos = (P * (logP - logQ)).sum(dim=-1)  # per position
        # mask positions
        valid = mask[:, 1:].float()  # align with logits shape
        total = valid.sum().clamp_min(1.0)
        kl_mean = (kl_pos * valid).sum() / total
        return float(kl_mean.item())


def turdpo_pair_loss_torch(tok, pol, ref, device: str, trainer: TURDPOTrainer, pair: Pair, beta: float, gamma: float) -> Tuple[torch.Tensor, float, float, float, float]:
    # Build pos and neg, compute policy and ref logprobs
    # Positive
    in_pos, P_pos = _build_io(tok, pair.prompt, pair.chosen, device)
    lp_pos, logits_pos_pol, labels_pos, mask_pos = _sum_response_logprob(pol, in_pos, P_pos)
    # Negative
    in_neg, P_neg = _build_io(tok, pair.prompt, pair.rejected, device)
    lp_neg, logits_neg_pol, labels_neg, mask_neg = _sum_response_logprob(pol, in_neg, P_neg)

    with torch.no_grad():
        lp_ref_pos, logits_pos_ref, _, _ = _sum_response_logprob(ref, in_pos, P_pos)
        lp_ref_neg, logits_neg_ref, _, _ = _sum_response_logprob(ref, in_neg, P_neg)

    # Compute shaped reward delta and weight using trainer heuristics
    Gp = extract_graph(pair.chosen)
    Gn = extract_graph(pair.rejected)
    s_top_p = topology_score(Gp)["s_topo"]
    s_top_n = topology_score(Gn)["s_topo"]
    u_p = 0.5 * epistemic_uncertainty([Gp]) + 0.5 * aleatoric_uncertainty(Gp)
    u_n = 0.5 * epistemic_uncertainty([Gn]) + 0.5 * aleatoric_uncertainty(Gn)
    # weight
    w = trainer.tau_w / (1.0 + 0.5 * (u_p + u_n))
    w = max(trainer.w_min, min(1.0, w))
    s_sem_p = trainer.semantic_score(pair.prompt, pair.chosen, pair.ref_answer)
    s_sem_n = trainer.semantic_score(pair.prompt, pair.rejected, pair.ref_answer)
    r_p = shaped_reward(s_sem_p, s_top_p, u_p, trainer.a, trainer.gamma_sem, trainer.gamma_topo, trainer.b_sem, trainer.b_topo, trainer.lam)["r"]
    r_n = shaped_reward(s_sem_n, s_top_n, u_n, trainer.a, trainer.gamma_sem, trainer.gamma_topo, trainer.b_sem, trainer.b_topo, trainer.lam)["r"]

    delta_r = r_p - r_n

    # Margin and loss (differentiable w.r.t. policy params)
    margin = beta * ((lp_pos - lp_neg) - (lp_ref_pos - lp_ref_neg)) + gamma * torch.tensor(delta_r, device=device, dtype=lp_pos.dtype)
    prob = torch.sigmoid(margin)
    loss = -torch.tensor(w, device=device, dtype=lp_pos.dtype) * torch.log(prob.clamp_min(1e-12))

    # Token-level KLs (policy || reference) on response tokens
    kl_pos = _token_kl(logits_pos_pol, logits_pos_ref, mask_pos)
    kl_neg = _token_kl(logits_neg_pol, logits_neg_ref, mask_neg)

    return loss, float(margin.item()), kl_pos, kl_neg, w
