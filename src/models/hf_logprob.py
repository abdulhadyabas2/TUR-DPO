from __future__ import annotations
from typing import List, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_causal_lm(model_name_or_path: str, device: str | None = None):
    tok = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else None,
    )
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return tok, model, device


def logprob_y_given_x(tok, model, device: str, prompt: str, response: str) -> float:
    enc = tok(prompt, return_tensors="pt")
    dec = tok(response, return_tensors="pt", add_special_tokens=False)
    input_ids = torch.cat([enc.input_ids, dec.input_ids], dim=1).to(device)
    with torch.no_grad():
        out = model(input_ids=input_ids)
        logits = out.logits[:, :-1, :]
        labels = input_ids[:, 1:]
        # Mask only response tokens in labels
        P = enc.input_ids.shape[1]
        mask = torch.zeros_like(labels)
        mask[:, P - 1 :] = 1
        logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
        token_lp = (logprobs.gather(-1, labels.unsqueeze(-1)).squeeze(-1) * mask).sum().item()
    return float(token_lp)


def batch_pair_logprobs(tok, model, device: str, prompts: List[str], chosens: List[str], rejecteds: List[str]) -> Tuple[List[float], List[float]]:
    pos = []
    neg = []
    for x, y_pos, y_neg in zip(prompts, chosens, rejecteds):
        pos.append(logprob_y_given_x(tok, model, device, x, y_pos))
        neg.append(logprob_y_given_x(tok, model, device, x, y_neg))
    return pos, neg


def ema_update_model(ref_model, pol_model, decay: float = 0.995):
    """In-place EMA update of ref_model parameters toward pol_model.
    ref := decay * ref + (1 - decay) * pol
    """
    import torch
    with torch.no_grad():
        ref_params = dict(ref_model.named_parameters())
        for name, p_pol in pol_model.named_parameters():
            p_ref = ref_params.get(name, None)
            if p_ref is None or p_ref.data.shape != p_pol.data.shape:
                continue
            p_ref.data.mul_(decay).add_(p_pol.data, alpha=(1.0 - decay))
