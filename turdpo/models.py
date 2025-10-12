from __future__ import annotations
from typing import List, Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_causal_lm(model_name_or_path: str, device: str = None):
    tok = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16 if torch.cuda.is_available() else None)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return tok, model, device


def batch_logprobs(tok, model, device: str, prompts: List[str], responses: List[str]) -> List[float]:
    """Compute log p(y|x) token-wise and sum. Naive version for demo; uses teacher-forcing of response tokens."""
    results: List[float] = []
    for x, y in zip(prompts, responses):
        # concatenate x + y and compute loss on y tokens
        enc = tok(x, return_tensors="pt")
        dec = tok(y, return_tensors="pt", add_special_tokens=False)
        input_ids = torch.cat([enc.input_ids, dec.input_ids], dim=1).to(device)
        with torch.no_grad():
            out = model(input_ids=input_ids)
            logits = out.logits[:, :-1, :]
            labels = input_ids[:, 1:]
            # mask only y tokens
            mask = torch.zeros_like(labels)
            mask[:, enc.input_ids.shape[1]-1:] = 1  # include the token right after prompt end
            logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
            token_lp = (logprobs.gather(-1, labels.unsqueeze(-1)).squeeze(-1) * mask).sum().item()
        results.append(float(token_lp))
    return results
