from __future__ import annotations
from typing import Dict
import re


def _exact_match(answer: str, reference: str) -> float:
    return 1.0 if answer.strip().lower() == reference.strip().lower() else 0.0


def _hallucination_penalty(text: str) -> float:
    # naive heuristic: penalize if many proper-noun-like tokens without numbers and references
    tokens = re.findall(r"[A-Za-z]+", text)
    caps = sum(1 for t in tokens if t[:1].isupper())
    return min(1.0, max(0.0, (caps - 3) / 10.0))


def semantic_score(x: str, y: str, ref_answer: str | None = None,
                   beta1: float = 0.7, beta2: float = 0.3, beta3: float = 0.4) -> Dict[str, float]:
    """
    Compute a simple semantic score:
    - q_fact: crude proxy via presence of numbers or cited evidence cues
    - q_task: exact-match to provided reference answer if available, else 0
    - q_hall: hallucination heuristic
    Returns components and s_sem.
    """
    cues = re.findall(r"\b(because|according to|evidence|proof|ref|cite)\b", y, flags=re.I)
    has_num = bool(re.search(r"\d", y))
    q_fact = 0.5 + 0.2 * bool(cues) + 0.3 * has_num
    q_fact = min(1.0, max(0.0, q_fact))

    q_task = _exact_match(y, ref_answer) if ref_answer is not None else 0.0

    q_hall = _hallucination_penalty(y)

    s_sem = beta1 * q_fact + beta2 * q_task - beta3 * q_hall
    return {
        "q_fact": float(q_fact),
        "q_task": float(q_task),
        "q_hall": float(q_hall),
        "s_sem": float(s_sem),
    }
