from __future__ import annotations
from typing import List, Dict
import json


def load_pairs_jsonl(path: str) -> List[Dict]:
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))
    return pairs
