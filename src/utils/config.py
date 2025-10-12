from __future__ import annotations
from typing import Dict, Any
import yaml


def load_config(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


def apply_overrides(cfg: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(cfg)
    for k, v in overrides.items():
        if v is not None:
            out[k] = v
    return out
