#!/usr/bin/env python
"""Run reproducible TUR-DPO controls, ablations, and multiple seeds.

The runner orchestrates local training only. It does not download datasets,
invent judge labels, or claim that a run reproduces manuscript numbers.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _default_variants() -> Dict[str, Dict[str, Any]]:
    return {
        "turdpo": {},
        "dpo_control": {
            "use_shaped_reward": False,
            "use_pair_weights": False,
            "use_precomputed_annotations": False,
        },
        "no_topology_penalty": {"alpha_peer_unverified": 0.0},
        "no_uncertainty_weighting": {"use_pair_weights": False},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TUR-DPO experiment matrices")
    parser.add_argument("--base_config", required=True, help="Base JSON config containing model and data paths")
    parser.add_argument("--variants", default=None, help="JSON object mapping variant names to config overrides")
    parser.add_argument("--seeds", default="42", help="Comma-separated random seeds")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--train_data", default=None, help="Optional override for every variant")
    parser.add_argument("--eval_data", default=None, help="Optional override for every variant")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    base = _load_json(Path(args.base_config))
    variants = _load_json(Path(args.variants)) if args.variants else _default_variants()
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    if not seeds:
        raise ValueError("At least one seed is required")
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    manifest: List[Dict[str, Any]] = []
    for variant_name, overrides in variants.items():
        if not isinstance(overrides, dict):
            raise ValueError(f"Variant {variant_name!r} must map to a JSON object")
        for seed in seeds:
            run_dir = output_root / variant_name / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            config = dict(base)
            config.update(overrides)
            config["seed"] = seed
            config["output_dir"] = str(run_dir)
            if args.train_data:
                config["train_data"] = args.train_data
            if args.eval_data:
                config["eval_data"] = args.eval_data
            config_path = run_dir / "experiment_config.json"
            config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
            command = [sys.executable, str(root / "train.py"), "--config", str(config_path)]
            record = {
                "variant": variant_name,
                "seed": seed,
                "config": str(config_path),
                "output_dir": str(run_dir),
                "command": command,
            }
            manifest.append(record)
            print(" ".join(command))
            if args.dry_run:
                continue
            log_path = run_dir / "train.log"
            try:
                with log_path.open("w", encoding="utf-8") as log:
                    subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
                record["status"] = "completed"
            except subprocess.CalledProcessError as exc:
                record["status"] = "failed"
                record["returncode"] = exc.returncode
                if not args.continue_on_error:
                    raise

    (output_root / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
