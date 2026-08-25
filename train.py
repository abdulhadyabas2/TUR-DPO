#!/usr/bin/env python
"""Command-line entry point for reproducible TUR-DPO training."""

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

from turdpo.data import (
    ListwisePreferenceDataset,
    PreferenceDataset,
    collate_listwise_batch,
    create_dataloader,
    split_dataset,
)
from turdpo.topology import TopologyExtractor
from turdpo.trainer import TURDPOConfig, TURDPOTrainer
from turdpo.utils import save_config, setup_logging
from turdpo.verifier import ContradictionDetector, NodeVerifier

logger = logging.getLogger(__name__)


def _config_defaults(config_path: str) -> Dict[str, Any]:
    if not config_path:
        return {}
    with open(config_path, "r", encoding="utf-8") as handle:
        values = json.load(handle)
    if not isinstance(values, dict):
        raise ValueError("Configuration file must contain a JSON object")
    # Historical config files used this spelling; the CLI uses one canonical name.
    if "use_ema_reference" in values and "use_ema" not in values:
        values["use_ema"] = values.pop("use_ema_reference")
    return values


def parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=None)
    pre_args, _ = pre_parser.parse_known_args()
    defaults = _config_defaults(pre_args.config) if pre_args.config else {}

    parser = argparse.ArgumentParser(description="Train a causal LM with TUR-DPO")
    parser.add_argument("--config", default=None, help="JSON config file; CLI values override it")
    parser.add_argument("--model_name", default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--reference_model", default=None)
    parser.add_argument("--topology_model", default=None, help="Optional causal LM used for JSON graph elicitation")
    parser.add_argument("--verifier_model", default=None, help="Optional sequence-classification NLI verifier")
    parser.add_argument("--train_data", default=None)
    parser.add_argument("--eval_data", default=None)
    parser.add_argument("--calibration_data", default=None)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--a", type=float, default=0.6)
    parser.add_argument("--lambda_uncertainty", type=float, default=0.5)
    parser.add_argument("--lambda_peer_pressure", type=float, default=0.0)
    parser.add_argument("--alpha_path", type=float, default=1.0)
    parser.add_argument("--alpha_cycle", type=float, default=0.5)
    parser.add_argument("--alpha_dangling", type=float, default=0.3)
    parser.add_argument("--alpha_contradict", type=float, default=0.4)
    parser.add_argument("--alpha_peer_unverified", type=float, default=0.0)
    parser.add_argument("--lambda_epi", type=float, default=0.5)
    parser.add_argument("--lambda_ale", type=float, default=0.5)
    parser.add_argument("--tau_smoothing", type=float, default=0.05)
    parser.add_argument("--tau_w", type=float, default=1.2)
    parser.add_argument("--w_min", type=float, default=0.05)
    parser.add_argument("--k_samples", type=int, default=3)
    parser.add_argument("--use_ema", dest="use_ema", action="store_true", default=True)
    parser.add_argument("--no-use-ema", dest="use_ema", action="store_false")
    parser.add_argument("--ema_decay", type=float, default=0.995)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--use_listwise", action="store_true", default=False)
    parser.add_argument("--num_candidates", type=int, default=4)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=2000)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--resume", default=None, help="Optional TUR-DPO checkpoint to resume")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument(
        "--fit_calibrators",
        action="store_true",
        help="Fit reward calibrators from explicit per-candidate held-out targets",
    )
    parser.add_argument(
        "--objective",
        choices=("turdpo", "dpo"),
        default=None,
        help="Use full TUR-DPO or the unshaped/unweighted DPO control",
    )
    parser.add_argument(
        "--use_precomputed_annotations", dest="use_precomputed_annotations",
        action="store_true", default=True,
        help="Use annotations generated by scripts/construct_pairs.py when present",
    )
    parser.add_argument(
        "--no-use-precomputed-annotations", dest="use_precomputed_annotations",
        action="store_false",
    )
    parser.set_defaults(**{key: value for key, value in defaults.items() if key != "config"})
    args = parser.parse_args()
    if not args.train_data:
        parser.error("--train_data is required (directly or in --config)")
    if args.fp16 and args.bf16:
        parser.error("--fp16 and --bf16 are mutually exclusive")
    return args


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _load_dataset(path: str, tokenizer, args: argparse.Namespace) -> PreferenceDataset:
    kwargs = {"max_length": args.max_length, "max_prompt_length": args.max_prompt_length}
    return (
        PreferenceDataset.from_jsonl(path, tokenizer, **kwargs)
        if path.lower().endswith(".jsonl")
        else PreferenceDataset.from_json(path, tokenizer, **kwargs)
    )


def _load_listwise_dataset(
    path: str,
    tokenizer,
    args: argparse.Namespace,
) -> ListwisePreferenceDataset:
    kwargs = {"max_length": args.max_length, "num_candidates": args.num_candidates}
    return (
        ListwisePreferenceDataset.from_jsonl(path, tokenizer, **kwargs)
        if path.lower().endswith(".jsonl")
        else ListwisePreferenceDataset.from_json(path, tokenizer, **kwargs)
    )


def _fit_reward_calibrators(trainer: TURDPOTrainer, path: str) -> None:
    """Fit reward calibrators from explicit annotation targets only."""
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        records = (
            [json.loads(line) for line in handle if line.strip()]
            if source.suffix.lower() == ".jsonl"
            else json.load(handle)
        )
    if isinstance(records, dict):
        records = records.get("data", records.get("records", records.get("pairs", [])))
    semantic_scores = []
    topology_scores = []
    targets = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        metadata = record.get("metadata", {}) if isinstance(record.get("metadata"), dict) else {}
        target_chosen = record.get("calibration_target_chosen", metadata.get("calibration_target_chosen"))
        target_rejected = record.get("calibration_target_rejected", metadata.get("calibration_target_rejected"))
        semantic_chosen = metadata.get("semantic_score_chosen")
        semantic_rejected = metadata.get("semantic_score_rejected")
        topology_chosen = metadata.get("topo_score_chosen")
        topology_rejected = metadata.get("topo_score_rejected")
        values = [
            (semantic_chosen, topology_chosen, target_chosen),
            (semantic_rejected, topology_rejected, target_rejected),
        ]
        for semantic, topology, target in values:
            if semantic is None or topology is None or target is None:
                continue
            semantic_scores.append(float(semantic))
            topology_scores.append(float(topology))
            targets.append(float(target))
    if not targets:
        raise ValueError(
            "Calibration data must contain metadata scores and explicit "
            "calibration_target_chosen/rejected fields"
        )
    trainer.shaped_reward.fit_calibrators(
        np.asarray(semantic_scores), np.asarray(topology_scores), np.asarray(targets)
    )
    logger.info("Fitted reward calibrators on %s explicit held-out targets", len(targets))


def main() -> None:
    args = parse_args()
    setup_logging()
    _set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable; using CPU")
        device = "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float32
    if args.bf16:
        dtype = torch.bfloat16
    elif args.fp16:
        dtype = torch.float16
    load_kwargs = {"torch_dtype": dtype}
    if device == "cuda":
        load_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **load_kwargs)
    reference_name = args.reference_model or args.model_name
    reference_model = AutoModelForCausalLM.from_pretrained(reference_name, **load_kwargs)

    topology_extractor = None
    if args.topology_model:
        topology_tokenizer = AutoTokenizer.from_pretrained(args.topology_model)
        if topology_tokenizer.pad_token is None:
            topology_tokenizer.pad_token = topology_tokenizer.eos_token
        topology_model = AutoModelForCausalLM.from_pretrained(args.topology_model, **load_kwargs)
        topology_extractor = TopologyExtractor(model=topology_model, tokenizer=topology_tokenizer)

    verifier = None
    contradiction_detector = None
    if args.verifier_model:
        verifier_tokenizer = AutoTokenizer.from_pretrained(args.verifier_model)
        verifier_model = AutoModelForSequenceClassification.from_pretrained(
            args.verifier_model, **load_kwargs
        )
        verifier = NodeVerifier(verifier_model, verifier_tokenizer)
        contradiction_detector = ContradictionDetector(verifier_model, verifier_tokenizer)

    if args.use_listwise:
        dataset = _load_listwise_dataset(args.train_data, tokenizer, args)
        train_dataset = dataset
        val_dataset = None
        calibration_dataset = None
    else:
        dataset = _load_dataset(args.train_data, tokenizer, args)
        train_dataset, val_dataset, calibration_dataset = split_dataset(dataset, seed=args.seed)
    with (output_dir / "split_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump({
            "train_ids": [item.get("id", str(index)) for index, item in enumerate(train_dataset.data)],
            "validation_ids": [] if val_dataset is None else [
                pair.prompt_id for pair in val_dataset.data
            ],
            "calibration_ids": [] if calibration_dataset is None else [
                pair.prompt_id for pair in calibration_dataset.data
            ],
        }, handle, indent=2)
    train_loader = create_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_listwise_batch if args.use_listwise else None,
    )
    eval_loader = None
    if args.eval_data:
        eval_dataset = (
            _load_listwise_dataset(args.eval_data, tokenizer, args)
            if args.use_listwise
            else _load_dataset(args.eval_data, tokenizer, args)
        )
        eval_loader = create_dataloader(
            eval_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_listwise_batch if args.use_listwise else None,
        )
    elif val_dataset is not None and len(val_dataset):
        eval_loader = create_dataloader(
            val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
        )

    config_values = {
        key: getattr(args, key)
        for key in TURDPOConfig.__dataclass_fields__
        if hasattr(args, key)
    }
    config_values["use_ema_reference"] = args.use_ema
    if args.objective is not None:
        config_values["use_shaped_reward"] = args.objective == "turdpo"
        config_values["use_pair_weights"] = args.objective == "turdpo"
    if args.fp16 or args.bf16:
        config_values["use_amp"] = True
        config_values["amp_dtype"] = "float16" if args.fp16 else "bf16"
    config = TURDPOConfig(**config_values)
    save_config({
        "model_name": args.model_name,
        "reference_model": args.reference_model or args.model_name,
        "topology_model": args.topology_model,
        "verifier_model": args.verifier_model,
        "seed": args.seed,
        "device": device,
        "dtype": str(dtype),
        "turdpo": config.__dict__,
    }, output_dir / "config.json")
    trainer = TURDPOTrainer(
        model=model,
        reference_model=reference_model,
        tokenizer=tokenizer,
        config=config,
        topology_extractor=topology_extractor,
        verifier=verifier,
        contradiction_detector=contradiction_detector,
        device=device,
    )
    if args.fit_calibrators or config.train_calibrators:
        if not args.calibration_data:
            raise ValueError("--calibration_data is required when fitting calibrators")
        _fit_reward_calibrators(trainer, args.calibration_data)
    if args.resume:
        trainer.load_checkpoint(args.resume)
    results = trainer.train(train_loader, eval_dataloader=eval_loader, num_epochs=args.num_epochs)
    with (output_dir / "training_results.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, default=str)
    final_model_path = output_dir / "final_model"
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    trainer.save_checkpoint(output_dir / "checkpoint.pt")
    logger.info("Training complete; artifacts saved to %s", output_dir)


if __name__ == "__main__":
    main()
