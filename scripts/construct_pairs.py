#!/usr/bin/env python
"""Construct reproducible TUR-DPO preference pairs and annotations."""

import argparse
import json
import logging
import sys
from pathlib import Path
from difflib import SequenceMatcher
from typing import Any, Callable, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from turdpo.rewards import SemanticScorer
from turdpo.topology import TopologyExtractor, TopologyGraph, TopologyScorer
from turdpo.uncertainty import UncertaintyEstimator
from turdpo.verifier import ContradictionDetector, NodeVerifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _candidate_text(candidate: Any) -> str:
    if isinstance(candidate, str):
        return candidate
    if isinstance(candidate, dict):
        return str(candidate.get("text", candidate.get("response", candidate.get("completion", ""))))
    return ""


def _candidate_score(candidate: Any) -> float:
    if isinstance(candidate, dict):
        for key in ("score", "task_score", "reward", "preference_score"):
            if candidate.get(key) is not None:
                try:
                    return float(candidate[key])
                except (TypeError, ValueError):
                    pass
    return 0.0


def _reference_logprob(model, tokenizer, prompt: str, response: str, device: Optional[str] = None) -> float:
    """Return response-token log probability for optional hard-negative mining."""
    import torch

    text = prompt + "\n\n" + response
    try:
        encoded = tokenizer(text, return_tensors="pt", truncation=True)
        prompt_encoded = tokenizer(prompt + "\n\n", add_special_tokens=False, return_tensors="pt")
    except TypeError:
        encoded = tokenizer(text, return_tensors="pt")
        prompt_encoded = tokenizer(prompt + "\n\n", return_tensors="pt")
    if device is None:
        device = str(next(model.parameters()).device)
    encoded = {key: value.to(device) for key, value in encoded.items() if hasattr(value, "to")}
    with torch.no_grad():
        output = model(**encoded)
    logits = output.logits if hasattr(output, "logits") else output[0]
    input_ids = encoded["input_ids"]
    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    targets = input_ids[:, 1:]
    token_log_probs = torch.gather(log_probs, -1, targets.unsqueeze(-1)).squeeze(-1)
    prompt_length = min(prompt_encoded["input_ids"].shape[-1], input_ids.shape[-1])
    positions = torch.arange(targets.shape[-1], device=targets.device) + 1
    response_mask = positions >= prompt_length
    if not response_mask.any():
        return 0.0
    return float(token_log_probs[0, response_mask].sum().item())


def _select_responses(
    item: Dict[str, Any],
    judge_fn: Optional[Callable[[str, str], float]] = None,
    pairwise_judge_fn: Optional[Callable[[str, str, str], float]] = None,
    reference_model=None,
    reference_tokenizer=None,
    reference_device: Optional[str] = None,
    hard_negative: bool = False,
) -> Optional[Tuple[str, str, float, float]]:
    if "chosen" in item and "rejected" in item:
        return (
            str(item.get("chosen", "")),
            str(item.get("rejected", "")),
            # A pair label is not a task metric.  Use the neutral prior when
            # no independent task score was supplied, otherwise the annotation
            # cache would leak the chosen/rejected label into semantic scoring.
            float(item.get("task_score_chosen", item.get("score_chosen", 0.5))),
            float(item.get("task_score_rejected", item.get("score_rejected", 0.5))),
        )
    candidates = item.get("candidates")
    if not isinstance(candidates, list) or len(candidates) < 2:
        return None
    prompt = str(item.get("prompt", ""))
    if pairwise_judge_fn is not None:
        def score(index: int) -> float:
            comparisons = []
            for other_index, other in enumerate(candidates):
                if index == other_index:
                    continue
                left = _candidate_text(candidates[index])
                right = _candidate_text(other)
                forward = float(pairwise_judge_fn(prompt, left, right))
                reverse = 1.0 - float(pairwise_judge_fn(prompt, right, left))
                comparisons.append((forward + reverse) / 2.0)
            return sum(comparisons) / max(len(comparisons), 1)
        scored = [(score(index), index, candidate) for index, candidate in enumerate(candidates)]
        ranked = [
            candidate for _, _, candidate in sorted(
                scored, key=lambda item: (item[0], item[1]), reverse=True
            )
        ]
    elif judge_fn is not None:
        ranked = sorted(
            candidates,
            key=lambda candidate: float(judge_fn(prompt, _candidate_text(candidate))),
            reverse=True,
        )
    else:
        ranked = sorted(candidates, key=_candidate_score, reverse=True)
    chosen = ranked[0]
    rejected = ranked[-1]
    if hard_negative and reference_model is not None and reference_tokenizer is not None:
        chosen_logprob = _reference_logprob(
            reference_model, reference_tokenizer, prompt, _candidate_text(chosen), reference_device
        )
        rejected = min(
            ranked[1:],
            key=lambda candidate: abs(
                _reference_logprob(
                    reference_model, reference_tokenizer, prompt, _candidate_text(candidate), reference_device
                ) - chosen_logprob
            ),
        )
    return (
        _candidate_text(chosen),
        _candidate_text(rejected),
        _candidate_score(chosen),
        _candidate_score(rejected),
    )


def _serialize_graph(graph: TopologyGraph) -> Dict[str, Any]:
    return {
        "nodes": [
            {
                "id": node.id,
                "content": node.content,
                "node_type": node.node_type,
                "correctness_prob": float(node.correctness_prob),
                "is_peer_assertion": bool(node.is_peer_assertion),
                "peer_verified": bool(node.peer_verified),
                "metadata": node.metadata,
            }
            for node in graph.nodes.values()
        ],
        "edges": [
            {
                "source_id": edge.source_id,
                "target_id": edge.target_id,
                "edge_type": edge.edge_type,
                "weight": float(edge.weight),
                "metadata": edge.metadata,
            }
            for edge in graph.edges
        ],
    }


def _graph_features(graph: TopologyGraph, scorer: TopologyScorer, detector: ContradictionDetector):
    contradiction_score, _ = detector.detect_contradictions(graph)
    score = scorer.compute_score(graph, contradiction_score)
    features = scorer.compute_features(graph)
    features["contradiction_score"] = float(contradiction_score)
    return score, features


def construct_preference_pairs(
    raw_data: List[Dict[str, Any]],
    extractor: Optional[TopologyExtractor] = None,
    scorer: Optional[TopologyScorer] = None,
    uncertainty_estimator: Optional[UncertaintyEstimator] = None,
    semantic_scorer: Optional[SemanticScorer] = None,
    verifier: Optional[NodeVerifier] = None,
    k_samples: int = 3,
    tau_w: float = 1.2,
    w_min: float = 0.05,
    alpha_peer_unverified: float = 0.0,
    include_graphs: bool = True,
    judge_fn: Optional[Callable[[str, str], float]] = None,
    pairwise_judge_fn: Optional[Callable[[str, str, str], float]] = None,
    reference_model=None,
    reference_tokenizer=None,
    reference_device: Optional[str] = None,
    hard_negative: bool = False,
    near_duplicate_threshold: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Construct pair records with the annotations consumed by the trainer."""
    if k_samples < 1:
        raise ValueError("k_samples must be at least 1")
    if not 0.0 <= w_min <= 1.0:
        raise ValueError("w_min must be in [0, 1]")
    if near_duplicate_threshold is not None and not 0.0 <= near_duplicate_threshold <= 1.0:
        raise ValueError("near_duplicate_threshold must be in [0, 1]")

    extractor = extractor or TopologyExtractor()
    scorer = scorer or TopologyScorer(alpha_peer_unverified=alpha_peer_unverified)
    uncertainty_estimator = uncertainty_estimator or UncertaintyEstimator(scorer=scorer)
    semantic_scorer = semantic_scorer or SemanticScorer()
    detector = ContradictionDetector()
    verifier = verifier or NodeVerifier()
    pairs: List[Dict[str, Any]] = []

    for index, item in enumerate(raw_data):
        if not isinstance(item, dict):
            logger.warning("Skipping item %s: expected an object", index)
            continue
        selected = _select_responses(
            item,
            judge_fn=judge_fn,
            pairwise_judge_fn=pairwise_judge_fn,
            reference_model=reference_model,
            reference_tokenizer=reference_tokenizer,
            reference_device=reference_device,
            hard_negative=hard_negative,
        )
        if selected is None:
            logger.warning("Skipping item %s: missing chosen/rejected or candidates", index)
            continue
        chosen_text, rejected_text, task_chosen, task_rejected = selected
        prompt = str(item.get("prompt", ""))
        chosen_graphs = extractor.extract_multiple(prompt, chosen_text, k=k_samples)
        rejected_graphs = extractor.extract_multiple(prompt, rejected_text, k=k_samples)
        for graph in chosen_graphs + rejected_graphs:
            verifier.verify_graph_nodes(graph, prompt)
        chosen_scores = [_graph_features(graph, scorer, detector) for graph in chosen_graphs]
        rejected_scores = [_graph_features(graph, scorer, detector) for graph in rejected_graphs]
        chosen_topology = float(sum(item[0] for item in chosen_scores) / max(len(chosen_scores), 1))
        rejected_topology = float(sum(item[0] for item in rejected_scores) / max(len(rejected_scores), 1))
        chosen_uncertainty = uncertainty_estimator.compute(chosen_graphs)
        rejected_uncertainty = uncertainty_estimator.compute(rejected_graphs)
        pair_weight = uncertainty_estimator.compute_pair_weight(
            chosen_uncertainty.total, rejected_uncertainty.total, tau_w=tau_w, w_min=w_min
        )
        chosen_graph = chosen_graphs[0] if chosen_graphs else None
        rejected_graph = rejected_graphs[0] if rejected_graphs else None
        chosen_semantic = semantic_scorer.compute(
            prompt, chosen_text, graph=chosen_graph, task_score=task_chosen,
            hallucination_score=chosen_scores[0][1].get("contradiction_score", 0.0) if chosen_scores else 0.0,
        )
        rejected_semantic = semantic_scorer.compute(
            prompt, rejected_text, graph=rejected_graph, task_score=task_rejected,
            hallucination_score=rejected_scores[0][1].get("contradiction_score", 0.0) if rejected_scores else 0.0,
        )
        metadata: Dict[str, Any] = {
            "annotation_version": "turdpo-annotations-v1",
            "topo_score_chosen": chosen_topology,
            "topo_score_rejected": rejected_topology,
            "semantic_score_chosen": float(chosen_semantic),
            "semantic_score_rejected": float(rejected_semantic),
            "uncertainty_chosen": float(chosen_uncertainty.total),
            "uncertainty_rejected": float(rejected_uncertainty.total),
            "pair_weight": float(pair_weight),
            "peer_pressure_rate_chosen": chosen_graph.peer_pressure_rate() if chosen_graph else 0.0,
            "peer_pressure_rate_rejected": rejected_graph.peer_pressure_rate() if rejected_graph else 0.0,
            "topology_features_chosen": chosen_scores[0][1] if chosen_scores else {},
            "topology_features_rejected": rejected_scores[0][1] if rejected_scores else {},
            "k_samples": k_samples,
            "tau_w": tau_w,
            "w_min": w_min,
        }
        if include_graphs:
            metadata["topology_graphs_chosen"] = [_serialize_graph(graph) for graph in chosen_graphs]
            metadata["topology_graphs_rejected"] = [_serialize_graph(graph) for graph in rejected_graphs]
        pairs.append({
            "id": item.get("id", f"pair_{index}"),
            "prompt": prompt,
            "chosen": chosen_text,
            "rejected": rejected_text,
            "task_score_chosen": task_chosen,
            "task_score_rejected": task_rejected,
            "metadata": metadata,
        })

    deduplicated = []
    seen = set()
    for pair in pairs:
        key = (pair["prompt"].strip(), pair["chosen"].strip(), pair["rejected"].strip())
        if key in seen:
            continue
        if near_duplicate_threshold is not None:
            duplicate = any(
                pair["prompt"].strip() == previous["prompt"].strip()
                and SequenceMatcher(None, pair["chosen"].strip(), previous["chosen"].strip()).ratio()
                >= near_duplicate_threshold
                and SequenceMatcher(None, pair["rejected"].strip(), previous["rejected"].strip()).ratio()
                >= near_duplicate_threshold
                for previous in deduplicated
            )
            if duplicate:
                continue
        seen.add(key)
        deduplicated.append(pair)
    logger.info(
        "Successfully constructed %s preference pairs (%s duplicates removed)",
        len(deduplicated), len(pairs) - len(deduplicated),
    )
    return deduplicated


def _load_records(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix.lower() == ".jsonl":
            value = [json.loads(line) for line in handle if line.strip()]
        else:
            value = json.load(handle)
    if isinstance(value, dict):
        value = value.get("data", value.get("pairs", value.get("records", [])))
    if not isinstance(value, list):
        raise ValueError("Input must be a JSON list, JSONL stream, or object containing data/pairs/records")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Construct annotated preference pairs for TUR-DPO")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--k_samples", type=int, default=3)
    parser.add_argument("--tau_w", type=float, default=1.2)
    parser.add_argument("--w_min", type=float, default=0.05)
    parser.add_argument("--alpha_peer_unverified", type=float, default=0.0)
    parser.add_argument("--no_graphs", action="store_true")
    parser.add_argument(
        "--reference_model",
        default=None,
        help="Optional causal LM used for reference-logprob hard-negative selection",
    )
    parser.add_argument("--reference_device", default="cpu")
    parser.add_argument("--hard_negative", action="store_true")
    parser.add_argument(
        "--near_duplicate_threshold",
        type=float,
        default=None,
        help="Optional lexical near-duplicate threshold in [0, 1]",
    )
    args = parser.parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(args.input)
    reference_model = None
    reference_tokenizer = None
    if args.hard_negative:
        if not args.reference_model:
            parser.error("--hard_negative requires --reference_model")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        reference_tokenizer = AutoTokenizer.from_pretrained(args.reference_model)
        reference_model = AutoModelForCausalLM.from_pretrained(args.reference_model)
        reference_model.to(args.reference_device)
        reference_model.eval()
    pairs = construct_preference_pairs(
        _load_records(input_path), k_samples=args.k_samples, tau_w=args.tau_w,
        w_min=args.w_min, alpha_peer_unverified=args.alpha_peer_unverified,
        include_graphs=not args.no_graphs,
        reference_model=reference_model,
        reference_tokenizer=reference_tokenizer,
        reference_device=args.reference_device,
        hard_negative=args.hard_negative,
        near_duplicate_threshold=args.near_duplicate_threshold,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        if output_path.suffix.lower() == ".jsonl":
            for pair in pairs:
                handle.write(json.dumps(pair, ensure_ascii=False) + "\n")
        else:
            json.dump(pairs, handle, indent=2, ensure_ascii=False)
    logger.info("Saved preference pairs to %s", output_path)


if __name__ == "__main__":
    main()
