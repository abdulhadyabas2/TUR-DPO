"""
Verifier Module for TUR-DPO

This module implements verification of node-level claims for:
1. Factual correctness (fact checking)
2. Logical consistency (contradiction detection)
3. Calibrated correctness probabilities
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod


def _find_label_index(id2label: Dict[Any, Any], candidates: set) -> Optional[int]:
    """Find a sequence-classification label index by normalized label name."""
    for index, label in id2label.items():
        normalized = str(label).lower().replace("_", " ").replace("-", " ")
        if any(candidate in normalized for candidate in candidates):
            try:
                return int(index)
            except (TypeError, ValueError):
                continue
    return None


@dataclass
class VerificationResult:
    """Container for verification results."""
    is_correct: bool
    probability: float
    confidence: float
    evidence: Optional[str] = None
    reasoning: Optional[str] = None
    metadata: Dict[str, Any] = None


class BaseVerifier(ABC):
    """Abstract base class for verifiers."""

    @abstractmethod
    def verify(self, claim: str, context: str) -> VerificationResult:
        """Verify a single claim."""
        pass

    @abstractmethod
    def verify_batch(self, claims: List[str], context: str) -> List[VerificationResult]:
        """Verify multiple claims."""
        pass


class NodeVerifier(BaseVerifier):
    """
    Verify node-level claims in a topology graph.

    Provides calibrated correctness probabilities for uncertainty estimation.
    """

    def __init__(
        self,
        model=None,
        tokenizer=None,
        calibration_temperature: float = 1.0,
        use_isotonic: bool = False,
        confidence_threshold: float = 0.5
    ):
        """
        Initialize node verifier.

        Args:
            model: Language model for verification
            tokenizer: Tokenizer for the model
            calibration_temperature: Temperature scaling for calibration
            use_isotonic: Whether to use isotonic regression for calibration
            confidence_threshold: Threshold for binary correctness decision
        """
        self.model = model
        self.tokenizer = tokenizer
        self.calibration_temperature = calibration_temperature
        self.use_isotonic = use_isotonic
        self.confidence_threshold = confidence_threshold

        # Calibration parameters (fitted on held-out data)
        self._isotonic_func = None

    def verify(self, claim: str, context: str) -> VerificationResult:
        """
        Verify a single claim against context.

        Args:
            claim: The claim to verify
            context: Context/evidence to verify against

        Returns:
            VerificationResult with correctness probability
        """
        if self.model is None:
            # Fallback: simple heuristic-based verification
            return self._heuristic_verify(claim, context)

        # Model-based verification
        return self._model_verify(claim, context)

    def verify_batch(
        self,
        claims: List[str],
        context: str
    ) -> List[VerificationResult]:
        """
        Verify multiple claims in batch.

        Args:
            claims: List of claims to verify
            context: Shared context for all claims

        Returns:
            List of VerificationResults
        """
        return [self.verify(claim, context) for claim in claims]

    def verify_graph_nodes(
        self,
        graph,
        context: str
    ) -> Dict[str, VerificationResult]:
        """
        Verify all nodes in a topology graph.

        Args:
            graph: TopologyGraph with nodes to verify
            context: Context for verification

        Returns:
            Dict mapping node_id to VerificationResult
        """
        results = {}
        for node_id, node in graph.nodes.items():
            result = self.verify(node.content, context)
            results[node_id] = result
            # Update node with correctness probability
            node.correctness_prob = result.probability
        return results

    def _heuristic_verify(self, claim: str, context: str) -> VerificationResult:
        """
        Simple heuristic verification when no model is available.

        Uses basic text matching and pattern detection.
        """
        claim_lower = claim.lower()
        context_lower = context.lower()

        # Check for claim keywords in context
        claim_words = set(claim_lower.split())
        context_words = set(context_lower.split())
        overlap = len(claim_words & context_words) / max(len(claim_words), 1)

        # Base probability on word overlap
        base_prob = min(overlap * 1.5, 0.9)  # Cap at 0.9

        # Penalize hedging language
        hedge_words = {"might", "maybe", "possibly", "perhaps", "could", "uncertain"}
        if any(w in claim_lower for w in hedge_words):
            base_prob *= 0.8

        # Boost for specific numeric claims if they appear in context
        import re
        numbers_in_claim = set(re.findall(r'\d+\.?\d*', claim))
        numbers_in_context = set(re.findall(r'\d+\.?\d*', context))
        if numbers_in_claim and numbers_in_claim.issubset(numbers_in_context):
            base_prob = min(base_prob + 0.2, 0.95)

        # Apply calibration
        calibrated_prob = self._calibrate(base_prob)

        return VerificationResult(
            is_correct=calibrated_prob >= self.confidence_threshold,
            probability=calibrated_prob,
            confidence=abs(calibrated_prob - 0.5) * 2,
            evidence=None,
            reasoning="Heuristic verification based on text overlap"
        )

    def _model_verify(self, claim: str, context: str) -> VerificationResult:
        """
        Model-based verification using an NLI/sequence-classification model.

        The model is expected to return sequence-classification logits.  Label
        names are read from ``config.id2label`` when available; otherwise the
        highest-probability class is treated as entailment.  If a caller supplies
        an incompatible model, the explicit heuristic fallback remains available
        and is recorded in ``reasoning``.
        """
        try:
            import torch

            encoded = self.tokenizer(
                context,
                claim,
                return_tensors="pt",
                truncation=True,
            )
            device = next(self.model.parameters()).device
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.no_grad():
                output = self.model(**encoded)
            logits = output.logits if hasattr(output, "logits") else output[0]
            probabilities = torch.softmax(logits, dim=-1)[0]
            labels = getattr(getattr(self.model, "config", None), "id2label", {}) or {}
            entailment_index = _find_label_index(labels, {"entailment", "entails", "support"})
            if entailment_index is None:
                entailment_index = int(torch.argmax(probabilities).item())
            probability = float(probabilities[entailment_index].item())
            calibrated_prob = self._calibrate(probability)
            return VerificationResult(
                is_correct=calibrated_prob >= self.confidence_threshold,
                probability=calibrated_prob,
                confidence=abs(calibrated_prob - 0.5) * 2,
                evidence=None,
                reasoning="NLI sequence-classification verification",
            )
        except Exception as exc:
            result = self._heuristic_verify(claim, context)
            result.reasoning = f"Heuristic fallback after model verification error: {exc}"
            return result

    def _calibrate(self, raw_prob: float) -> float:
        """
        Apply calibration to raw probability.

        Uses temperature scaling: p_cal = σ(logit / T)
        """
        if self.use_isotonic and self._isotonic_func is not None:
            return self._isotonic_func(raw_prob)

        # Temperature scaling
        if raw_prob <= 0 or raw_prob >= 1:
            return np.clip(raw_prob, 0.01, 0.99)

        logit = np.log(raw_prob / (1 - raw_prob))
        scaled_logit = logit / self.calibration_temperature
        calibrated = 1 / (1 + np.exp(-scaled_logit))

        return calibrated

    def fit_calibration(
        self,
        raw_probs: np.ndarray,
        true_labels: np.ndarray
    ) -> None:
        """
        Fit calibration parameters on held-out data.

        Args:
            raw_probs: Raw predicted probabilities
            true_labels: True binary labels (0 or 1)
        """
        if self.use_isotonic:
            from sklearn.isotonic import IsotonicRegression
            self._isotonic_func = IsotonicRegression(
                y_min=0.01, y_max=0.99, out_of_bounds='clip'
            ).fit(raw_probs, true_labels).predict
        else:
            # Fit temperature using NLL
            def nll(T):
                logits = np.log(raw_probs / (1 - raw_probs + 1e-10) + 1e-10)
                scaled = 1 / (1 + np.exp(-logits / T))
                scaled = np.clip(scaled, 1e-10, 1 - 1e-10)
                return -np.mean(
                    true_labels * np.log(scaled) +
                    (1 - true_labels) * np.log(1 - scaled)
                )

            from scipy.optimize import minimize_scalar
            result = minimize_scalar(nll, bounds=(0.1, 10.0), method='bounded')
            self.calibration_temperature = result.x


class FactChecker(BaseVerifier):
    """
    Fact checking verifier using external knowledge sources.

    Checks claims against Wikipedia, knowledge bases, or retrieval systems.
    """

    def __init__(
        self,
        retriever=None,
        model=None,
        tokenizer=None,
        top_k: int = 5
    ):
        """
        Initialize fact checker.

        Args:
            retriever: Retrieval system for finding evidence
            model: Model for entailment checking
            tokenizer: Tokenizer for the model
            top_k: Number of evidence passages to retrieve
        """
        self.retriever = retriever
        self.model = model
        self.tokenizer = tokenizer
        self.top_k = top_k

    def verify(self, claim: str, context: str = "") -> VerificationResult:
        """
        Verify a factual claim.

        Args:
            claim: The claim to verify
            context: Optional additional context

        Returns:
            VerificationResult with fact-checking outcome
        """
        # Retrieve evidence
        evidence = self._retrieve_evidence(claim)

        if not evidence:
            return VerificationResult(
                is_correct=False,
                probability=0.5,
                confidence=0.0,
                evidence=None,
                reasoning="No evidence found"
            )

        # Check entailment with evidence
        return self._check_entailment(claim, evidence)

    def verify_batch(
        self,
        claims: List[str],
        context: str = ""
    ) -> List[VerificationResult]:
        """Verify multiple claims."""
        return [self.verify(claim, context) for claim in claims]

    def _retrieve_evidence(self, claim: str) -> List[str]:
        """Retrieve relevant evidence for a claim."""
        if self.retriever is None:
            return []
        try:
            if hasattr(self.retriever, "retrieve"):
                evidence = self.retriever.retrieve(claim, k=self.top_k)
            elif callable(self.retriever):
                evidence = self.retriever(claim, self.top_k)
            else:
                return []
            normalized = []
            for item in evidence or []:
                if isinstance(item, str):
                    normalized.append(item)
                elif isinstance(item, dict):
                    normalized.append(str(item.get("text", item.get("content", ""))))
            return [item for item in normalized if item]
        except Exception:
            return []

    def _check_entailment(
        self,
        claim: str,
        evidence: List[str]
    ) -> VerificationResult:
        """
        Check if evidence entails or contradicts the claim.
        """
        if not evidence:
            return VerificationResult(
                is_correct=False,
                probability=0.5,
                confidence=0.0,
                evidence=None,
                reasoning="No evidence to check"
            )

        if self.model is not None and self.tokenizer is not None:
            verifier = NodeVerifier(self.model, self.tokenizer)
            results = [verifier.verify(claim, item) for item in evidence]
            best = max(results, key=lambda result: result.probability)
            best.evidence = evidence[results.index(best)]
            best.reasoning = "NLI verification against retrieved evidence"
            return best

        claim_words = set(claim.lower().split())
        best_evidence = max(evidence, key=lambda item: len(claim_words & set(item.lower().split())))
        overlap = len(claim_words & set(best_evidence.lower().split())) / max(len(claim_words), 1)
        probability = float(np.clip(overlap, 0.0, 1.0))
        return VerificationResult(
            is_correct=probability >= 0.5,
            probability=probability,
            confidence=abs(probability - 0.5) * 2,
            evidence=best_evidence,
            reasoning="Lexical-overlap fallback against retrieved evidence",
        )


class ContradictionDetector:
    """
    Detect contradictions between claims in a topology graph.

    Used for computing the contradiction score q_contradict in topology scoring.
    """

    def __init__(
        self,
        model=None,
        tokenizer=None,
        threshold: float = 0.7
    ):
        """
        Initialize contradiction detector.

        Args:
            model: NLI model for contradiction detection
            tokenizer: Tokenizer for the model
            threshold: Threshold for contradiction classification
        """
        self.model = model
        self.tokenizer = tokenizer
        self.threshold = threshold

    def detect_contradictions(
        self,
        graph
    ) -> Tuple[float, List[Tuple[str, str]]]:
        """
        Detect contradictions in a topology graph.

        Args:
            graph: TopologyGraph to analyze

        Returns:
            Tuple of (contradiction_score, list of contradicting pairs)
        """
        if len(graph.nodes) < 2:
            return 0.0, []

        contradictions = []
        total_pairs = 0

        # Check pairs of connected nodes
        for edge in graph.edges:
            source_node = graph.nodes.get(edge.source_id)
            target_node = graph.nodes.get(edge.target_id)

            if source_node and target_node:
                is_contradiction = self._check_contradiction(
                    source_node.content,
                    target_node.content
                )
                total_pairs += 1

                if is_contradiction:
                    contradictions.append((edge.source_id, edge.target_id))

        # Compute contradiction score as fraction of contradicting edges
        score = len(contradictions) / max(total_pairs, 1)

        return score, contradictions

    def _check_contradiction(self, claim1: str, claim2: str) -> bool:
        """
        Check if two claims contradict each other.

        Args:
            claim1: First claim
            claim2: Second claim

        Returns:
            True if claims contradict
        """
        if self.model is None:
            return self._heuristic_contradiction(claim1, claim2)

        try:
            import torch

            encoded = self.tokenizer(
                claim1,
                claim2,
                return_tensors="pt",
                truncation=True,
            )
            device = next(self.model.parameters()).device
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.no_grad():
                output = self.model(**encoded)
            logits = output.logits if hasattr(output, "logits") else output[0]
            probabilities = torch.softmax(logits, dim=-1)[0]
            labels = getattr(getattr(self.model, "config", None), "id2label", {}) or {}
            contradiction_index = _find_label_index(
                labels, {"contradiction", "contradicts", "contradict"}
            )
            if contradiction_index is None:
                return self._heuristic_contradiction(claim1, claim2)
            return float(probabilities[contradiction_index].item()) >= self.threshold
        except Exception:
            return self._heuristic_contradiction(claim1, claim2)

    def _heuristic_contradiction(self, claim1: str, claim2: str) -> bool:
        """
        Simple heuristic for contradiction detection.
        """
        # Check for negation patterns
        negations = ["not", "no", "never", "none", "neither", "cannot", "won't", "don't"]

        claim1_lower = claim1.lower()
        claim2_lower = claim2.lower()

        # Count negations in each claim
        neg1 = sum(1 for neg in negations if neg in claim1_lower)
        neg2 = sum(1 for neg in negations if neg in claim2_lower)

        # Check for opposite polarity
        if (neg1 % 2) != (neg2 % 2):
            # Check if claims are about similar content
            words1 = set(claim1_lower.split())
            words2 = set(claim2_lower.split())

            # Remove common stop words
            stop_words = {"the", "a", "an", "is", "are", "was", "were", "it", "this", "that"}
            words1 -= stop_words
            words2 -= stop_words

            overlap = len(words1 & words2) / max(len(words1 | words2), 1)

            if overlap > 0.3:
                return True

        return False


class ArithmeticVerifier:
    """
    Verify arithmetic computations in mathematical reasoning.
    """

    def __init__(self):
        self.epsilon = 1e-6

    def verify_calculation(
        self,
        expression: str,
        claimed_result: float
    ) -> VerificationResult:
        """
        Verify an arithmetic calculation.

        Args:
            expression: Mathematical expression
            claimed_result: The claimed answer

        Returns:
            VerificationResult
        """
        try:
            # Safe eval for basic arithmetic
            actual_result = self._safe_eval(expression)

            if actual_result is None:
                return VerificationResult(
                    is_correct=False,
                    probability=0.5,
                    confidence=0.0,
                    reasoning="Could not evaluate expression"
                )

            is_correct = abs(actual_result - claimed_result) < self.epsilon

            return VerificationResult(
                is_correct=is_correct,
                probability=1.0 if is_correct else 0.0,
                confidence=1.0,
                evidence=f"Computed: {actual_result}",
                reasoning=f"Expression evaluates to {actual_result}"
            )

        except Exception as e:
            return VerificationResult(
                is_correct=False,
                probability=0.5,
                confidence=0.0,
                reasoning=f"Error evaluating: {str(e)}"
            )

    def _safe_eval(self, expression: str) -> Optional[float]:
        """
        Safely evaluate a mathematical expression.
        """
        import re

        # Allow only basic arithmetic operators and numbers
        allowed = re.compile(r'^[\d\s\+\-\*\/\.\(\)]+$')

        if not allowed.match(expression):
            return None

        try:
            # Use eval with restricted globals
            result = eval(expression, {"__builtins__": {}}, {})
            return float(result)
        except:
            return None


def compute_graph_correctness_probs(
    graph,
    verifier: NodeVerifier,
    context: str
) -> Dict[str, float]:
    """
    Compute correctness probabilities for all nodes in a graph.

    Args:
        graph: TopologyGraph to verify
        verifier: NodeVerifier instance
        context: Context for verification

    Returns:
        Dict mapping node_id to correctness probability
    """
    results = verifier.verify_graph_nodes(graph, context)
    return {node_id: result.probability for node_id, result in results.items()}
