"""
Rewards Module for TUR-DPO

This module implements the shaped reward computation combining semantic, topology,
and uncertainty signals.

Based on Equations (7) and (8) from the paper:

Shaped reward (Eq. 7):
    r_φ(x, y, G) = a * f^sem_φ(s_sem) + (1-a) * f^topo_φ(s_topo) - λ * u(G)

Linear calibrators (Eq. 8):
    f^sem_φ(z) = γ_sem * z + b_sem
    f^topo_φ(z) = γ_topo * z + b_topo
"""

import numpy as np
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass


@dataclass
class RewardComponents:
    """Container for reward computation components."""
    total_reward: float
    semantic_component: float
    topology_component: float
    uncertainty_penalty: float
    raw_semantic_score: float
    raw_topology_score: float
    raw_uncertainty: float
    peer_pressure_penalty: float = 0.0

    @property
    def total(self) -> float:
        return self.total_reward

    @property
    def semantic(self) -> float:
        return self.semantic_component

    @property
    def topology(self) -> float:
        return self.topology_component

    def __getitem__(self, key: str) -> float:
        mapping = {
            "total": self.total_reward,
            "total_reward": self.total_reward,
            "semantic": self.semantic_component,
            "semantic_component": self.semantic_component,
            "topology": self.topology_component,
            "topology_component": self.topology_component,
            "uncertainty_penalty": self.uncertainty_penalty,
            "raw_semantic_score": self.raw_semantic_score,
            "raw_topology_score": self.raw_topology_score,
            "raw_uncertainty": self.raw_uncertainty,
            "peer_pressure_penalty": self.peer_pressure_penalty,
        }
        if key in mapping:
            return mapping[key]
        raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        return key in [
            "total", "total_reward", "semantic", "semantic_component",
            "topology", "topology_component", "uncertainty_penalty",
            "raw_semantic_score", "raw_topology_score", "raw_uncertainty",
            "peer_pressure_penalty",
        ]

    def to_dict(self) -> Dict[str, float]:
        return {
            "total": self.total_reward,
            "semantic": self.semantic_component,
            "topology": self.topology_component,
            "uncertainty_penalty": self.uncertainty_penalty,
            "raw_semantic_score": self.raw_semantic_score,
            "raw_topology_score": self.raw_topology_score,
            "raw_uncertainty": self.raw_uncertainty,
            "peer_pressure_penalty": self.peer_pressure_penalty,
        }


class LinearCalibrator:
    """
    Linear calibrator for score transformation.

    Based on Equation (8):
        f_φ(z) = γ * z + b

    Provides monotonic transformation with optional range mapping and learnable parameters.
    """

    def __init__(
        self,
        gamma: float = 1.0,
        bias: float = 0.0,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
        requires_grad: bool = True
    ):
        """
        Initialize linear calibrator.

        Args:
            gamma: Scale parameter (slope)
            bias: Bias parameter (intercept)
            min_val: Optional minimum value of input range
            max_val: Optional maximum value of input range
            requires_grad: Whether parameters are learnable
        """
        self.gamma = gamma
        self.bias = bias
        self.min_val = min_val
        self.max_val = max_val
        self.requires_grad = requires_grad

    def calibrate(self, z: float) -> float:
        """Calibrate input score with optional range normalization and clamping to [0, 1]."""
        if self.min_val is not None and self.max_val is not None:
            if self.max_val > self.min_val:
                norm_z = (z - self.min_val) / (self.max_val - self.min_val)
            else:
                norm_z = z
        else:
            norm_z = z

        result = self.gamma * norm_z + self.bias
        return float(np.clip(result, 0.0, 1.0))

    def forward(self, z: float) -> float:
        """Apply linear transformation."""
        return self.gamma * z + self.bias

    def __call__(self, z: float) -> float:
        return self.forward(z)

    def get_params(self) -> Dict[str, float]:
        return {"gamma": self.gamma, "bias": self.bias}

    def set_params(self, gamma: float, bias: float) -> None:
        self.gamma = gamma
        self.bias = bias

    def fit(
        self,
        scores: np.ndarray,
        targets: np.ndarray,
        enforce_monotonic: bool = True,
    ) -> 'LinearCalibrator':
        """Fit the affine calibrator on an explicit held-out calibration set."""
        scores = np.asarray(scores, dtype=float).reshape(-1)
        targets = np.asarray(targets, dtype=float).reshape(-1)
        if scores.size == 0 or scores.size != targets.size:
            raise ValueError("scores and targets must be non-empty arrays of equal length")
        design = np.column_stack([scores, np.ones_like(scores)])
        gamma, bias = np.linalg.lstsq(design, targets, rcond=None)[0]
        self.gamma = max(0.0, float(gamma)) if enforce_monotonic else float(gamma)
        self.bias = float(bias)
        return self


class SemanticScorer:
    """
    Compute semantic score balancing task success, factuality, and hallucination.

    Based on Equation (2):
        s_sem(x, y) = β₁ * q_fact + β₂ * q_task - β₃ * q_hall
    """

    def __init__(
        self,
        beta_fact: float = 0.4,
        beta_task: float = 0.4,
        beta_hall: float = 0.2,
        beta_1: Optional[float] = None,
        beta_2: Optional[float] = None,
        beta_3: Optional[float] = None,
        verifier=None
    ):
        """
        Initialize semantic scorer.

        Args:
            beta_fact: Weight for factuality score (or beta_1)
            beta_task: Weight for task-specific metric (or beta_2)
            beta_hall: Weight for hallucination penalty (or beta_3)
            verifier: Optional verifier for fact checking
        """
        self.beta_fact = beta_1 if beta_1 is not None else beta_fact
        self.beta_task = beta_2 if beta_2 is not None else beta_task
        self.beta_hall = beta_3 if beta_3 is not None else beta_hall
        self.verifier = verifier

    def score(
        self,
        prompt: str,
        response: str,
        graph=None,
        task_score: Optional[float] = None,
        fact_scores: Optional[Dict[str, float]] = None,
        hallucination_score: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Compute granular semantic score dictionary.
        """
        q_fact = self._compute_factuality(graph, fact_scores)
        q_task = task_score if task_score is not None else 0.5
        q_hall = hallucination_score if hallucination_score is not None else 0.0

        return {
            "fact": float(q_fact),
            "task": float(q_task),
            "hallucination": float(q_hall),
            "combined": float(self.compute_combined_score(q_fact, q_task, q_hall))
        }

    def compute_combined_score(
        self,
        fact_score: float,
        task_score: float,
        hallucination_score: float
    ) -> float:
        """Compute combined semantic score from component scores."""
        return (
            self.beta_fact * fact_score
            + self.beta_task * task_score
            - self.beta_hall * hallucination_score
        )

    def compute(
        self,
        prompt: str,
        response: str,
        graph=None,
        task_score: Optional[float] = None,
        fact_scores: Optional[Dict[str, float]] = None,
        hallucination_score: Optional[float] = None
    ) -> float:
        """
        Compute semantic score for a response.

        Args:
            prompt: Input prompt
            response: Model response
            graph: Topology graph (for node-level scoring)
            task_score: Pre-computed task metric (e.g., exact match, ROUGE)
            fact_scores: Dict of fact scores per node
            hallucination_score: Pre-computed hallucination penalty

        Returns:
            Semantic score (higher is better)
        """
        scores = self.score(
            prompt=prompt,
            response=response,
            graph=graph,
            task_score=task_score,
            fact_scores=fact_scores,
            hallucination_score=hallucination_score
        )
        return scores["combined"]

    def _compute_factuality(
        self,
        graph,
        fact_scores: Optional[Dict[str, float]] = None
    ) -> float:
        """Aggregate factuality from node-level scores."""
        if graph is None or len(graph.nodes) == 0:
            return 0.5

        if fact_scores is None:
            # Use node correctness probabilities
            scores = [node.correctness_prob for node in graph.nodes.values()]
        else:
            scores = [fact_scores.get(node_id, 0.5) for node_id in graph.nodes]

        return np.mean(scores)


class ShapedReward:
    """
    Shaped reward combining semantic, topology, and uncertainty signals.

    Based on Equation (7):
        r_φ(x, y, G) = a * f^sem_φ(s_sem) + (1-a) * f^topo_φ(s_topo) - λ * u(G)
    """

    def __init__(
        self,
        a: float = 0.6,
        lambda_uncertainty: float = 0.5,
        lambda_peer_pressure: float = 0.0,
        semantic_calibrator: Optional[LinearCalibrator] = None,
        topology_calibrator: Optional[LinearCalibrator] = None,
        semantic_scorer: Optional[SemanticScorer] = None
    ):
        """
        Initialize shaped reward.

        Args:
            a: Mixing parameter between semantic (a) and topology (1-a)
            lambda_uncertainty: Weight for uncertainty penalty
            semantic_calibrator: Linear calibrator for semantic scores
            topology_calibrator: Linear calibrator for topology scores
            semantic_scorer: SemanticScorer instance
        """
        self.a = a
        self.lambda_uncertainty = lambda_uncertainty
        self.lambda_peer_pressure = lambda_peer_pressure

        self.sem_calibrator = semantic_calibrator or LinearCalibrator(
            gamma=1.0, bias=0.0
        )
        self.topo_calibrator = topology_calibrator or LinearCalibrator(
            gamma=1.0, bias=0.0
        )
        self.semantic_scorer = semantic_scorer or SemanticScorer()

    def compute(
        self,
        semantic_score: Optional[float] = None,
        topology_score: Optional[float] = None,
        uncertainty: float = 0.0,
        prompt: Optional[str] = None,
        response: Optional[str] = None,
        graph=None,
        task_score: Optional[float] = None,
        peer_penalty: float = 0.0,
        peer_pressure_rate: float = 0.0,
        **kwargs
    ) -> RewardComponents:
        """
        Compute shaped reward from scores or raw inputs.

        Based on Equation (7):
            r_φ = a * f^sem(s_sem) + (1-a) * f^topo(s_topo) - λ * u(G)
                  - peer_penalty - λ_peer * peer_pressure_rate

        Args:
            semantic_score: Raw semantic score s_sem (computed if None and prompt/response given)
            topology_score: Raw topology score s_topo (computed/defaulted if None)
            uncertainty: Total uncertainty u(G)
            prompt: Optional prompt text
            response: Optional response text
            graph: Optional topology graph
            task_score: Optional task metric score
            peer_penalty: Optional multi-agent peer pressure penalty

        Returns:
            RewardComponents with total reward and breakdown
        """
        if semantic_score is None:
            if prompt is not None and response is not None:
                semantic_score = self.semantic_scorer.compute(
                    prompt=prompt,
                    response=response,
                    graph=graph,
                    task_score=task_score
                )
            else:
                semantic_score = 0.5

        if topology_score is None:
            if graph is not None:
                from .topology import TopologyScorer
                topology_score = TopologyScorer().compute_score(graph)
            else:
                topology_score = 0.5

        # Apply calibrators
        calibrated_sem = self.sem_calibrator(semantic_score)
        calibrated_topo = self.topo_calibrator(topology_score)

        # Compute weighted components
        semantic_component = self.a * calibrated_sem
        topology_component = (1 - self.a) * calibrated_topo
        uncertainty_penalty = self.lambda_uncertainty * uncertainty

        # Total reward: Equation (7) - optional peer penalty
        peer_pressure_penalty = float(peer_penalty + self.lambda_peer_pressure * peer_pressure_rate)
        total_reward = semantic_component + topology_component - uncertainty_penalty - peer_pressure_penalty

        return RewardComponents(
            total_reward=float(total_reward),
            semantic_component=float(semantic_component),
            topology_component=float(topology_component),
            uncertainty_penalty=float(uncertainty_penalty),
            raw_semantic_score=float(semantic_score),
            raw_topology_score=float(topology_score),
            raw_uncertainty=float(uncertainty),
            peer_pressure_penalty=peer_pressure_penalty,
        )

    def compute_from_inputs(
        self,
        prompt: str,
        response: str,
        graph,
        uncertainty: float,
        task_score: Optional[float] = None,
        topology_score: Optional[float] = None
    ) -> RewardComponents:
        """Compute shaped reward directly from raw inputs."""
        return self.compute(
            prompt=prompt,
            response=response,
            graph=graph,
            uncertainty=uncertainty,
            task_score=task_score,
            topology_score=topology_score
        )

    def compute_reward_difference(
        self,
        reward_pos: RewardComponents,
        reward_neg: RewardComponents
    ) -> float:
        """
        Compute reward difference for preference pair.

        Δr_φ = r_φ(x, y+, G+) - r_φ(x, y-, G-)
        """
        return reward_pos.total_reward - reward_neg.total_reward

    def get_params(self) -> Dict[str, Any]:
        """Get all parameters."""
        return {
            "a": self.a,
            "lambda_uncertainty": self.lambda_uncertainty,
            "lambda_peer_pressure": self.lambda_peer_pressure,
            "sem_calibrator": self.sem_calibrator.get_params(),
            "topo_calibrator": self.topo_calibrator.get_params()
        }

    def set_mixing_param(self, a: float) -> None:
        """Set the semantic/topology mixing parameter."""
        self.a = np.clip(a, 0.0, 1.0)

    def set_uncertainty_weight(self, lambda_u: float) -> None:
        """Set the uncertainty penalty weight."""
        self.lambda_uncertainty = max(0.0, lambda_u)

    def fit_calibrators(
        self,
        semantic_scores: np.ndarray,
        topology_scores: np.ndarray,
        targets: np.ndarray,
        enforce_monotonic: bool = True,
    ) -> None:
        """Fit both reward calibrators from explicit held-out targets."""
        self.sem_calibrator.fit(semantic_scores, targets, enforce_monotonic)
        self.topo_calibrator.fit(topology_scores, targets, enforce_monotonic)


class RewardDifferenceComputer:
    """
    Compute reward differences for preference pairs.

    Used in the TUR-DPO loss to augment the DPO margin.
    """

    def __init__(
        self,
        shaped_reward: Optional[ShapedReward] = None,
        gamma: float = 1.0,
        a: Optional[float] = None,
        lambda_uncertainty: Optional[float] = None,
        lambda_peer_pressure: Optional[float] = None,
    ):
        """
        Initialize reward difference computer.

        Args:
            shaped_reward: ShapedReward instance
            gamma: Scaling factor for reward difference in loss
            a: Optional mixing parameter if constructing default ShapedReward
            lambda_uncertainty: Optional uncertainty penalty if constructing default ShapedReward
        """
        if shaped_reward is not None:
            self.shaped_reward = shaped_reward
        else:
            kwargs = {}
            if a is not None:
                kwargs['a'] = a
            if lambda_uncertainty is not None:
                kwargs['lambda_uncertainty'] = lambda_uncertainty
            if lambda_peer_pressure is not None:
                kwargs['lambda_peer_pressure'] = lambda_peer_pressure
            self.shaped_reward = ShapedReward(**kwargs)

        self.gamma = gamma

    def compute(
        self,
        sem_score_pos: float,
        sem_score_neg: float,
        topo_score_pos: float,
        topo_score_neg: float,
        uncertainty_pos: float,
        uncertainty_neg: float,
        peer_pressure_pos: float = 0.0,
        peer_pressure_neg: float = 0.0,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute scaled reward difference for loss computation.

        Returns γ * Δr_φ for use in TUR-DPO loss.
        """
        reward_pos = self.shaped_reward.compute(
            semantic_score=sem_score_pos,
            topology_score=topo_score_pos,
            uncertainty=uncertainty_pos,
            peer_pressure_rate=peer_pressure_pos,
        )

        reward_neg = self.shaped_reward.compute(
            semantic_score=sem_score_neg,
            topology_score=topo_score_neg,
            uncertainty=uncertainty_neg,
            peer_pressure_rate=peer_pressure_neg,
        )

        delta_reward = self.shaped_reward.compute_reward_difference(reward_pos, reward_neg)
        scaled_delta = self.gamma * delta_reward

        return scaled_delta, {
            "delta_reward": delta_reward,
            "gamma_delta_reward": scaled_delta,
            "reward_pos": reward_pos.total_reward,
            "reward_neg": reward_neg.total_reward,
            "sem_component_pos": reward_pos.semantic_component,
            "sem_component_neg": reward_neg.semantic_component,
            "topo_component_pos": reward_pos.topology_component,
            "topo_component_neg": reward_neg.topology_component,
        }

    def compute_difference(
        self,
        prompt: str,
        response_pos: str,
        response_neg: str,
        graph_pos=None,
        graph_neg=None,
        uncertainty_pos: float = 0.0,
        uncertainty_neg: float = 0.0,
        **kwargs
    ) -> float:
        """
        Convenience method to compute raw reward difference from input pairs.
        """
        reward_pos = self.shaped_reward.compute(
            prompt=prompt,
            response=response_pos,
            graph=graph_pos,
            uncertainty=uncertainty_pos,
            **kwargs
        )
        reward_neg = self.shaped_reward.compute(
            prompt=prompt,
            response=response_neg,
            graph=graph_neg,
            uncertainty=uncertainty_neg,
            **kwargs
        )
        return self.shaped_reward.compute_reward_difference(reward_pos, reward_neg)
