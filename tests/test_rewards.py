"""
Tests for TUR-DPO rewards module.
"""

import pytest
import torch
import numpy as np

from turdpo.rewards import (
    ShapedReward,
    SemanticScorer,
    LinearCalibrator,
    RewardDifferenceComputer
)
from turdpo.topology import TopologyGraph, Node, Edge


def create_test_graph(num_nodes=4, correctness_probs=None, add_cycle=False):
    """Helper to create test graphs."""
    graph = TopologyGraph()

    if correctness_probs is None:
        correctness_probs = [0.8] * num_nodes

    for i in range(num_nodes):
        graph.add_node(Node(
            id=f"n{i}",
            content=f"Step {i}",
            correctness_prob=correctness_probs[i] if i < len(correctness_probs) else 0.5
        ))

    for i in range(num_nodes - 1):
        graph.add_edge(Edge(source_id=f"n{i}", target_id=f"n{i+1}"))

    if add_cycle and num_nodes > 2:
        graph.add_edge(Edge(source_id=f"n{num_nodes-1}", target_id="n0"))

    return graph


class TestLinearCalibrator:
    """Tests for LinearCalibrator class."""

    def test_default_calibration(self):
        """Test default calibration range."""
        calibrator = LinearCalibrator(min_val=0.0, max_val=1.0)

        assert calibrator.calibrate(0.0) == 0.0
        assert calibrator.calibrate(1.0) == 1.0
        assert calibrator.calibrate(0.5) == 0.5

    def test_custom_range(self):
        """Test calibration with custom range."""
        calibrator = LinearCalibrator(min_val=-1.0, max_val=1.0)

        assert calibrator.calibrate(0.0) == 0.5
        assert calibrator.calibrate(-1.0) == 0.0
        assert calibrator.calibrate(1.0) == 1.0

    def test_clamping(self):
        """Test that values are clamped to [0, 1]."""
        calibrator = LinearCalibrator(min_val=0.0, max_val=1.0)

        assert calibrator.calibrate(-0.5) == 0.0
        assert calibrator.calibrate(1.5) == 1.0

    def test_fit_on_held_out_targets(self):
        calibrator = LinearCalibrator()
        calibrator.fit(np.array([0.0, 0.5, 1.0]), np.array([0.1, 0.5, 0.9]))
        assert calibrator.calibrate(0.5) == pytest.approx(0.5, abs=0.05)


class TestSemanticScorer:
    """Tests for SemanticScorer class."""

    def test_default_scores(self):
        """Test default semantic scoring."""
        scorer = SemanticScorer()
        scores = scorer.score("prompt", "response")

        assert "fact" in scores
        assert "task" in scores
        assert "hallucination" in scores
        assert all(0 <= v <= 1 for v in scores.values())

    def test_combined_score(self):
        """Test combined semantic score."""
        scorer = SemanticScorer(beta_1=0.4, beta_2=0.4, beta_3=0.2)

        combined = scorer.compute_combined_score(
            fact_score=0.8,
            task_score=0.9,
            hallucination_score=0.1
        )

        expected = 0.4 * 0.8 + 0.4 * 0.9 - 0.2 * 0.1
        assert abs(combined - expected) < 0.001


class TestShapedReward:
    """Tests for ShapedReward class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.reward = ShapedReward(
            a=0.6,
            lambda_uncertainty=0.5
        )

    def test_basic_computation(self):
        """Test basic reward computation."""
        graph = create_test_graph()

        result = self.reward.compute(
            prompt="Test prompt",
            response="Test response",
            graph=graph,
            uncertainty=0.2
        )

        assert "total" in result
        assert "semantic" in result
        assert "topology" in result
        assert "uncertainty_penalty" in result

    def test_uncertainty_effect(self):
        """Test that higher uncertainty reduces reward."""
        graph = create_test_graph()

        result_low_u = self.reward.compute(
            prompt="Test", response="Test",
            graph=graph, uncertainty=0.1
        )
        result_high_u = self.reward.compute(
            prompt="Test", response="Test",
            graph=graph, uncertainty=0.9
        )

        # Higher uncertainty should give lower reward
        assert result_low_u["total"] > result_high_u["total"]

    def test_peer_pressure_scalar_reduces_reward(self):
        baseline = ShapedReward(a=0.5, lambda_uncertainty=0.0, lambda_peer_pressure=0.0)
        penalized = ShapedReward(a=0.5, lambda_uncertainty=0.0, lambda_peer_pressure=0.5)
        base_value = baseline.compute(semantic_score=0.8, topology_score=0.8)
        penalized_value = penalized.compute(
            semantic_score=0.8, topology_score=0.8, peer_pressure_rate=0.6
        )
        assert penalized_value.total < base_value.total

    def test_semantic_weighting(self):
        """Test semantic vs topology weighting."""
        graph = create_test_graph()

        reward_sem = ShapedReward(a=0.9, lambda_uncertainty=0.0)
        reward_topo = ShapedReward(a=0.1, lambda_uncertainty=0.0)

        result_sem = reward_sem.compute(
            prompt="Test", response="Test", graph=graph, uncertainty=0.0
        )
        result_topo = reward_topo.compute(
            prompt="Test", response="Test", graph=graph, uncertainty=0.0
        )

        # Both should give valid rewards but with different emphasis
        assert result_sem["total"] >= 0
        assert result_topo["total"] >= 0


class TestRewardDifferenceComputer:
    """Tests for RewardDifferenceComputer class."""

    def test_basic_computation(self):
        """Test reward difference computation."""
        computer = RewardDifferenceComputer()

        graph_pos = create_test_graph(correctness_probs=[0.9, 0.9, 0.9])
        graph_neg = create_test_graph(correctness_probs=[0.5, 0.5, 0.5])

        diff = computer.compute_difference(
            prompt="Test",
            response_pos="Good response",
            response_neg="Bad response",
            graph_pos=graph_pos,
            graph_neg=graph_neg,
            uncertainty_pos=0.1,
            uncertainty_neg=0.5
        )

        assert isinstance(diff, float)

    def test_positive_wins(self):
        """Test when positive response clearly wins."""
        computer = RewardDifferenceComputer(a=0.5, lambda_uncertainty=0.3)

        graph_pos = create_test_graph(correctness_probs=[0.95, 0.95, 0.95])
        graph_neg = create_test_graph(correctness_probs=[0.3, 0.3, 0.3], add_cycle=True)

        diff = computer.compute_difference(
            prompt="Test",
            response_pos="Excellent response",
            response_neg="Poor response",
            graph_pos=graph_pos,
            graph_neg=graph_neg,
            uncertainty_pos=0.05,
            uncertainty_neg=0.8
        )

        # Positive should have higher reward, so diff should be positive
        assert diff > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
