"""
Tests for TUR-DPO uncertainty module.
"""

import pytest
import numpy as np

from turdpo.uncertainty import (
    EpistemicUncertainty,
    AleatoricUncertainty,
    UncertaintyEstimator,
    PairWeightComputer
)
from turdpo.topology import TopologyGraph, Node, Edge, TopologyScorer


def create_test_graph(num_nodes=4, correctness_probs=None):
    """Helper to create a test graph."""
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

    return graph


class TestEpistemicUncertainty:
    """Tests for EpistemicUncertainty class."""

    def test_single_graph(self):
        """Single graph should have zero epistemic uncertainty."""
        graph = create_test_graph()
        epistemic = EpistemicUncertainty()

        uncertainty = epistemic.compute([graph])
        assert uncertainty == 0.0

    def test_identical_graphs(self):
        """Identical graphs should have low epistemic uncertainty."""
        graphs = [create_test_graph(num_nodes=4) for _ in range(3)]
        epistemic = EpistemicUncertainty()

        uncertainty = epistemic.compute(graphs)
        # Should be very low since graphs are identical
        assert uncertainty < 0.1

    def test_varied_graphs(self):
        """Varied graphs should have higher epistemic uncertainty."""
        graphs = [
            create_test_graph(num_nodes=3),
            create_test_graph(num_nodes=5),
            create_test_graph(num_nodes=4),
        ]
        epistemic = EpistemicUncertainty()

        uncertainty = epistemic.compute(graphs)
        # Should be higher due to variation
        assert uncertainty > 0

    def test_empty_graphs(self):
        """Empty list should return maximum uncertainty."""
        epistemic = EpistemicUncertainty()
        uncertainty = epistemic.compute([])
        assert uncertainty == 1.0


class TestAleatoricUncertainty:
    """Tests for AleatoricUncertainty class."""

    def test_certain_nodes(self):
        """Nodes with high certainty should have low aleatoric uncertainty."""
        graph = create_test_graph(correctness_probs=[0.99, 0.99, 0.99, 0.99])
        aleatoric = AleatoricUncertainty(tau=0.05)

        uncertainty = aleatoric.compute(graph)
        # High certainty = low aleatoric uncertainty
        assert uncertainty < 0.25

    def test_uncertain_nodes(self):
        """Nodes with 50% certainty should have maximum aleatoric uncertainty."""
        graph = create_test_graph(correctness_probs=[0.5, 0.5, 0.5, 0.5])
        aleatoric = AleatoricUncertainty(tau=0.0)  # No smoothing

        uncertainty = aleatoric.compute(graph)
        # 50% certainty = maximum entropy
        assert uncertainty > 0.6

    def test_smoothing(self):
        """Smoothing should prevent extreme uncertainties."""
        graph = create_test_graph(correctness_probs=[1.0, 1.0, 0.0, 0.0])

        aleatoric_no_smooth = AleatoricUncertainty(tau=0.0)
        aleatoric_smooth = AleatoricUncertainty(tau=0.1)

        u_no_smooth = aleatoric_no_smooth.compute(graph)
        u_smooth = aleatoric_smooth.compute(graph)

        # Smoothing should increase uncertainty for extreme probs
        assert u_smooth > u_no_smooth

    def test_empty_graph(self):
        """Empty graph should return maximum uncertainty."""
        graph = TopologyGraph()
        aleatoric = AleatoricUncertainty()

        uncertainty = aleatoric.compute(graph)
        assert uncertainty == 1.0


class TestUncertaintyEstimator:
    """Tests for combined UncertaintyEstimator."""

    def test_combined_uncertainty(self):
        """Test combined epistemic + aleatoric estimation."""
        graphs = [create_test_graph() for _ in range(3)]

        estimator = UncertaintyEstimator(
            lambda_epi=0.5,
            lambda_ale=0.5,
            tau=0.05
        )

        result = estimator.compute(graphs)

        assert result.total >= 0
        assert result.epistemic >= 0
        assert result.aleatoric >= 0
        assert abs(result.total - (0.5 * result.epistemic + 0.5 * result.aleatoric)) < 0.001

    def test_pair_weight_computation(self):
        """Test pair weight computation."""
        estimator = UncertaintyEstimator()

        # Low uncertainty should give high weight
        weight_low = estimator.compute_pair_weight(0.1, 0.1, tau_w=1.0, w_min=0.05)
        assert weight_low > 0.8

        # High uncertainty should give low weight
        weight_high = estimator.compute_pair_weight(2.0, 2.0, tau_w=1.0, w_min=0.05)
        assert weight_high < 0.5

    def test_weight_clipping(self):
        """Test that weights are clipped to valid range."""
        estimator = UncertaintyEstimator()

        # Very high uncertainty
        weight = estimator.compute_pair_weight(100.0, 100.0, tau_w=1.0, w_min=0.05)
        assert weight >= 0.05
        assert weight <= 1.0

        # Very low uncertainty
        weight = estimator.compute_pair_weight(0.0, 0.0, tau_w=1.0, w_min=0.05)
        assert weight <= 1.0


class TestPairWeightComputer:
    """Tests for PairWeightComputer class."""

    def test_weight_computation(self):
        """Test full pair weight computation pipeline."""
        computer = PairWeightComputer(tau_w=1.2, w_min=0.05)

        graphs_pos = [create_test_graph(correctness_probs=[0.9, 0.9, 0.9])]
        graphs_neg = [create_test_graph(correctness_probs=[0.6, 0.6, 0.6])]

        weight, details = computer.compute_weight(graphs_pos, graphs_neg)

        assert 0.05 <= weight <= 1.0
        assert "u_pos_total" in details
        assert "u_neg_total" in details
        assert "weight" in details

    def test_equal_uncertainty(self):
        """Equal uncertainties should give consistent weights."""
        computer = PairWeightComputer(tau_w=1.0)

        graphs = [create_test_graph()]

        w1, _ = computer.compute_weight(graphs, graphs)
        w2, _ = computer.compute_weight(graphs, graphs)

        assert abs(w1 - w2) < 0.001


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
