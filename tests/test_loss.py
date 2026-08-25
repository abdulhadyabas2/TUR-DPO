"""
Tests for TUR-DPO loss module.
"""

import pytest
import torch
import torch.nn as nn

from turdpo.loss import TURDPOLoss, ListwiseTURDPOLoss, DPOLoss, IPOLoss


class TestTURDPOLoss:
    """Tests for TURDPOLoss class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.loss_fn = TURDPOLoss(beta=2.0, gamma=1.0)
        self.batch_size = 4

    def test_basic_loss_computation(self):
        """Test basic loss computation."""
        log_probs_pos = torch.randn(self.batch_size, requires_grad=True)
        log_probs_neg = torch.randn(self.batch_size, requires_grad=True)
        ref_log_probs_pos = torch.randn(self.batch_size)
        ref_log_probs_neg = torch.randn(self.batch_size)
        reward_diff = torch.randn(self.batch_size)
        weights = torch.ones(self.batch_size)

        loss, metrics = self.loss_fn(
            log_probs_pos, log_probs_neg,
            ref_log_probs_pos, ref_log_probs_neg,
            reward_diff, weights
        )

        assert loss.shape == ()  # Scalar
        assert loss.requires_grad
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
        assert "loss" in metrics
        assert "chosen_rewards" in metrics

    def test_gradients_flow(self):
        """Test that gradients flow correctly."""
        log_probs_pos = torch.randn(self.batch_size, requires_grad=True)
        log_probs_neg = torch.randn(self.batch_size, requires_grad=True)
        ref_log_probs_pos = torch.randn(self.batch_size)
        ref_log_probs_neg = torch.randn(self.batch_size)
        reward_diff = torch.randn(self.batch_size)
        weights = torch.ones(self.batch_size)

        loss, _ = self.loss_fn(
            log_probs_pos, log_probs_neg,
            ref_log_probs_pos, ref_log_probs_neg,
            reward_diff, weights
        )
        loss.backward()

        assert log_probs_pos.grad is not None
        assert log_probs_neg.grad is not None

    def test_weight_effect(self):
        """Test that weights affect the loss."""
        torch.manual_seed(42)
        log_probs_pos = torch.randn(self.batch_size)
        log_probs_neg = torch.randn(self.batch_size)
        ref_log_probs_pos = torch.randn(self.batch_size)
        ref_log_probs_neg = torch.randn(self.batch_size)
        reward_diff = torch.randn(self.batch_size)

        loss_equal, _ = self.loss_fn(
            log_probs_pos.clone(), log_probs_neg.clone(),
            ref_log_probs_pos, ref_log_probs_neg,
            reward_diff, torch.ones(self.batch_size)
        )

        loss_varied, _ = self.loss_fn(
            log_probs_pos.clone(), log_probs_neg.clone(),
            ref_log_probs_pos, ref_log_probs_neg,
            reward_diff, torch.tensor([0.1, 0.2, 0.8, 1.0])
        )

        # Different weights should give different losses
        assert not torch.allclose(loss_equal, loss_varied)

    def test_beta_gamma_effect(self):
        """Test that beta and gamma parameters affect loss."""
        log_probs = torch.randn(self.batch_size)
        ref_log_probs = torch.randn(self.batch_size)
        reward_diff = torch.randn(self.batch_size)
        weights = torch.ones(self.batch_size)

        loss_fn_low = TURDPOLoss(beta=0.5, gamma=0.5)
        loss_fn_high = TURDPOLoss(beta=4.0, gamma=2.0)

        loss_low, _ = loss_fn_low(
            log_probs.clone(), -log_probs.clone(),
            ref_log_probs, -ref_log_probs,
            reward_diff, weights
        )
        loss_high, _ = loss_fn_high(
            log_probs.clone(), -log_probs.clone(),
            ref_log_probs, -ref_log_probs,
            reward_diff, weights
        )

        # Different parameters should give different losses
        assert not torch.allclose(loss_low, loss_high)


class TestListwiseTURDPOLoss:
    """Tests for ListwiseTURDPOLoss class."""

    def test_basic_computation(self):
        """Test listwise loss computation."""
        loss_fn = ListwiseTURDPOLoss(beta=2.0, gamma=1.0, tau=0.1)

        batch_size = 2
        list_size = 4

        log_probs = torch.randn(batch_size, list_size, requires_grad=True)
        ref_log_probs = torch.randn(batch_size, list_size)
        rewards = torch.randn(batch_size, list_size)
        weights = torch.ones(batch_size)

        loss, metrics = loss_fn(log_probs, ref_log_probs, rewards, weight=weights)

        assert loss.shape == ()
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
        assert "avg_utility" in metrics

    def test_gradient_flow(self):
        """Test gradient flow in listwise loss."""
        loss_fn = ListwiseTURDPOLoss(beta=2.0, gamma=1.0)

        log_probs = torch.randn(2, 4, requires_grad=True)
        ref_log_probs = torch.randn(2, 4)
        rewards = torch.randn(2, 4)
        weights = torch.ones(2)

        loss, _ = loss_fn(log_probs, ref_log_probs, rewards, weight=weights)
        loss.backward()

        assert log_probs.grad is not None


class TestDPOLoss:
    """Tests for baseline DPO loss."""

    def test_basic_computation(self):
        """Test DPO loss computation."""
        loss_fn = DPOLoss(beta=0.1)

        log_probs_pos = torch.randn(4, requires_grad=True)
        log_probs_neg = torch.randn(4, requires_grad=True)
        ref_log_probs_pos = torch.randn(4)
        ref_log_probs_neg = torch.randn(4)

        loss, metrics = loss_fn(
            log_probs_pos, log_probs_neg,
            ref_log_probs_pos, ref_log_probs_neg
        )

        assert loss.shape == ()
        assert not torch.isnan(loss)
        assert "chosen_rewards" in metrics

    def test_positive_preference(self):
        """Test loss when positive is preferred."""
        loss_fn = DPOLoss(beta=0.5)

        # Make positive clearly better
        log_probs_pos = torch.tensor([0.0, 0.0, 0.0, 0.0])
        log_probs_neg = torch.tensor([-5.0, -5.0, -5.0, -5.0])
        ref_log_probs_pos = torch.tensor([0.0, 0.0, 0.0, 0.0])
        ref_log_probs_neg = torch.tensor([-5.0, -5.0, -5.0, -5.0])

        loss, _ = loss_fn(
            log_probs_pos, log_probs_neg,
            ref_log_probs_pos, ref_log_probs_neg
        )

        # Loss should be small when preferences are learned
        assert loss < 1.0


class TestIPOLoss:
    """Tests for IPO loss."""

    def test_basic_computation(self):
        """Test IPO loss computation."""
        loss_fn = IPOLoss(beta=0.5)

        log_probs_pos = torch.randn(4, requires_grad=True)
        log_probs_neg = torch.randn(4, requires_grad=True)
        ref_log_probs_pos = torch.randn(4)
        ref_log_probs_neg = torch.randn(4)

        loss, metrics = loss_fn(
            log_probs_pos, log_probs_neg,
            ref_log_probs_pos, ref_log_probs_neg
        )

        assert loss.shape == ()
        assert not torch.isnan(loss)
        assert loss >= 0  # IPO loss uses squared term


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
