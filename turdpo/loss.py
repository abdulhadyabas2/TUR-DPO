"""
Loss Module for TUR-DPO

This module implements the TUR-DPO loss functions for training.

Based on Equations (9), (10), and (11) from the paper:

Pairwise loss (Eq. 9):
    L_TUR-DPO = -w * log σ(β * [Δlog π_θ - Δlog π_ref] + γ * Δr_φ)

Listwise loss (Eq. 10):
    L_list = -w * Σ_i log [exp(z_i) / Σ_j exp(z_j)]
    where z_i = β * (log π_θ(y_i|x) - log π_ref(y_i|x)) + γ * r_φ(x, y_i, G_i)

Margin definition (Eq. 11):
    m_θ = β * [Δlog π_θ - Δlog π_ref] + γ * Δr_φ
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List


class TURDPOLoss(nn.Module):
    """
    TUR-DPO pairwise loss with shaped reward and uncertainty weighting.

    Based on Equation (9):
        L = -w * log σ(β * [Δlog π_θ - Δlog π_ref] + γ * Δr_φ)

    This loss modifies standard DPO by:
    1. Adding shaped reward difference (γ * Δr_φ) to the margin
    2. Weighting the loss by pair uncertainty weight (w)
    """

    def __init__(
        self,
        beta: float = 2.0,
        gamma: float = 1.0,
        label_smoothing: float = 0.0,
        average_log_prob: bool = False
    ):
        """
        Initialize TUR-DPO loss.

        Args:
            beta: Temperature parameter controlling sharpness (default: 2.0)
            gamma: Weight for shaped reward difference (default: 1.0)
            label_smoothing: Label smoothing factor (default: 0.0)
            average_log_prob: Whether to average log probs over sequence length
        """
        super().__init__()
        if beta <= 0:
            raise ValueError("beta must be positive")
        if gamma < 0:
            raise ValueError("gamma must be non-negative")
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1)")
        self.beta = beta
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.average_log_prob = average_log_prob

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
        reward_diff: torch.Tensor,
        weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute TUR-DPO loss.

        Args:
            policy_chosen_logps: Log probs of chosen responses under policy [batch]
            policy_rejected_logps: Log probs of rejected responses under policy [batch]
            reference_chosen_logps: Log probs of chosen responses under reference [batch]
            reference_rejected_logps: Log probs of rejected responses under reference [batch]
            reward_diff: Shaped reward difference Δr_φ [batch]
            weights: Per-pair uncertainty weights w [batch]

        Returns:
            Tuple of (loss, metrics_dict)
        """
        # Compute policy log ratio: Δlog π_θ = log π_θ(y+|x) - log π_θ(y-|x)
        policy_log_ratio = policy_chosen_logps - policy_rejected_logps

        # Compute reference log ratio: Δlog π_ref
        reference_log_ratio = reference_chosen_logps - reference_rejected_logps

        # Compute margin: m_θ = β * [Δlog π_θ - Δlog π_ref] + γ * Δr_φ
        # Based on Equation (11)
        logits = self.beta * (policy_log_ratio - reference_log_ratio) + self.gamma * reward_diff

        # Apply label smoothing if specified
        if self.label_smoothing > 0:
            # Soft labels for binary classification
            soft_label = 1.0 - self.label_smoothing
            losses = -soft_label * F.logsigmoid(logits) - (1 - soft_label) * F.logsigmoid(-logits)
        else:
            # Standard binary cross-entropy: -log σ(m_θ)
            losses = -F.logsigmoid(logits)

        # Apply uncertainty weights: L = -w * log σ(m_θ)
        weights = weights.to(device=losses.device, dtype=losses.dtype)
        if weights.shape != losses.shape:
            raise ValueError("weights must have one value per preference pair")
        weighted_losses = weights * losses

        # Average over batch
        loss = weighted_losses.mean()

        # Compute metrics
        with torch.no_grad():
            chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps)
            rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps)
            reward_accuracies = (chosen_rewards > rejected_rewards).float()
            reward_margins = chosen_rewards - rejected_rewards

        metrics = {
            "loss": loss.detach(),
            "chosen_rewards": chosen_rewards.mean().detach(),
            "rejected_rewards": rejected_rewards.mean().detach(),
            "reward_accuracies": reward_accuracies.mean().detach(),
            "reward_margins": reward_margins.mean().detach(),
            "policy_log_ratio": policy_log_ratio.mean().detach(),
            "reference_log_ratio": reference_log_ratio.mean().detach(),
            "logits": logits.mean().detach(),
            "weights": weights.mean().detach(),
            "reward_diff": reward_diff.mean().detach(),
        }

        return loss, metrics

    def compute_logps(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute log probabilities from logits.

        Args:
            logits: Model output logits [batch, seq_len, vocab_size]
            labels: Target token ids [batch, seq_len]
            attention_mask: Mask for valid positions [batch, seq_len]

        Returns:
            Log probabilities [batch]
        """
        if attention_mask is None:
            attention_mask = torch.ones_like(labels)

        # Shift for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_mask = attention_mask[..., 1:].contiguous()

        # In DPO, labels might have -100 for ignored prompt tokens or padding
        loss_mask = (shift_labels != -100) & (shift_mask == 1)
        clamped_shift_labels = shift_labels.clone()
        clamped_shift_labels[~loss_mask] = 0

        # Compute log probs
        log_probs = F.log_softmax(shift_logits, dim=-1)

        # Gather log probs for target tokens
        per_token_logps = torch.gather(
            log_probs,
            dim=-1,
            index=clamped_shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        # Apply mask and sum/average
        masked_logps = per_token_logps * loss_mask

        if self.average_log_prob:
            return masked_logps.sum(-1) / loss_mask.sum(-1).clamp(min=1)
        else:
            return masked_logps.sum(-1)


class ListwiseTURDPOLoss(nn.Module):
    """
    Listwise TUR-DPO loss using Plackett-Luce utilities.

    Based on Equation (10):
        L_list = -w * Σ_{i∈P} log [exp(z_i) / Σ_j exp(z_j)]
        where z_i = β * (log π_θ(y_i|x) - log π_ref(y_i|x)) + γ * r_φ(x, y_i, G_i)

    Uses multiple candidates per prompt for reduced variance.
    """

    def __init__(
        self,
        beta: float = 2.0,
        gamma: float = 1.0,
        num_candidates: int = 4,
        tau: float = 0.1,
        **kwargs
    ):
        """
        Initialize listwise TUR-DPO loss.

        Args:
            beta: Temperature parameter
            gamma: Weight for shaped rewards
            num_candidates: Number of candidates per prompt (k)
            tau: Temperature parameter for Plackett-Luce (optional)
        """
        super().__init__()
        if beta <= 0 or gamma < 0 or num_candidates < 2 or tau <= 0:
            raise ValueError("invalid listwise TUR-DPO configuration")
        self.beta = beta
        self.gamma = gamma
        self.num_candidates = num_candidates
        self.tau = tau

    def forward(
        self,
        policy_logps: torch.Tensor,
        reference_logps: torch.Tensor,
        rewards: torch.Tensor,
        preferences: Optional[torch.Tensor] = None,
        weight: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute listwise TUR-DPO loss.

        Args:
            policy_logps: Log probs under policy [batch, k]
            reference_logps: Log probs under reference [batch, k]
            rewards: Shaped rewards r_φ for each candidate [batch, k]
            preferences: Optional preference indicators (1 for preferred) [batch, k]
            weight: Optional per-pair uncertainty weight [batch]

        Returns:
            Tuple of (loss, metrics_dict)
        """
        if preferences is None:
            # Top-1 reward is preferred by default
            preferences = (rewards == rewards.max(dim=-1, keepdim=True)[0]).float()
        elif preferences.dim() == 1 and weight is None:
            # 4 args called as loss_fn(policy, ref, rewards, weight)
            weight = preferences
            preferences = (rewards == rewards.max(dim=-1, keepdim=True)[0]).float()

        if weight is None:
            weight = torch.ones(policy_logps.shape[0], device=policy_logps.device)
        weight = weight.to(device=policy_logps.device, dtype=policy_logps.dtype)
        # Compute utilities: z_i = β * (log π_θ - log π_ref) + γ * r_φ
        utilities = self.beta * (policy_logps - reference_logps) + self.gamma * rewards

        # Plackett-Luce loss over preferred items
        # L = -Σ_{i∈P} log [exp(z_i) / Σ_j exp(z_j)]
        log_softmax_utilities = F.log_softmax(utilities, dim=-1)

        # Mask for preferred items
        preference_mask = preferences.float()

        # Sum log probs of preferred items
        preferred_log_probs = (log_softmax_utilities * preference_mask).sum(dim=-1)
        num_preferred = preference_mask.sum(dim=-1).clamp(min=1)

        # Normalize by number of preferred items
        per_sample_loss = -preferred_log_probs / num_preferred

        # Apply uncertainty weight
        weighted_loss = weight * per_sample_loss

        # Average over batch
        loss = weighted_loss.mean()

        # Metrics
        with torch.no_grad():
            avg_utility = utilities.mean()
            max_utility = utilities.max(dim=-1)[0].mean()
            min_utility = utilities.min(dim=-1)[0].mean()

        metrics = {
            "loss": loss.detach(),
            "avg_utility": avg_utility.detach(),
            "max_utility": max_utility.detach(),
            "min_utility": min_utility.detach(),
            "weight": weight.mean().detach(),
        }

        return loss, metrics

    def compute_logps(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute response-token log probabilities for listwise batches."""
        if attention_mask is None:
            attention_mask = torch.ones_like(labels)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_mask = attention_mask[..., 1:].contiguous()
        loss_mask = (shift_labels != -100) & (shift_mask == 1)
        safe_labels = shift_labels.masked_fill(~loss_mask, 0)
        log_probs = F.log_softmax(shift_logits, dim=-1)
        per_token_logps = torch.gather(
            log_probs, dim=-1, index=safe_labels.unsqueeze(-1)
        ).squeeze(-1)
        masked_logps = per_token_logps * loss_mask
        return masked_logps.sum(dim=-1)


class DPOLoss(nn.Module):
    """
    Standard DPO loss for comparison baseline.

    L_DPO = -log σ(β * [log π_θ(y+|x)/π_ref(y+|x) - log π_θ(y-|x)/π_ref(y-|x)])
    """

    def __init__(
        self,
        beta: float = 0.1,
        label_smoothing: float = 0.0,
        average_log_prob: bool = False
    ):
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.average_log_prob = average_log_prob

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute standard DPO loss.

        Args:
            policy_chosen_logps: Log probs of chosen under policy
            policy_rejected_logps: Log probs of rejected under policy
            reference_chosen_logps: Log probs of chosen under reference
            reference_rejected_logps: Log probs of rejected under reference

        Returns:
            Tuple of (loss, metrics_dict)
        """
        # Log ratios
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps

        # DPO logits
        logits = self.beta * (pi_logratios - ref_logratios)

        # Loss
        if self.label_smoothing > 0:
            soft_label = 1.0 - self.label_smoothing
            losses = -soft_label * F.logsigmoid(logits) - (1 - soft_label) * F.logsigmoid(-logits)
        else:
            losses = -F.logsigmoid(logits)

        loss = losses.mean()

        # Metrics
        with torch.no_grad():
            chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps)
            rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps)
            reward_accuracies = (chosen_rewards > rejected_rewards).float()

        metrics = {
            "loss": loss.detach(),
            "chosen_rewards": chosen_rewards.mean().detach(),
            "rejected_rewards": rejected_rewards.mean().detach(),
            "reward_accuracies": reward_accuracies.mean().detach(),
            "logits": logits.mean().detach(),
        }

        return loss, metrics


def compute_reference_free_logps(
    policy_logps_chosen: torch.Tensor,
    policy_logps_rejected: torch.Tensor,
    beta: float = 0.1,
    reference_free: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute effective log probs for reference-free methods (like ORPO).

    When reference_free=True, uses policy as its own reference.

    Args:
        policy_logps_chosen: Policy log probs for chosen
        policy_logps_rejected: Policy log probs for rejected
        beta: Temperature parameter
        reference_free: Whether to use reference-free formulation

    Returns:
        Tuple of (effective_chosen_logps, effective_rejected_logps)
    """
    if reference_free:
        # Use odds ratio (ORPO-style)
        return policy_logps_chosen, policy_logps_rejected
    else:
        return policy_logps_chosen, policy_logps_rejected


class IPOLoss(nn.Module):
    """
    Identity Preference Optimization (IPO) loss for comparison.

    Uses squared hinge loss instead of logistic loss.
    """

    def __init__(self, beta: float = 0.1):
        super().__init__()
        self.beta = beta

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute IPO loss."""
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps

        # IPO loss: (logits - 0.5)^2
        logits = pi_logratios - ref_logratios
        losses = (logits - 1 / (2 * self.beta)) ** 2

        loss = losses.mean()

        metrics = {
            "loss": loss.detach(),
            "logits": logits.mean().detach(),
        }

        return loss, metrics
