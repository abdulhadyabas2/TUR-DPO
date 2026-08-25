"""
Utilities Module for TUR-DPO

This module provides utility functions for metrics, logging, and common operations.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


# ==============================================================================
# Metrics
# ==============================================================================

def compute_exact_match(predictions: List[str], references: List[str]) -> float:
    """
    Compute exact match accuracy.

    Args:
        predictions: List of predicted answers
        references: List of reference answers

    Returns:
        Exact match accuracy (0-1)
    """
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have same length")

    if len(predictions) == 0:
        return 0.0

    matches = sum(
        1 for pred, ref in zip(predictions, references)
        if normalize_answer(pred) == normalize_answer(ref)
    )
    return matches / len(predictions)


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    # Lowercase
    answer = answer.lower()
    # Remove articles
    answer = ' '.join(w for w in answer.split() if w not in ['a', 'an', 'the'])
    # Remove punctuation
    answer = ''.join(c for c in answer if c.isalnum() or c.isspace())
    # Normalize whitespace
    answer = ' '.join(answer.split())
    return answer


def compute_f1(prediction: str, reference: str) -> float:
    """
    Compute token-level F1 score.

    Args:
        prediction: Predicted text
        reference: Reference text

    Returns:
        F1 score (0-1)
    """
    pred_tokens = set(normalize_answer(prediction).split())
    ref_tokens = set(normalize_answer(reference).split())

    if len(pred_tokens) == 0 or len(ref_tokens) == 0:
        return float(pred_tokens == ref_tokens)

    common = pred_tokens & ref_tokens

    if len(common) == 0:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)

    return 2 * precision * recall / (precision + recall)


def compute_rouge_l(prediction: str, reference: str) -> float:
    """
    Compute ROUGE-L score (longest common subsequence based).

    Args:
        prediction: Predicted text
        reference: Reference text

    Returns:
        ROUGE-L F1 score (0-1)
    """
    pred_tokens = prediction.split()
    ref_tokens = reference.split()

    if len(pred_tokens) == 0 or len(ref_tokens) == 0:
        return 0.0

    # Compute LCS length
    m, n = len(pred_tokens), len(ref_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i-1] == ref_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    lcs_length = dp[m][n]

    if lcs_length == 0:
        return 0.0

    precision = lcs_length / len(pred_tokens)
    recall = lcs_length / len(ref_tokens)

    return 2 * precision * recall / (precision + recall)


# ==============================================================================
# Statistical utilities
# ==============================================================================

def bootstrap_ci(
    metric_fn,
    data: np.ndarray,
    labels: np.ndarray,
    n_bootstrap: int = 10000,
    ci_level: float = 0.95,
    seed: int = 42
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a metric.

    Args:
        metric_fn: Function computing metric from (predictions, labels)
        data: Predictions array
        labels: Labels array
        n_bootstrap: Number of bootstrap samples
        ci_level: Confidence level (e.g., 0.95 for 95% CI)
        seed: Random seed

    Returns:
        Tuple of (point_estimate, ci_lower, ci_upper)
    """
    np.random.seed(seed)

    n = len(data)
    point_estimate = metric_fn(data, labels)

    bootstrap_estimates = []
    for _ in range(n_bootstrap):
        indices = np.random.randint(0, n, size=n)
        boot_data = data[indices]
        boot_labels = labels[indices]
        boot_metric = metric_fn(boot_data, boot_labels)
        bootstrap_estimates.append(boot_metric)

    bootstrap_estimates = np.array(bootstrap_estimates)

    alpha = 1 - ci_level
    ci_lower = np.percentile(bootstrap_estimates, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_estimates, 100 * (1 - alpha / 2))

    return point_estimate, ci_lower, ci_upper


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Compute Cohen's d effect size.

    Args:
        group1: First group values
        group2: Second group values

    Returns:
        Cohen's d value
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    return (np.mean(group1) - np.mean(group2)) / pooled_std


def paired_bootstrap_test(
    metric1: np.ndarray,
    metric2: np.ndarray,
    n_bootstrap: int = 10000,
    seed: int = 42
) -> float:
    """
    Paired bootstrap test for statistical significance.

    Args:
        metric1: Metrics for method 1
        metric2: Metrics for method 2
        n_bootstrap: Number of bootstrap samples
        seed: Random seed

    Returns:
        p-value (two-tailed)
    """
    np.random.seed(seed)

    n = len(metric1)
    observed_diff = np.mean(metric1) - np.mean(metric2)

    # Center the differences
    diff = metric1 - metric2
    centered_diff = diff - np.mean(diff)

    count = 0
    for _ in range(n_bootstrap):
        indices = np.random.randint(0, n, size=n)
        boot_diff = np.mean(centered_diff[indices])
        if abs(boot_diff) >= abs(observed_diff):
            count += 1

    return count / n_bootstrap


# ==============================================================================
# Logging utilities
# ==============================================================================

def setup_logging(
    log_file: Optional[str] = None,
    level: int = logging.INFO
) -> None:
    """
    Setup logging configuration.

    Args:
        log_file: Optional path to log file
        level: Logging level
    """
    handlers = [logging.StreamHandler()]

    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


class MetricsLogger:
    """Logger for training metrics."""

    def __init__(self, log_dir: str):
        """
        Initialize metrics logger.

        Args:
            log_dir: Directory to save logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_file = self.log_dir / 'metrics.jsonl'
        self.history = []

    def log(self, step: int, metrics: Dict[str, float]) -> None:
        """
        Log metrics for a step.

        Args:
            step: Training step
            metrics: Dict of metric values
        """
        entry = {'step': step, **metrics}
        self.history.append(entry)

        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')

    def get_history(self) -> List[Dict[str, Any]]:
        """Get full metrics history."""
        return self.history

    def get_metric_series(self, metric_name: str) -> Tuple[List[int], List[float]]:
        """
        Get time series for a specific metric.

        Args:
            metric_name: Name of the metric

        Returns:
            Tuple of (steps, values)
        """
        steps = []
        values = []

        for entry in self.history:
            if metric_name in entry:
                steps.append(entry['step'])
                values.append(entry[metric_name])

        return steps, values


# ==============================================================================
# Tensor utilities
# ==============================================================================

def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int = -1
) -> torch.Tensor:
    """
    Compute mean over masked positions.

    Args:
        tensor: Input tensor
        mask: Boolean or float mask
        dim: Dimension to average over

    Returns:
        Masked mean tensor
    """
    mask = mask.float()
    return (tensor * mask).sum(dim=dim) / mask.sum(dim=dim).clamp(min=1e-10)


def get_batch_logps(
    logits: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    average_log_prob: bool = False
) -> torch.Tensor:
    """
    Compute log probabilities for a batch.

    Args:
        logits: Model logits [batch, seq_len, vocab_size]
        labels: Target labels [batch, seq_len]
        attention_mask: Attention mask [batch, seq_len]
        average_log_prob: Whether to average over sequence

    Returns:
        Log probabilities [batch]
    """
    # Shift for next-token prediction
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_mask = attention_mask[..., 1:].contiguous()
    loss_mask = (shift_labels != -100) & (shift_mask == 1)
    safe_labels = shift_labels.masked_fill(~loss_mask, 0)

    # Compute per-token log probs
    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    per_token_logps = torch.gather(
        log_probs,
        dim=-1,
        index=safe_labels.unsqueeze(-1)
    ).squeeze(-1)

    # Apply mask
    masked_logps = per_token_logps * loss_mask

    if average_log_prob:
        return masked_logps.sum(-1) / loss_mask.sum(-1).clamp(min=1)
    else:
        return masked_logps.sum(-1)


# ==============================================================================
# EMA utilities
# ==============================================================================

class EMAModel:
    """
    Exponential Moving Average model wrapper.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        decay: float = 0.995
    ):
        """
        Initialize EMA model.

        Args:
            model: Base model
            decay: EMA decay rate ρ
        """
        self.model = model
        self.decay = decay
        self.shadow = {}

        # Initialize shadow params
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self) -> None:
        """Update shadow parameters with EMA."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1 - self.decay
                )

    def apply_shadow(self) -> None:
        """Apply shadow parameters to model."""
        for name, param in self.model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name])

    def restore(self, backup: Dict[str, torch.Tensor]) -> None:
        """Restore model from backup."""
        for name, param in self.model.named_parameters():
            if name in backup:
                param.data.copy_(backup[name])

    def get_backup(self) -> Dict[str, torch.Tensor]:
        """Get backup of current model parameters."""
        return {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
        }


# ==============================================================================
# Configuration utilities
# ==============================================================================

def save_config(config: Any, path: str) -> None:
    """
    Save configuration to JSON file.

    Args:
        config: Configuration object (dataclass or dict)
        path: Output file path
    """
    if hasattr(config, '__dict__'):
        config_dict = config.__dict__
    else:
        config_dict = dict(config)

    with open(path, 'w') as f:
        json.dump(config_dict, f, indent=2, default=str)


def load_config(path: str, config_class=None) -> Any:
    """
    Load configuration from JSON file.

    Args:
        path: Input file path
        config_class: Optional dataclass to instantiate

    Returns:
        Configuration object or dict
    """
    with open(path, 'r') as f:
        config_dict = json.load(f)

    if config_class is not None:
        return config_class(**config_dict)

    return config_dict
