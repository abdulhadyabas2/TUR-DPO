"""
Trainer Module for TUR-DPO

This module implements the complete training pipeline for TUR-DPO.
Based on the training protocol described in Section 2 of the paper:

1. Elicit graphs for positive and negative candidates
2. Compute semantic and topology scores
3. Compute epistemic and aleatoric uncertainties
4. Map to pair weight
5. Load or fit calibrator parameters from an explicit held-out set (optional)
6. Update policy parameters
7. Optionally update reference by EMA
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
import numpy as np
import random
from tqdm import tqdm
import logging
from contextlib import nullcontext

from .topology import TopologyExtractor, TopologyScorer, TopologyGraph
from .uncertainty import UncertaintyEstimator, PairWeightComputer
from .rewards import ShapedReward, RewardDifferenceComputer
from .loss import TURDPOLoss, ListwiseTURDPOLoss
from .verifier import NodeVerifier, ContradictionDetector

logger = logging.getLogger(__name__)


@dataclass
class TURDPOConfig:
    """Configuration for TUR-DPO training."""

    # Temperature and reward parameters
    beta: float = 2.0  # DPO temperature
    gamma: float = 1.0  # Reward difference weight

    # Shaped reward parameters
    a: float = 0.6  # Semantic vs topology mixing
    lambda_uncertainty: float = 0.5  # Uncertainty penalty in reward
    lambda_peer_pressure: float = 0.0

    # Topology scoring weights (Equation 1)
    alpha_path: float = 1.0
    alpha_cycle: float = 0.5
    alpha_dangling: float = 0.3
    alpha_contradict: float = 0.4
    alpha_peer_unverified: float = 0.0

    # Uncertainty estimation
    lambda_epi: float = 0.5  # Epistemic weight
    lambda_ale: float = 0.5  # Aleatoric weight
    tau_smoothing: float = 0.05  # Smoothing prior for aleatoric

    # Pair weighting
    tau_w: float = 1.2  # Weight mapping temperature
    w_min: float = 0.05  # Minimum weight floor

    # Graph re-elicitation
    k_samples: int = 3  # Number of re-elicited graphs

    # Reference policy
    use_ema_reference: bool = True
    ema_decay: float = 0.995  # EMA decay ρ

    # Training
    learning_rate: float = 1e-6
    weight_decay: float = 0.1
    warmup_steps: int = 2000
    max_steps: int = 100000
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

    # Evaluation
    eval_steps: int = 500
    save_steps: int = 2000
    logging_steps: int = 100

    # Calibrator fitting is explicit/offline; the trainer does not invent labels.
    train_calibrators: bool = False
    calibrator_lr: float = 1e-4

    # Listwise training
    use_listwise: bool = False
    num_candidates: int = 4

    # Data and execution
    use_precomputed_annotations: bool = True
    num_workers: int = 0
    use_shaped_reward: bool = True
    use_pair_weights: bool = True
    use_amp: bool = False
    amp_dtype: str = "bf16"

    def __post_init__(self) -> None:
        if self.beta <= 0 or self.gamma < 0:
            raise ValueError("beta must be > 0 and gamma must be >= 0")
        if not 0.0 <= self.a <= 1.0:
            raise ValueError("a must be in [0, 1]")
        if self.k_samples < 1:
            raise ValueError("k_samples must be at least 1")
        if not 0.0 <= self.w_min <= 1.0:
            raise ValueError("w_min must be in [0, 1]")
        if self.gradient_accumulation_steps < 1 or self.max_steps < 1:
            raise ValueError("gradient_accumulation_steps and max_steps must be positive")


@dataclass
class TrainingState:
    """Training state for checkpointing."""
    step: int = 0
    epoch: int = 0
    best_metric: float = 0.0
    metrics_history: List[Dict[str, float]] = field(default_factory=list)


class TURDPOTrainer:
    """
    Complete TUR-DPO training pipeline.

    Implements the training protocol from the paper:
    1. Graph elicitation with perturbations
    2. Score computation (semantic, topology)
    3. Uncertainty estimation and pair weighting
    4. Loss computation with shaped reward augmentation
    5. Optional EMA reference update
    """

    def __init__(
        self,
        model: nn.Module,
        reference_model: nn.Module,
        tokenizer,
        config: TURDPOConfig,
        topology_extractor: Optional[TopologyExtractor] = None,
        verifier: Optional[NodeVerifier] = None,
        contradiction_detector: Optional[ContradictionDetector] = None,
        device: str = "cuda"
    ):
        """
        Initialize TUR-DPO trainer.

        Args:
            model: Policy model to train
            reference_model: Reference policy (frozen or EMA-updated)
            tokenizer: Tokenizer for the model
            config: Training configuration
            topology_extractor: Extractor for reasoning graphs
            verifier: Verifier for node correctness
            device: Device to use for training
        """
        self.model = model
        self.reference_model = reference_model
        self.tokenizer = tokenizer
        self.config = config
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but unavailable; falling back to CPU")
            device = "cpu"
        self.device = torch.device(device)
        self.use_amp = bool(config.use_amp and self.device.type == "cuda")
        self.amp_dtype = str(config.amp_dtype).lower()
        if self.amp_dtype not in {"bf16", "float16", "fp16"}:
            raise ValueError("amp_dtype must be 'bf16' or 'float16'")
        scaler_enabled = self.use_amp and self.amp_dtype in {"float16", "fp16"}
        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
        except (AttributeError, TypeError):
            self.scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)

        # Initialize components
        self.topology_extractor = topology_extractor or TopologyExtractor()
        self.topology_scorer = TopologyScorer(
            alpha_path=config.alpha_path,
            alpha_cycle=config.alpha_cycle,
            alpha_dangling=config.alpha_dangling,
            alpha_contradict=config.alpha_contradict,
            alpha_peer_unverified=config.alpha_peer_unverified,
        )

        self.uncertainty_estimator = UncertaintyEstimator(
            lambda_epi=config.lambda_epi,
            lambda_ale=config.lambda_ale,
            tau=config.tau_smoothing,
            scorer=self.topology_scorer
        )

        self.pair_weight_computer = PairWeightComputer(
            tau_w=config.tau_w,
            w_min=config.w_min,
            uncertainty_estimator=self.uncertainty_estimator
        )

        self.shaped_reward = ShapedReward(
            a=config.a,
            lambda_uncertainty=config.lambda_uncertainty,
            lambda_peer_pressure=config.lambda_peer_pressure,
        )

        self.reward_computer = RewardDifferenceComputer(
            shaped_reward=self.shaped_reward,
            gamma=config.gamma
        )

        self.verifier = verifier or NodeVerifier()
        self.contradiction_detector = contradiction_detector or ContradictionDetector()

        # Loss functions
        if config.use_listwise:
            self.loss_fn = ListwiseTURDPOLoss(
                beta=config.beta,
                gamma=config.gamma,
                num_candidates=config.num_candidates
            )
        else:
            self.loss_fn = TURDPOLoss(
                beta=config.beta,
                gamma=config.gamma
            )

        # Move models to device
        self.model.to(device)
        self.reference_model.to(device)
        self.reference_model.eval()
        for parameter in self.reference_model.parameters():
            parameter.requires_grad_(False)

        # Training state
        self.state = TrainingState()

        # Setup optimizer
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()

    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup AdamW optimizer."""
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999)
        )

    def _setup_scheduler(self):
        """Setup linear warmup followed by cosine decay."""
        from torch.optim.lr_scheduler import LambdaLR
        import math

        def lr_lambda(step):
            if step < self.config.warmup_steps:
                return step / max(1, self.config.warmup_steps)
            progress = (step - self.config.warmup_steps) / max(
                1, self.config.max_steps - self.config.warmup_steps
            )
            progress = min(max(progress, 0.0), 1.0)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return LambdaLR(self.optimizer, lr_lambda)

    def _autocast_context(self):
        """Return a CUDA autocast context when mixed precision is enabled."""
        if not self.use_amp:
            return nullcontext()
        dtype = torch.float16 if self.amp_dtype in {"float16", "fp16"} else torch.bfloat16
        return torch.autocast(device_type="cuda", dtype=dtype)

    def _backward_and_step(self, loss: torch.Tensor, optimizer_step: bool) -> None:
        """Backpropagate with optional GradScaler and perform an optimizer step."""
        scaled_loss = loss / max(1, self.config.gradient_accumulation_steps)
        if self.scaler.is_enabled():
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        if not optimizer_step:
            return
        if self.config.max_grad_norm > 0:
            if self.scaler.is_enabled():
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        if self.scaler.is_enabled():
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.state.step += 1

    def train(
        self,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        num_epochs: int = 1
    ) -> Dict[str, Any]:
        """
        Run TUR-DPO training.

        Args:
            train_dataloader: DataLoader for training data
            eval_dataloader: Optional DataLoader for evaluation
            num_epochs: Number of training epochs

        Returns:
            Training results and metrics
        """
        logger.info("Starting TUR-DPO training...")
        logger.info(f"Config: beta={self.config.beta}, gamma={self.config.gamma}, "
                   f"a={self.config.a}, lambda_u={self.config.lambda_uncertainty}")

        self.model.train()

        epoch_metrics = {"loss": 0.0}
        for epoch in range(num_epochs):
            self.state.epoch = epoch
            epoch_metrics = self._train_epoch(train_dataloader)

            logger.info(f"Epoch {epoch + 1}/{num_epochs} completed. "
                       f"Loss: {epoch_metrics['loss']:.4f}")

            # Evaluation
            if eval_dataloader is not None:
                eval_metrics = self.evaluate(eval_dataloader)
                logger.info(f"Eval metrics: {eval_metrics}")

                # Track best model
                if eval_metrics.get('accuracy', 0) > self.state.best_metric:
                    self.state.best_metric = eval_metrics['accuracy']

            if self.state.step >= self.config.max_steps:
                logger.info("Reached max_steps=%s", self.config.max_steps)
                break

        return {
            "final_loss": epoch_metrics['loss'],
            "best_metric": self.state.best_metric,
            "total_steps": self.state.step,
            "metrics_history": self.state.metrics_history
        }

    def _train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        total_loss = 0.0
        num_batches = 0
        self.optimizer.zero_grad(set_to_none=True)
        accumulation = max(1, self.config.gradient_accumulation_steps)
        total_batches = len(dataloader) if hasattr(dataloader, "__len__") else None
        progress_bar = tqdm(dataloader, desc=f"Epoch {self.state.epoch + 1}")

        for batch_idx, batch in enumerate(progress_bar):
            if self.state.step >= self.config.max_steps:
                break
            # Training step
            is_last_batch = total_batches is not None and batch_idx + 1 == total_batches
            should_update = ((batch_idx + 1) % accumulation == 0) or is_last_batch
            metrics = self._train_step(batch, optimizer_step=should_update)

            total_loss += metrics['loss']
            num_batches += 1

            # Update progress bar
            progress_bar.set_postfix({
                'loss': metrics['loss'],
                'weight': metrics.get('weights', 1.0),
                'reward_diff': metrics.get('reward_diff', 0.0)
            })

            # Logging
            if self.state.step and self.state.step % self.config.logging_steps == 0:
                self.state.metrics_history.append(metrics)

            # EMA reference update
            if should_update and self.config.use_ema_reference:
                self._update_ema_reference()

        return {'loss': total_loss / max(num_batches, 1)}

    def _train_step(self, batch: Dict[str, Any], optimizer_step: bool = True) -> Dict[str, float]:
        """
        Single training step implementing the TUR-DPO protocol.

        Protocol:
        1. Elicit graphs for positive and negative candidates
        2. Compute semantic and topology scores
        3. Compute uncertainties and pair weight
        4. Compute loss with shaped reward
        5. Update model
        """
        if self.config.use_listwise and "input_ids" in batch:
            return self._train_listwise_step(batch, optimizer_step=optimizer_step)

        # Move batch to device
        batch = self._prepare_batch(batch)

        if not self.config.use_shaped_reward and not self.config.use_pair_weights:
            # Fast canonical-DPO control: no annotation calls or auxiliary scores.
            batch_size = len(batch.get("prompts", []))
            graphs_pos, graphs_neg = [], []
            reward_diffs = torch.zeros(batch_size, device=self.device)
            weights = torch.ones(batch_size, device=self.device)
        else:
            # Step 1: Elicit graphs
            graphs_pos, graphs_neg = self._elicit_graphs(batch)

            # Step 2: Compute scores
            topo_scores_pos, topo_scores_neg = self._compute_topology_scores(
                graphs_pos, graphs_neg, batch
            )
            sem_scores_pos, sem_scores_neg = self._compute_semantic_scores(
                batch, graphs_pos, graphs_neg
            )

            # Step 3: Compute uncertainties and weights
            weights, uncertainties = self._compute_weights(
                graphs_pos, graphs_neg, batch
            )

            # Step 4: Compute reward differences
            reward_diffs = self._compute_reward_differences(
                sem_scores_pos, sem_scores_neg,
                topo_scores_pos, topo_scores_neg,
                uncertainties,
                graphs_pos=graphs_pos,
                graphs_neg=graphs_neg,
                batch=batch,
            )

        # Step 5: Forward pass and loss computation
        with self._autocast_context():
            loss, metrics = self._compute_loss(batch, reward_diffs, weights)

        # Step 6: Backward pass
        self._backward_and_step(loss, optimizer_step)

        return metrics

    def _train_listwise_step(
        self,
        batch: Dict[str, Any],
        optimizer_step: bool = True,
    ) -> Dict[str, float]:
        """Train one true ``[batch, candidates]`` listwise batch."""
        batch = self._prepare_batch(batch)
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch.get("labels")
        if input_ids.dim() != 3:
            raise ValueError("Listwise input_ids must have shape [batch, candidates, tokens]")
        batch_size, num_candidates, sequence_length = input_ids.shape
        flat_ids = input_ids.reshape(batch_size * num_candidates, sequence_length)
        flat_mask = attention_mask.reshape(batch_size * num_candidates, sequence_length)
        flat_labels = None if labels is None else labels.reshape(
            batch_size * num_candidates, sequence_length
        )

        with torch.no_grad(), self._autocast_context():
            reference_logps = self._compute_logps(
                self.reference_model, flat_ids, flat_mask, flat_labels
            ).reshape(batch_size, num_candidates)
        with self._autocast_context():
            policy_logps = self._compute_logps(
                self.model, flat_ids, flat_mask, flat_labels
            ).reshape(batch_size, num_candidates)

        rewards = batch.get("rewards")
        if rewards is None:
            rewards = torch.zeros_like(policy_logps)
        rewards = rewards.to(device=self.device, dtype=policy_logps.dtype)
        preferences = batch.get("preferences")
        if preferences is not None:
            preferences = preferences.to(device=self.device, dtype=policy_logps.dtype)
        weights = batch.get("weights")
        if weights is None:
            weights = torch.ones(batch_size, device=self.device, dtype=policy_logps.dtype)

        with self._autocast_context():
            loss, metrics = self.loss_fn(
                policy_logps=policy_logps,
                reference_logps=reference_logps,
                rewards=rewards,
                preferences=preferences,
                weight=weights,
            )
        self._backward_and_step(loss, optimizer_step)

        return {
            key: value.item() if isinstance(value, torch.Tensor) else float(value)
            for key, value in metrics.items()
        }

    def _prepare_batch(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Move batch tensors to device."""
        prepared = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                prepared[key] = value.to(self.device)
            else:
                prepared[key] = value
        return prepared

    def _elicit_graphs(
        self,
        batch: Dict[str, Any]
    ) -> Tuple[List[List[TopologyGraph]], List[List[TopologyGraph]]]:
        """
        Elicit topology graphs for positive and negative responses.

        Returns K re-elicited graphs per response for uncertainty estimation.
        """
        prompts = batch.get('prompts', [])
        chosen_responses = batch.get('chosen_responses', [])
        rejected_responses = batch.get('rejected_responses', [])

        graphs_pos = []
        graphs_neg = []
        metadata = self._metadata(batch)

        for index, (prompt, chosen, rejected) in enumerate(zip(prompts, chosen_responses, rejected_responses)):
            item_metadata = metadata[index] if index < len(metadata) else {}
            required = {
                "topo_score_chosen", "topo_score_rejected", "semantic_score_chosen",
                "semantic_score_rejected", "uncertainty_chosen", "uncertainty_rejected",
                "pair_weight",
            }
            if self.config.use_precomputed_annotations and required.issubset(item_metadata):
                # Annotation caches make graph elicitation an offline preprocessing
                # step rather than a repeated training-time model call.
                graphs_pos.append([])
                graphs_neg.append([])
                continue
            # Extract K graphs for each response
            pos_graphs = self.topology_extractor.extract_multiple(
                prompt=prompt,
                response=chosen,
                k=self.config.k_samples
            )
            neg_graphs = self.topology_extractor.extract_multiple(
                prompt=prompt,
                response=rejected,
                k=self.config.k_samples
            )

            # Populate node correctness probabilities before uncertainty and
            # semantic scoring.  A model-backed verifier can be injected; the
            # default verifier is deterministic and explicitly documented as a
            # fallback.
            for graph in pos_graphs + neg_graphs:
                self.verifier.verify_graph_nodes(graph, prompt)

            graphs_pos.append(pos_graphs)
            graphs_neg.append(neg_graphs)

        return graphs_pos, graphs_neg

    def _compute_topology_scores(
        self,
        graphs_pos: List[List[TopologyGraph]],
        graphs_neg: List[List[TopologyGraph]],
        batch: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute topology scores for all graphs."""
        scores_pos = []
        scores_neg = []

        metadata = self._metadata(batch)
        for index, (pos_graphs, neg_graphs) in enumerate(zip(graphs_pos, graphs_neg)):
            item_metadata = metadata[index] if index < len(metadata) else {}
            if self.config.use_precomputed_annotations and "topo_score_chosen" in item_metadata:
                scores_pos.append(float(item_metadata["topo_score_chosen"]))
            else:
                scores_pos.append(self._mean_topology_score(pos_graphs))
            if self.config.use_precomputed_annotations and "topo_score_rejected" in item_metadata:
                scores_neg.append(float(item_metadata["topo_score_rejected"]))
            else:
                scores_neg.append(self._mean_topology_score(neg_graphs))

        return (
            torch.tensor(scores_pos, device=self.device),
            torch.tensor(scores_neg, device=self.device)
        )

    def _mean_topology_score(self, graphs: List[TopologyGraph]) -> float:
        """Average scored re-elicitation samples, including contradiction checks."""
        if not graphs:
            return 0.5
        values = []
        for graph in graphs:
            contradiction, _ = self.contradiction_detector.detect_contradictions(graph)
            values.append(self.topology_scorer.compute_score(graph, contradiction))
        return float(np.mean(values))

    def _metadata(self, batch: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Return normalized per-example annotations from a collated batch."""
        if batch is None:
            return []
        values = batch.get("metadata", [])
        return [value if isinstance(value, dict) else {} for value in values]

    def _compute_semantic_scores(
        self,
        batch: Dict[str, Any],
        graphs_pos: List[List[TopologyGraph]],
        graphs_neg: List[List[TopologyGraph]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute semantic scores for responses."""
        metadata = self._metadata(batch)
        task_scores_pos = batch.get('task_scores_chosen', None)
        task_scores_neg = batch.get('task_scores_rejected', None)
        scores_pos = []
        scores_neg = []

        for i, (pos_graphs, neg_graphs) in enumerate(zip(graphs_pos, graphs_neg)):
            item_metadata = metadata[i] if i < len(metadata) else {}
            if self.config.use_precomputed_annotations and "semantic_score_chosen" in item_metadata:
                pos_score = float(item_metadata["semantic_score_chosen"])
            else:
                task_pos = None if task_scores_pos is None else float(task_scores_pos[i].item())
                pos_score = self.shaped_reward.semantic_scorer.compute(
                    batch["prompts"][i], batch["chosen_responses"][i],
                    graph=pos_graphs[0] if pos_graphs else None,
                    task_score=task_pos,
                    hallucination_score=self._mean_contradiction(pos_graphs),
                )
            if self.config.use_precomputed_annotations and "semantic_score_rejected" in item_metadata:
                neg_score = float(item_metadata["semantic_score_rejected"])
            else:
                task_neg = None if task_scores_neg is None else float(task_scores_neg[i].item())
                neg_score = self.shaped_reward.semantic_scorer.compute(
                    batch["prompts"][i], batch["rejected_responses"][i],
                    graph=neg_graphs[0] if neg_graphs else None,
                    task_score=task_neg,
                    hallucination_score=self._mean_contradiction(neg_graphs),
                )
            scores_pos.append(pos_score)
            scores_neg.append(neg_score)

        return (
            torch.tensor(scores_pos, device=self.device),
            torch.tensor(scores_neg, device=self.device)
        )

    def _mean_contradiction(self, graphs: List[TopologyGraph]) -> float:
        if not graphs:
            return 0.0
        return float(np.mean([
            self.contradiction_detector.detect_contradictions(graph)[0]
            for graph in graphs
        ]))

    def _compute_weights(
        self,
        graphs_pos: List[List[TopologyGraph]],
        graphs_neg: List[List[TopologyGraph]],
        batch: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute pair weights from uncertainties."""
        if not self.config.use_pair_weights:
            zeros = torch.zeros(len(graphs_pos), device=self.device)
            return torch.ones(len(graphs_pos), device=self.device), {"u_pos": zeros, "u_neg": zeros}
        weights = []
        u_pos_list = []
        u_neg_list = []

        metadata = self._metadata(batch)
        for index, (pos_graphs, neg_graphs) in enumerate(zip(graphs_pos, graphs_neg)):
            item_metadata = metadata[index] if index < len(metadata) else {}
            if self.config.use_precomputed_annotations and "pair_weight" in item_metadata:
                weight = float(item_metadata["pair_weight"])
                weight = float(np.clip(weight, self.config.w_min, 1.0))
                weights.append(weight)
                u_pos_list.append(float(item_metadata.get("uncertainty_chosen", 0.0)))
                u_neg_list.append(float(item_metadata.get("uncertainty_rejected", 0.0)))
                continue
            weight, u_dict = self.pair_weight_computer.compute_weight(
                graphs_pos=pos_graphs,
                graphs_neg=neg_graphs
            )
            weights.append(weight)
            u_pos_list.append(u_dict['u_pos_total'])
            u_neg_list.append(u_dict['u_neg_total'])

        return (
            torch.tensor(weights, device=self.device),
            {
                'u_pos': torch.tensor(u_pos_list, device=self.device),
                'u_neg': torch.tensor(u_neg_list, device=self.device)
            }
        )

    def _compute_reward_differences(
        self,
        sem_pos: torch.Tensor,
        sem_neg: torch.Tensor,
        topo_pos: torch.Tensor,
        topo_neg: torch.Tensor,
        uncertainties: Dict[str, torch.Tensor],
        graphs_pos: Optional[List[List[TopologyGraph]]] = None,
        graphs_neg: Optional[List[List[TopologyGraph]]] = None,
        batch: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Compute shaped reward differences."""
        if not self.config.use_shaped_reward:
            return torch.zeros_like(sem_pos)
        batch_size = sem_pos.shape[0]
        reward_diffs = []
        metadata = self._metadata(batch)

        for i in range(batch_size):
            peer_pos = 0.0
            peer_neg = 0.0
            if graphs_pos is not None and i < len(graphs_pos):
                peer_pos = self._mean_peer_pressure(graphs_pos[i])
            if graphs_neg is not None and i < len(graphs_neg):
                peer_neg = self._mean_peer_pressure(graphs_neg[i])
            if i < len(metadata):
                peer_pos = float(metadata[i].get("peer_pressure_rate_chosen", peer_pos))
                peer_neg = float(metadata[i].get("peer_pressure_rate_rejected", peer_neg))

            delta_r, _ = self.reward_computer.compute(
                sem_score_pos=sem_pos[i].item(),
                sem_score_neg=sem_neg[i].item(),
                topo_score_pos=topo_pos[i].item(),
                topo_score_neg=topo_neg[i].item(),
                uncertainty_pos=uncertainties['u_pos'][i].item(),
                uncertainty_neg=uncertainties['u_neg'][i].item(),
                peer_pressure_pos=peer_pos,
                peer_pressure_neg=peer_neg,
            )
            reward_diffs.append(delta_r)

        return torch.tensor(reward_diffs, device=self.device)

    def _mean_peer_pressure(self, graphs: List[TopologyGraph]) -> float:
        if not graphs:
            return 0.0
        return float(np.mean([graph.peer_pressure_rate() for graph in graphs]))

    def _compute_loss(
        self,
        batch: Dict[str, Any],
        reward_diffs: torch.Tensor,
        weights: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute TUR-DPO loss."""
        # Get model outputs
        with torch.no_grad():
            ref_chosen_logps = self._compute_logps(
                self.reference_model,
                batch['chosen_input_ids'],
                batch['chosen_attention_mask'],
                batch.get('chosen_labels')
            )
            ref_rejected_logps = self._compute_logps(
                self.reference_model,
                batch['rejected_input_ids'],
                batch['rejected_attention_mask'],
                batch.get('rejected_labels')
            )

        policy_chosen_logps = self._compute_logps(
            self.model,
            batch['chosen_input_ids'],
            batch['chosen_attention_mask'],
            batch.get('chosen_labels')
        )
        policy_rejected_logps = self._compute_logps(
            self.model,
            batch['rejected_input_ids'],
            batch['rejected_attention_mask'],
            batch.get('rejected_labels')
        )

        if self.config.use_listwise:
            # A pairwise dataloader is represented as a two-candidate list.  A
            # ListwisePreferenceDataset can provide larger candidate lists through
            # its own adapter; this branch keeps the pairwise trainer functional
            # and preserves the preferred-first ordering.
            policy_logps = torch.stack([policy_chosen_logps, policy_rejected_logps], dim=-1)
            reference_logps = torch.stack([ref_chosen_logps, ref_rejected_logps], dim=-1)
            rewards = torch.stack([reward_diffs / 2.0, -reward_diffs / 2.0], dim=-1)
            preferences = torch.zeros_like(rewards)
            preferences[:, 0] = 1.0
            loss, metrics = self.loss_fn(
                policy_logps=policy_logps,
                reference_logps=reference_logps,
                rewards=rewards,
                preferences=preferences,
                weight=weights,
            )
        else:
            loss, metrics = self.loss_fn(
                policy_chosen_logps=policy_chosen_logps,
                policy_rejected_logps=policy_rejected_logps,
                reference_chosen_logps=ref_chosen_logps,
                reference_rejected_logps=ref_rejected_logps,
                reward_diff=reward_diffs,
                weights=weights
            )

        # Convert metrics to Python floats
        metrics = {k: v.item() if isinstance(v, torch.Tensor) else v
                  for k, v in metrics.items()}

        return loss, metrics

    def _compute_logps(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute log probabilities for sequences."""
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]

        if labels is None:
            labels = input_ids

        return self.loss_fn.compute_logps(logits, labels, attention_mask)

    def _update_ema_reference(self) -> None:
        """Update reference model with exponential moving average."""
        decay = self.config.ema_decay

        with torch.no_grad():
            for param, ref_param in zip(
                self.model.parameters(),
                self.reference_model.parameters()
            ):
                ref_param.data.mul_(decay).add_(param.data, alpha=1 - decay)

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate model on validation data."""
        self.model.eval()

        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                batch = self._prepare_batch(batch)

                if self.config.use_listwise and "input_ids" in batch:
                    input_ids = batch["input_ids"]
                    attention_mask = batch["attention_mask"]
                    labels = batch.get("labels")
                    batch_size, num_candidates, sequence_length = input_ids.shape
                    flat_ids = input_ids.reshape(batch_size * num_candidates, sequence_length)
                    flat_mask = attention_mask.reshape(batch_size * num_candidates, sequence_length)
                    flat_labels = None if labels is None else labels.reshape(
                        batch_size * num_candidates, sequence_length
                    )
                    ref_logps = self._compute_logps(
                        self.reference_model, flat_ids, flat_mask, flat_labels
                    ).reshape(batch_size, num_candidates)
                    policy_logps = self._compute_logps(
                        self.model, flat_ids, flat_mask, flat_labels
                    ).reshape(batch_size, num_candidates)
                    utilities = self.config.beta * (policy_logps - ref_logps)
                    predictions = utilities.argmax(dim=-1)
                    targets = batch["preferences"].argmax(dim=-1)
                    total_accuracy += (predictions == targets).float().mean().item()
                    num_batches += 1
                    continue

                # Simplified evaluation without full graph elicitation
                ref_chosen_logps = self._compute_logps(
                    self.reference_model,
                    batch['chosen_input_ids'],
                    batch['chosen_attention_mask'],
                    batch.get('chosen_labels')
                )
                ref_rejected_logps = self._compute_logps(
                    self.reference_model,
                    batch['rejected_input_ids'],
                    batch['rejected_attention_mask'],
                    batch.get('rejected_labels')
                )

                policy_chosen_logps = self._compute_logps(
                    self.model,
                    batch['chosen_input_ids'],
                    batch['chosen_attention_mask'],
                    batch.get('chosen_labels')
                )
                policy_rejected_logps = self._compute_logps(
                    self.model,
                    batch['rejected_input_ids'],
                    batch['rejected_attention_mask'],
                    batch.get('rejected_labels')
                )

                # Compute accuracy
                chosen_rewards = self.config.beta * (policy_chosen_logps - ref_chosen_logps)
                rejected_rewards = self.config.beta * (policy_rejected_logps - ref_rejected_logps)
                accuracy = (chosen_rewards > rejected_rewards).float().mean()

                total_accuracy += accuracy.item()
                num_batches += 1

        self.model.train()

        return {
            'accuracy': total_accuracy / max(num_batches, 1),
            'num_batches': num_batches
        }

    def save_checkpoint(self, path: str) -> None:
        """Save training checkpoint."""
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'reference_state_dict': self.reference_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'state': self.state,
            'config': self.config,
            'random_state': random.getstate(),
            'numpy_random_state': np.random.get_state(),
            'torch_random_state': torch.get_rng_state(),
            'cuda_random_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }, path)
        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint."""
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.reference_model.load_state_dict(checkpoint['reference_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.state = checkpoint['state']
        if checkpoint.get('random_state') is not None:
            random.setstate(checkpoint['random_state'])
        if checkpoint.get('numpy_random_state') is not None:
            np.random.set_state(checkpoint['numpy_random_state'])
        if checkpoint.get('torch_random_state') is not None:
            torch.set_rng_state(checkpoint['torch_random_state'])
        if torch.cuda.is_available() and checkpoint.get('cuda_random_state') is not None:
            torch.cuda.set_rng_state_all(checkpoint['cuda_random_state'])

        logger.info(f"Checkpoint loaded from {path}")
