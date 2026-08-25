"""
Integration tests for TURDPOTrainer with PyTorch causal language models.
"""

import pytest
import torch
import torch.nn as nn
from typing import Optional, NamedTuple

from turdpo.trainer import TURDPOTrainer, TURDPOConfig
from turdpo.data import (
    ListwisePreferenceDataset,
    PreferencePair,
    PreferenceDataset,
    collate_listwise_batch,
    create_dataloader,
)


class ModelOutput(NamedTuple):
    logits: torch.Tensor


class SimpleCausalLM(nn.Module):
    """
    A PyTorch causal language model adhering to the HuggingFace AutoModelForCausalLM interface.
    """
    def __init__(self, vocab_size: int = 1000, hidden_dim: int = 64):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.transformer_block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> ModelOutput:
        hidden = self.embedding(input_ids)
        hidden = self.transformer_block(hidden)
        logits = self.lm_head(hidden)
        return ModelOutput(logits=logits)


class DummyTokenizer:
    """Lightweight dummy tokenizer for fast unit testing."""
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.pad_token_id = 0
        self.eos_token_id = 1

    def __call__(self, text, max_length=64, padding="max_length", truncation=True, return_tensors="pt"):
        words = text.split()
        ids = [(abs(hash(w)) % (self.vocab_size - 2)) + 2 for w in words]
        if len(ids) > max_length:
            ids = ids[:max_length]
        else:
            ids = ids + [self.pad_token_id] * (max_length - len(ids))

        mask = [1 if i != self.pad_token_id else 0 for i in ids]

        return {
            "input_ids": torch.tensor([ids], dtype=torch.long),
            "attention_mask": torch.tensor([mask], dtype=torch.long)
        }


class TestTURDPOTrainerIntegration:
    """Integration test suite for TURDPOTrainer."""

    def test_train_step_with_causal_lm(self):
        """Verify complete forward and backward training step with policy & reference models."""
        device = "cpu"
        tokenizer = DummyTokenizer()

        # Instantiate policy and reference models
        model = SimpleCausalLM(vocab_size=1000, hidden_dim=32)
        reference_model = SimpleCausalLM(vocab_size=1000, hidden_dim=32)

        # Sample dataset
        pairs = [
            PreferencePair(
                prompt="Explain the reasoning behind 10 + 20.",
                chosen="First add the tens digits 1 + 2 = 3. Multiply by 10 to get 30.",
                rejected="It is 50 because everyone agrees it is 50."
            ),
            PreferencePair(
                prompt="Is the earth round or flat?",
                chosen="The earth is spherical due to gravity pulling mass inward symmetrically.",
                rejected="The earth is flat because my peers asserted it."
            )
        ]

        dataset = PreferenceDataset(pairs, tokenizer, max_length=32, max_prompt_length=16)
        dataloader = create_dataloader(dataset, batch_size=2, shuffle=False)

        config = TURDPOConfig(
            beta=2.0,
            gamma=1.0,
            a=0.6,
            lambda_uncertainty=0.5,
            k_samples=2,
            learning_rate=1e-4,
            use_ema_reference=True,
            ema_decay=0.99
        )

        trainer = TURDPOTrainer(
            model=model,
            reference_model=reference_model,
            tokenizer=tokenizer,
            config=config,
            device=device
        )

        # Run 1 epoch
        results = trainer.train(dataloader, num_epochs=1)

        assert "final_loss" in results
        assert results["total_steps"] == 1
        assert not torch.isnan(torch.tensor(results["final_loss"]))

    def test_true_listwise_train_step(self):
        """Verify that use_listwise consumes K candidates rather than a fake pair."""
        tokenizer = DummyTokenizer()
        model = SimpleCausalLM(vocab_size=1000, hidden_dim=32)
        reference_model = SimpleCausalLM(vocab_size=1000, hidden_dim=32)
        dataset = ListwisePreferenceDataset(
            [{
                "id": "list-1",
                "prompt": "What is 2 + 2?",
                "candidates": [
                    {"text": "4", "reward": 1.0},
                    {"text": "5", "reward": 0.0},
                    {"text": "6", "reward": -0.2},
                ],
                "preferences": [1, 0, 0],
            }],
            tokenizer,
            num_candidates=3,
            max_length=16,
        )
        dataloader = create_dataloader(
            dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_listwise_batch,
        )
        trainer = TURDPOTrainer(
            model=model,
            reference_model=reference_model,
            tokenizer=tokenizer,
            config=TURDPOConfig(
                use_listwise=True,
                num_candidates=3,
                learning_rate=1e-4,
                max_steps=1,
            ),
            device="cpu",
        )
        results = trainer.train(dataloader, num_epochs=1)
        assert results["total_steps"] == 1
        assert not torch.isnan(torch.tensor(results["final_loss"]))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
