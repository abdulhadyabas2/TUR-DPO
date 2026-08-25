"""
Data Module for TUR-DPO

This module implements data loading and preprocessing for TUR-DPO training.
Supports preference pair datasets for DPO-style training.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class PreferencePair:
    """A single preference pair for training."""
    prompt: str
    chosen: str
    rejected: str
    prompt_id: Optional[str] = None
    task_score_chosen: Optional[float] = None
    task_score_rejected: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class PreferenceDataset(Dataset):
    """
    Dataset for preference pairs (x, y+, y-).

    Supports multiple formats:
    - JSON/JSONL files
    - HuggingFace datasets
    - Direct list of PreferencePair objects
    """

    def __init__(
        self,
        data: List[PreferencePair],
        tokenizer,
        max_length: int = 2048,
        max_prompt_length: int = 512,
        truncation_mode: str = "keep_end"
    ):
        """
        Initialize preference dataset.

        Args:
            data: List of PreferencePair objects
            tokenizer: Tokenizer for encoding
            max_length: Maximum total sequence length
            max_prompt_length: Maximum prompt length
            truncation_mode: How to truncate ("keep_start" or "keep_end")
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length
        self.truncation_mode = truncation_mode

        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self) -> int:
        return len(self.data)

    @staticmethod
    def _pair_from_item(item: Dict[str, Any], index: int) -> PreferencePair:
        """Validate and normalize one JSON record."""
        if not isinstance(item, dict):
            raise ValueError("Preference record %s must be an object" % index)
        prompt = item.get('prompt', '')
        chosen = item.get('chosen', item.get('preferred', ''))
        rejected = item.get('rejected', item.get('dispreferred', ''))
        if not all(isinstance(value, str) for value in (prompt, chosen, rejected)):
            raise ValueError("Preference record %s must contain string prompt/chosen/rejected fields" % index)
        if not chosen.strip() or not rejected.strip():
            raise ValueError("Preference record %s contains an empty candidate" % index)
        return PreferencePair(
            prompt=prompt,
            chosen=chosen,
            rejected=rejected,
            prompt_id=item.get('id'),
            task_score_chosen=item.get('task_score_chosen'),
            task_score_rejected=item.get('task_score_rejected'),
            metadata=item.get('metadata'),
        )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single training example."""
        pair = self.data[idx]

        # Encode chosen response
        chosen_encoded = self._encode_pair(pair.prompt, pair.chosen)

        # Encode rejected response
        rejected_encoded = self._encode_pair(pair.prompt, pair.rejected)

        return {
            'id': pair.prompt_id or str(idx),
            'prompts': pair.prompt,
            'chosen_responses': pair.chosen,
            'rejected_responses': pair.rejected,
            'chosen_input_ids': chosen_encoded['input_ids'],
            'chosen_attention_mask': chosen_encoded['attention_mask'],
            'chosen_labels': chosen_encoded['labels'],
            'rejected_input_ids': rejected_encoded['input_ids'],
            'rejected_attention_mask': rejected_encoded['attention_mask'],
            'rejected_labels': rejected_encoded['labels'],
            'task_scores_chosen': pair.task_score_chosen,
            'task_scores_rejected': pair.task_score_rejected,
            'metadata': pair.metadata or {},
        }

    def _encode_pair(
        self,
        prompt: str,
        response: str
    ) -> Dict[str, torch.Tensor]:
        """Encode a prompt-response pair with labels on response tokens only.

        The previous implementation inferred the prompt boundary from a padded,
        independently-truncated tokenization.  That can mask the complete response
        when the prompt is short (and can include padding in the loss).  Building the
        sequence from unpadded prompt and response token ids makes the boundary
        explicit and is compatible with standard HuggingFace causal LMs.
        """
        prompt_ids = self._tokenize_without_padding(prompt + "\n\n")
        response_ids = self._tokenize_without_padding(response)

        prompt_ids = self._truncate(prompt_ids, min(self.max_prompt_length, max(self.max_length - 1, 0)))
        max_response_tokens = max(self.max_length - len(prompt_ids) - 1, 0)
        response_ids = self._truncate(response_ids, max_response_tokens)

        token_ids = prompt_ids + response_ids
        if getattr(self.tokenizer, "eos_token_id", None) is not None and len(token_ids) < self.max_length:
            token_ids.append(int(self.tokenizer.eos_token_id))

        token_ids = token_ids[:self.max_length]
        response_start = min(len(prompt_ids), len(token_ids))
        valid_length = len(token_ids)

        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(self.tokenizer, "eos_token_id", 0)
        padded_ids = token_ids + [int(pad_token_id)] * (self.max_length - valid_length)
        attention_mask = [1] * valid_length + [0] * (self.max_length - valid_length)
        labels = [-100] * response_start + token_ids[response_start:]
        labels += [-100] * (self.max_length - len(labels))

        return {
            'input_ids': torch.tensor(padded_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
        }

    def _tokenize_without_padding(self, text: str) -> List[int]:
        """Tokenize text without padding, with a compatibility fallback for test tokenizers."""
        try:
            encoded = self.tokenizer(
                text,
                add_special_tokens=False,
                padding=False,
                truncation=False,
                return_tensors='pt',
            )
        except TypeError:
            # Lightweight tokenizers used in examples/tests may not expose the
            # HuggingFace keyword arguments.  Their attention mask still lets us
            # remove synthetic padding deterministically.
            encoded = self.tokenizer(
                text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
            )

        ids = encoded['input_ids'][0]
        mask = encoded.get('attention_mask')
        if mask is not None:
            ids = ids[mask[0].bool()]
        return [int(token_id) for token_id in ids.tolist()]

    def _truncate(self, token_ids: List[int], max_length: int) -> List[int]:
        """Truncate token ids according to the configured policy."""
        if max_length <= 0:
            return []
        if len(token_ids) <= max_length:
            return token_ids
        if self.truncation_mode == "keep_start":
            return token_ids[:max_length]
        if self.truncation_mode == "keep_end":
            return token_ids[-max_length:]
        raise ValueError("truncation_mode must be 'keep_start' or 'keep_end'")

    @classmethod
    def from_json(
        cls,
        path: str,
        tokenizer,
        **kwargs
    ) -> 'PreferenceDataset':
        """Load dataset from JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        if isinstance(raw_data, dict):
            raw_data = raw_data.get('data', raw_data.get('pairs', raw_data.get('records', [])))
        if not isinstance(raw_data, list):
            raise ValueError("JSON preference data must be a list or contain data/pairs/records")

        pairs = []
        for index, item in enumerate(raw_data):
            pairs.append(cls._pair_from_item(item, index))

        return cls(pairs, tokenizer, **kwargs)

    @classmethod
    def from_jsonl(
        cls,
        path: str,
        tokenizer,
        **kwargs
    ) -> 'PreferenceDataset':
        """Load dataset from JSONL file."""
        pairs = []
        with open(path, 'r', encoding='utf-8') as f:
            for index, line in enumerate(f):
                if line.strip():
                    item = json.loads(line)
                    pairs.append(cls._pair_from_item(item, index))

        return cls(pairs, tokenizer, **kwargs)


class ListwisePreferenceDataset(Dataset):
    """
    Dataset for listwise preference optimization with multiple candidates.
    """

    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer,
        num_candidates: int = 4,
        max_length: int = 2048
    ):
        """
        Initialize listwise dataset.

        Args:
            data: List of items with 'prompt' and 'candidates' keys
            tokenizer: Tokenizer for encoding
            num_candidates: Number of candidates per prompt (k)
            max_length: Maximum sequence length
        """
        self.data = data
        self.tokenizer = tokenizer
        self.num_candidates = num_candidates
        self.max_length = max_length

        if self.num_candidates < 2:
            raise ValueError("num_candidates must be at least 2")

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self) -> int:
        return len(self.data)

    @classmethod
    def from_json(
        cls,
        path: str,
        tokenizer,
        **kwargs,
    ) -> 'ListwisePreferenceDataset':
        """Load listwise records from JSON or a wrapped data root."""
        with open(path, 'r', encoding='utf-8') as handle:
            raw_data = json.load(handle)
        if isinstance(raw_data, dict):
            raw_data = raw_data.get('data', raw_data.get('records', raw_data.get('pairs', [])))
        if not isinstance(raw_data, list):
            raise ValueError("Listwise JSON must be a list or contain data/records/pairs")
        return cls(raw_data, tokenizer, **kwargs)

    @classmethod
    def from_jsonl(
        cls,
        path: str,
        tokenizer,
        **kwargs,
    ) -> 'ListwisePreferenceDataset':
        """Load one listwise record per JSONL line."""
        with open(path, 'r', encoding='utf-8') as handle:
            raw_data = [json.loads(line) for line in handle if line.strip()]
        return cls(raw_data, tokenizer, **kwargs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single training example with k candidates."""
        item = self.data[idx]
        prompt = item['prompt']
        candidates = item['candidates'][:self.num_candidates]
        if len(candidates) < 2:
            raise ValueError("Each listwise example must contain at least two candidates")

        candidate_texts = [
            candidate if isinstance(candidate, str) else str(
                candidate.get('response', candidate.get('text', candidate.get('completion', '')))
            )
            for candidate in candidates
        ]
        if any(not text.strip() for text in candidate_texts):
            raise ValueError("Listwise candidates must contain non-empty text")
        candidate_metadata = [
            candidate.get('metadata', {}) if isinstance(candidate, dict) else {}
            for candidate in candidates
        ]
        preferences = list(item.get('preferences', [1] + [0] * (len(candidates) - 1)))
        preferences.extend([0] * max(0, len(candidates) - len(preferences)))
        candidate_rewards = [
            float(candidate.get('reward', 0.0)) if isinstance(candidate, dict) else 0.0
            for candidate in candidates
        ]

        # Encode all candidates
        encoded_candidates = []
        for response in candidate_texts:
            encoded = self._encode_pair(prompt, response)
            encoded_candidates.append(encoded)

        # Pad to num_candidates if needed
        while len(encoded_candidates) < self.num_candidates:
            encoded_candidates.append(encoded_candidates[-1])
            candidate_texts.append(candidate_texts[-1])
            candidate_metadata.append({"padded": True})
            preferences.append(0)
            candidate_rewards.append(0.0)

        return {
            'id': item.get('id', str(idx)),
            'prompt': prompt,
            'candidates': candidate_texts[:self.num_candidates],
            'input_ids': torch.stack([e['input_ids'] for e in encoded_candidates]),
            'attention_mask': torch.stack([e['attention_mask'] for e in encoded_candidates]),
            'labels': torch.stack([e['labels'] for e in encoded_candidates]),
            'preferences': torch.tensor(preferences[:self.num_candidates], dtype=torch.float32),
            'rewards': torch.tensor(candidate_rewards[:self.num_candidates], dtype=torch.float32),
            'candidate_metadata': candidate_metadata[:self.num_candidates],
            'metadata': item.get('metadata', {}),
        }

    def _encode_pair(
        self,
        prompt: str,
        response: str
    ) -> Dict[str, torch.Tensor]:
        """Encode a prompt-response pair with response-only labels."""
        prompt_ids = self._tokenize_without_padding(prompt + "\n\n")
        response_ids = self._tokenize_without_padding(response)
        prompt_ids = prompt_ids[:max(self.max_length - 1, 0)]
        response_ids = response_ids[:max(self.max_length - len(prompt_ids) - 1, 0)]
        token_ids = prompt_ids + response_ids
        if getattr(self.tokenizer, "eos_token_id", None) is not None and len(token_ids) < self.max_length:
            token_ids.append(int(self.tokenizer.eos_token_id))
        token_ids = token_ids[:self.max_length]

        valid_length = len(token_ids)
        response_start = min(len(prompt_ids), valid_length)
        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(self.tokenizer, "eos_token_id", 0)
        labels = [-100] * response_start + token_ids[response_start:]
        labels += [-100] * (self.max_length - len(labels))
        return {
            'input_ids': torch.tensor(
                token_ids + [int(pad_token_id)] * (self.max_length - valid_length),
                dtype=torch.long,
            ),
            'attention_mask': torch.tensor(
                [1] * valid_length + [0] * (self.max_length - valid_length),
                dtype=torch.long,
            ),
            'labels': torch.tensor(labels, dtype=torch.long),
        }

    def _tokenize_without_padding(self, text: str) -> List[int]:
        """Tokenize text without padding for listwise examples."""
        try:
            encoded = self.tokenizer(
                text,
                add_special_tokens=False,
                padding=False,
                truncation=False,
                return_tensors='pt',
            )
        except TypeError:
            encoded = self.tokenizer(
                text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
            )
        ids = encoded['input_ids'][0]
        mask = encoded.get('attention_mask')
        if mask is not None:
            ids = ids[mask[0].bool()]
        return [int(token_id) for token_id in ids.tolist()]


def collate_preference_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for preference pairs.
    """
    collated = {
        'ids': [item.get('id', str(i)) for i, item in enumerate(batch)],
        'prompts': [item['prompts'] for item in batch],
        'chosen_responses': [item['chosen_responses'] for item in batch],
        'rejected_responses': [item['rejected_responses'] for item in batch],
        'chosen_input_ids': torch.stack([item['chosen_input_ids'] for item in batch]),
        'chosen_attention_mask': torch.stack([item['chosen_attention_mask'] for item in batch]),
        'chosen_labels': torch.stack([item['chosen_labels'] for item in batch]),
        'rejected_input_ids': torch.stack([item['rejected_input_ids'] for item in batch]),
        'rejected_attention_mask': torch.stack([item['rejected_attention_mask'] for item in batch]),
        'rejected_labels': torch.stack([item['rejected_labels'] for item in batch]),
        'metadata': [item.get('metadata') or {} for item in batch],
    }

    # Handle optional task scores
    if any(item.get('task_scores_chosen') is not None for item in batch):
        collated['task_scores_chosen'] = torch.tensor(
            [0.5 if item.get('task_scores_chosen') is None else item['task_scores_chosen'] for item in batch],
            dtype=torch.float32,
        )
        collated['task_scores_rejected'] = torch.tensor(
            [0.5 if item.get('task_scores_rejected') is None else item['task_scores_rejected'] for item in batch],
            dtype=torch.float32,
        )

    return collated


def collate_listwise_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate true listwise examples with shape ``[batch, candidates, tokens]``."""
    return {
        'ids': [item.get('id', str(index)) for index, item in enumerate(batch)],
        'prompts': [item['prompt'] for item in batch],
        'candidates': [item['candidates'] for item in batch],
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'labels': torch.stack([item['labels'] for item in batch]),
        'preferences': torch.stack([item['preferences'] for item in batch]),
        'rewards': torch.stack([item['rewards'] for item in batch]),
        'candidate_metadata': [item.get('candidate_metadata', []) for item in batch],
        'metadata': [item.get('metadata') or {} for item in batch],
    }


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    collate_fn=None
) -> DataLoader:
    """
    Create a DataLoader for training/evaluation.

    Args:
        dataset: Dataset to load from
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of data loading workers
        collate_fn: Custom collate function

    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn or collate_preference_batch,
        pin_memory=torch.cuda.is_available()
    )


def split_dataset(
    dataset: PreferenceDataset,
    train_ratio: float = 0.9,
    calibration_ratio: float = 0.02,
    seed: int = 42
) -> Tuple[PreferenceDataset, PreferenceDataset, PreferenceDataset]:
    """
    Split dataset into train, validation, and calibration sets.

    Args:
        dataset: Full dataset
        train_ratio: Fraction for training
        calibration_ratio: Fraction for calibration (from paper: 2%)
        seed: Random seed

    Returns:
        Tuple of (train_dataset, val_dataset, calibration_dataset)
    """
    import random
    random.seed(seed)

    n = len(dataset)
    groups: Dict[str, List[int]] = {}
    for index, pair in enumerate(dataset.data):
        # Keep all variants of a prompt in the same split to reduce leakage.
        group_key = str(pair.prompt_id or pair.prompt)
        groups.setdefault(group_key, []).append(index)
    group_keys = list(groups)
    random.shuffle(group_keys)
    target_calib = int(n * calibration_ratio)
    target_train = int(n * train_ratio)
    calib_indices: List[int] = []
    train_indices: List[int] = []
    val_indices: List[int] = []
    for key in group_keys:
        group = groups[key]
        if len(calib_indices) < target_calib:
            calib_indices.extend(group)
        elif len(train_indices) < target_train:
            train_indices.extend(group)
        else:
            val_indices.extend(group)

    train_data = [dataset.data[i] for i in train_indices]
    val_data = [dataset.data[i] for i in val_indices]
    calib_data = [dataset.data[i] for i in calib_indices]

    return (
        PreferenceDataset(train_data, dataset.tokenizer,
                         dataset.max_length, dataset.max_prompt_length),
        PreferenceDataset(val_data, dataset.tokenizer,
                         dataset.max_length, dataset.max_prompt_length),
        PreferenceDataset(calib_data, dataset.tokenizer,
                         dataset.max_length, dataset.max_prompt_length)
    )
