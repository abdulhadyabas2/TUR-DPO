"""Tests for the JSON annotation cache format."""

from scripts.construct_pairs import construct_preference_pairs


def test_pair_constructor_accepts_string_candidates_and_deduplicates():
    raw = [
        {
            "id": "a",
            "prompt": "1+1?",
            "candidates": [
                {"text": "2", "score": 1.0},
                {"text": "3", "score": 0.0},
            ],
        },
        {
            "id": "b",
            "prompt": "1+1?",
            "chosen": "2",
            "rejected": "3",
        },
    ]
    pairs = construct_preference_pairs(raw, k_samples=1, include_graphs=False)
    assert len(pairs) == 1
    metadata = pairs[0]["metadata"]
    assert metadata["annotation_version"] == "turdpo-annotations-v1"
    assert 0.05 <= metadata["pair_weight"] <= 1.0
