# Reproducibility status

The repository now provides the core TUR-DPO implementation and an auditable
annotation path:

- `scripts/construct_pairs.py` creates JSON/JSONL records with topology graphs,
  semantic scores, uncertainty components, and pair weights.
- `train.py --config configs/default.json` runs the policy/reference training
  loop with response-only causal-LM labels, gradient accumulation, warmup and
  cosine decay, optional EMA reference updates, and checkpointing.
- `scripts/evaluate.py` computes exact match, token F1, and ROUGE-L from saved
  predictions, calibration/structural metrics when fields are present, bootstrap
  intervals, and optionally applies the documented answer guardrail.
- `scripts/run_experiments.py` and `scripts/aggregate_runs.py` provide a local
  multi-seed/ablation matrix and aggregate scalar training metrics.
- `scripts/normalize_dataset.py` maps common benchmark schemas to canonical
  records without fabricating rejected responses, and `scripts/aggregate_judgments.py`
  summarizes supplied human/LLM judgment records.
- `use_listwise=True` now consumes real multi-candidate batches; pair construction
  also exposes optional side-flipped judge hooks, reference-logprob hard-negative
  mining, and lexical near-duplicate filtering.

The following manuscript claims require separate artifacts and are not implied
by the example data or unit tests: benchmark win-rates, human or LLM-judge
studies, PPO comparisons, multimodal experiments, long-context benchmark
results, calibrated judge/human labels, and private/full-corpus statistics. Do
not report those values from this repository unless the corresponding datasets,
checkpoints, evaluator configuration, and raw outputs are released and
independently rerun. The optional pair-construction hooks are interfaces for
those artifacts, not substitutes for them.

For a real release, record at minimum:

1. model and tokenizer identifiers plus revision hashes;
2. verifier and topology-elicitor identifiers plus revision hashes;
3. dataset commit/version and split manifest;
4. seed, hardware, batch size, accumulation, precision, and all config values;
5. annotation cache checksums and prediction files;
6. aggregate metrics with confidence intervals and the exact evaluation command.
