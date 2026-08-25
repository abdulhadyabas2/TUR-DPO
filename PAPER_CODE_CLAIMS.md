# Paper-to-code claim audit

This file records what the current repository implements and what still needs
external experimental artifacts. It is intentionally conservative: an API or
hook is not treated as an experimental result.

## Implemented in the repository

- Pairwise TUR-DPO Equation 9 with policy and frozen/reference-model
  response log-probabilities.
- Fixed and EMA reference modes, CLI/config alignment, gradient accumulation,
  maximum optimizer steps, warmup/cosine scheduling, checkpointing, and seeded
  Python/NumPy/PyTorch/CUDA runs.
- JSON topology elicitation template, robust parser, rule-based fallback,
  cycle/self-loop sanitization, near-duplicate node merging, node and edge path
  coverage, and optional model-backed extraction.
- Optional NLI sequence-classification verification with explicit heuristic
  fallback, contradiction detection, semantic scoring, and uncertainty
  integration in the training path.
- Peer-pressure structural penalty and shaped-reward scalar. Both are active
  when `alpha_peer_unverified` and `lambda_peer_pressure` are nonzero; zero is
  retained as the backwards-compatible default.
- True listwise batches with `B x K` candidates and `ListwiseTURDPOLoss`.
- Annotation caching, response-only labels, exact split manifest, basic answer
  post-processing, exact match/F1/ROUGE-L plus calibration/structural metrics,
  bootstrap summaries, multi-seed experiment orchestration, AMP, and
  reproducibility docs.

## Optional interfaces, not reproduced experiments

- Pair construction accepts side-flipped judge callbacks, reference-logprob
  hard-negative mining, and lexical near-duplicate filtering. A released judge,
  calibration labels, and its revision/configuration are still required to
  reproduce a paper-specific judge pipeline.
- `LinearCalibrator.fit` and `ShapedReward.fit_calibrators` fit parameters only
  when the caller supplies a held-out target set. The repository does not invent
  human or judge labels and does not silently fit on training data.
- A topology or verifier model is optional. Without one, the documented
  deterministic fallback is used and must not be described as NLI/model-backed
  evidence.

## Not contained in this code repository

- The paper's six benchmark datasets, 614k-pair corpus, exact 7--8B model
  checkpoints/revisions, raw annotation caches, or reported result tables.
- Human evaluation, LLM-judge calibration results, PPO baselines, ablation and
  robustness result tables, confidence intervals, or statistical claims.
- Multimodal model training/evaluation and the paper's long-context benchmark
  experiments.

Those claims can be made only after the corresponding data, model revisions,
evaluation protocol, raw outputs, and independent reruns are released.
