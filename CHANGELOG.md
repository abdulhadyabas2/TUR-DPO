# Changelog

All notable changes to TUR-DPO will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-01-XX

### Added

- **Core Algorithm Implementation**
  - TUR-DPO loss function with uncertainty-weighted preference optimization
  - Listwise TUR-DPO loss for comparing multiple responses
  - Baseline DPO and IPO loss implementations

- **Topology Module**
  - `TopologyGraph` class for representing reasoning graphs
  - `TopologyExtractor` for eliciting graphs from LLM responses
  - `TopologyScorer` implementing Equation 1 from the paper
  - Graph quality metrics: path coverage, cycle detection, dangling nodes, contradiction detection

- **Uncertainty Estimation**
  - `EpistemicUncertainty` using variance + Jensen-Shannon divergence
  - `AleatoricUncertainty` using smoothed entropy
  - `UncertaintyEstimator` combining both types (Equation 3)
  - `PairWeightComputer` for computing preference pair weights (Equation 6)

- **Reward Shaping**
  - `ShapedReward` implementing Equation 7
  - `SemanticScorer` for semantic quality assessment (Equation 2)
  - `LinearCalibrator` for reward calibration (Equation 8)
  - `RewardDifferenceComputer` for computing reward differences

- **Training Infrastructure**
  - `TURDPOTrainer` with complete training loop
  - `TURDPOConfig` dataclass for configuration management
  - EMA model updates
  - Gradient accumulation support
  - Checkpoint saving and loading

- **Verification Module**
  - `NodeVerifier` for multi-type node verification
  - `FactChecker` for factual correctness
  - `ArithmeticVerifier` for mathematical expressions
  - `ContradictionDetector` for finding logical contradictions

- **Calibration Module**
  - Expected Calibration Error (ECE) computation
  - Brier score calculation
  - Temperature scaling calibration
  - Isotonic regression calibration

- **Data Handling**
  - `PreferenceDataset` for pairwise preferences
  - `ListwisePreferenceDataset` for listwise ranking
  - Flexible data loading utilities

- **Utilities**
  - Evaluation metrics: Exact Match, Token F1, ROUGE-L
  - Statistical utilities: bootstrap CI, Cohen's d
  - EMA model wrapper
  - Logging setup

### Documentation

- Comprehensive README with installation and usage instructions
- API documentation with mathematical formulations
- Example scripts demonstrating usage
- Contributing guidelines

### Dependencies

- PyTorch >= 2.0.0
- Transformers >= 4.30.0
- NumPy >= 1.21.0
- SciPy >= 1.7.0
- tqdm >= 4.60.0

## [Unreleased]

### Publication-readiness updates

- Added response-only causal-LM labels and padding-safe sequence log-probabilities.
- Added strict topology JSON parsing, model-backed elicitation, graph annotation caching,
  minimal-path peer-pressure scoring, and optional NLI verification.
- Fixed gradient accumulation, max-step handling, cosine decay, EMA/frozen reference
  handling, dataset split leakage, CLI config loading, and checkpoint resume support.
- Wired true multi-candidate listwise batches into the trainer and added explicit
  calibrator fitting plus optional judge/hard-negative pair-construction hooks.
- Added reproducible pair construction, prediction evaluation, answer post-processing,
  requirements, regression tests, claim audit, and an explicit reproducibility scope.

### Planned Features

- Multi-GPU distributed training support
- Integration with Hugging Face TRL library
- Pre-trained model checkpoints
- Visualization tools for reasoning graphs
- Extended benchmark evaluation scripts

---

## Version History

- **0.1.0**: Initial release with core TUR-DPO implementation
