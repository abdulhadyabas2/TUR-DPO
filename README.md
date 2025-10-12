# TUR-DPO: Topology- and Uncertainty-Aware Direct Preference Optimization

This repository accompanies the paper:

> Abdulhady A. Abdullah, et al.  
> Topology- and Uncertainty-Aware Direct Preference Optimization (TUR-DPO).  
> 2025. Preprint / under review.

---

## Overview

TUR-DPO augments DPO with two signals:
- Topology-aware structure score over lightweight reasoning graphs.
- Uncertainty-aware weighting that down-weights brittle pairs.

It preserves RL-free simplicity while improving calibration and faithfulness.

## Layout

```
TUR-DPO/
├─ README.md
├─ LICENSE
├─ CITATION.cff
├─ requirements.txt
├─ configs/
│  ├─ base.yaml
│  ├─ turdpo_llama.yaml
│  └─ turdpo_gptj.yaml
├─ src/
│  ├─ __init__.py
│  ├─ models/
│  │  ├─ tur_dpo_trainer.py
│  │  ├─ topology_graph.py
│  │  ├─ uncertainty_layer.py
│  │  └─ reward_estimator.py
│  ├─ utils/
│  │  ├─ data_utils.py
│  │  ├─ metrics.py
│  │  ├─ plotting.py
│  │  └─ logging_utils.py
│  ├─ data/
│  │  └─ README.md
│  └─ experiments/
│     ├─ train_turdpo.py
│     ├─ eval_turdpo.py
│     └─ plots.py
├─ results/
│  ├─ logs/
│  ├─ figures/
│  └─ tables/
└─ paper/
   ├─ main.tex (see top-level latex/ folder)
   └─ figures/
```

## Quickstart (Windows PowerShell)

```
python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/prepare_dummy_data.py
python src/experiments/train_turdpo.py --data data/sample_pairs.jsonl --output results/logs/demo_run
```

This runs a toy pass that computes TUR-DPO losses using heuristic topology/uncertainty and fake log-probs.

To plug in real model log-probs, adapt `src/experiments/train_turdpo.py` to use your HF model, or reuse helpers from `turdpo/models.py` in the minimal demo.

## Reproduce simple plots

```
python src/experiments/plots.py --input results/tables/demo_curve.csv --out results/figures/demo_curve.png
```

## License
Apache-2.0
