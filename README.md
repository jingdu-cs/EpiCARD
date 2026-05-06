# EpiCARD

Code for paper "EpiCARD: Graph-Prior-Free Epidemic Forecasting with Card-Guided Case Semantics".

This repository contains the code, configuration, preprocessing pipeline, and tests.

---

## Architecture

```
EpiCARD
├── CaseEventBranch                  case features → z_case
├── LocationTemporalBranch           location features → z_loc
├── SharedPrivateFusionHead          z_case, z_loc → s
├── HorizonCrossAttentionFusion      s, per-case LLM tokens → s_h
└── MoEHorizonHead                   s_h, y_hist → ŷ
```

Loss: `L_pred (Huber log1p) + decomp_loss_weight · L_decomp (cosine²(shared, private))`.


---

## Installation

The code targets Python ≥ 3.8 and has been tested under PyTorch 2.x.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .          # installs the local `src/` package
```

For GPU training, install a PyTorch wheel that matches your CUDA toolchain (see https://pytorch.org/get-started/locally/) before running `pip install -r requirements.txt`.

---

## Data
One sample dataset are provided here: https://figshare.com/s/509fd12a77f550dc0529 

### Strain LLM embeddings

The model consumes precomputed sample-level LLM embeddings indexed by `Unique_Identifier`. Generate them with:

```bash
python -m data_processing.generate_sample_embeddings \
  --dataset covid \
  --causal-cutoff \
  --as-of-dates data/processed/covid/as_of_dates.json \
  --model-name meta-llama/Meta-Llama-3-8B \
  --batch-size 4 --device cuda
```

Embedding generation requires `transformers` and a HuggingFace-compatible weight cache; canonical training does not (only the resulting `.pt` file is read).

---

## Training & evaluation

```bash
# Train
python scripts/run_train.py --config configs/aiv.yaml   --gpu 0
python scripts/run_train.py --config configs/covid.yaml --gpu 0
python scripts/run_train.py --config configs/japan.yaml --gpu 0

# Evaluate a checkpoint on the held-out test split
python scripts/run_eval.py --config configs/aiv.yaml \
    --checkpoint checkpoints/aiv_42/best.pt --gpu 0
```

Optional CLI overrides: `--max_epochs`, `--seed`.

The reported metrics are produced by `src/evaluation/metrics.py` (`compute_all_metrics`, `compute_per_horizon_metrics`): MAE, RMSE, MAPE, sMAPE, PearsonR, SpearmanR, OutbreakAUROC, OutbreakAUPRC.

---

## Tests

```bash
pytest tests/ -q
```
---

## Repository layout

```
configs/                 # default + per-dataset YAMLs
data_processing/         # raw → processed pipeline + LLM-embedding generator
docs/                    # architecture summary, PRD
scripts/                 # train and eval entry points
src/
  data/                  # datasets, feature builders, splits
  models/                # model + fusion-horizon modules
  training/              # trainer, combined loss
  evaluation/            # metrics
  utils/                 # seed helpers
tests/                   # unit + integration tests
```

---
