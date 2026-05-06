# HierEpiGNN — Anonymous Submission

A graph-free dual-branch epidemic forecaster with a fusion-horizon Mixture-of-Experts head and an LLM semantic prior over per-case strain descriptions. Trained and evaluated on three epidemic datasets: Avian Influenza Virus (AIV), US COVID-19, and Japan COVID-19.

This repository contains the code, configuration, preprocessing pipeline, and tests required to reproduce the results reported in the accompanying manuscript. Authorship and affiliation are intentionally omitted for double-blind review.

---

## Architecture

```
GraphFreeDualBranchForecaster
├── CaseEventBranch                  case features → z_case
├── LocationTemporalBranch           location features → z_loc
├── SharedPrivateFusionHead          z_case, z_loc → s
├── HorizonCrossAttentionFusion      s, per-case LLM tokens → s_h
└── MoEHorizonHead                   s_h, y_hist → ŷ
    ├── shared expert (2-layer MLP)
    ├── per-horizon experts (one MLP per H)
    ├── adaptive mixing (sigmoid α_h ∈ R^H)
    └── PersistenceAnchor (last + linear-trend baseline)
```

Loss: `L_pred (Huber log1p) + decomp_loss_weight · L_decomp (cosine²(shared, private))`.

A complete description of the architecture and configuration surface lives in [`docs/architecture-summary.md`](docs/architecture-summary.md). The ASCII pipeline diagram is in [`structure`](structure).

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

Raw datasets are not redistributed with this artifact. Reviewers can obtain them from the original public sources:

| Dataset | Source | Place under |
|---|---|---|
| AIV (HPAI surveillance) | GISAID EpiFlu + APHIS HPAI Detections | `data/aiv/` |
| COVID-19 (USA) | GISAID EpiCoV + JHU CSSE confirmed cases | `data/covid/` |
| COVID-19 (Japan) | GISAID EpiCoV + MHLW prefecture-level cases | `data/japan/` |

Once the raw data is in place, run the preprocessing pipeline:

```bash
python -m data_processing.run_all                    # process all three datasets
python -m data_processing.run_all --aiv              # subset
python -m data_processing.run_all --fuzzy-threshold 90
```

Outputs land under `data/processed/{aiv,covid,japan}/`.

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

The list of as-of dates can be enumerated automatically from the dataset config:

```bash
python -m src.data.dataset_origins --config configs/covid.yaml \
    --output data/processed/covid/as_of_dates.json
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

Optional CLI overrides: `--max_epochs`, `--seed`. Full usage is documented in [`USAGE.md`](USAGE.md).

The reported metrics are produced by `src/evaluation/metrics.py` (`compute_all_metrics`, `compute_per_horizon_metrics`): MAE, RMSE, MAPE, sMAPE, PearsonR, SpearmanR, OutbreakAUROC, OutbreakAUPRC, CRPS, Coverage50, Coverage90.

---

## Tests

```bash
pytest tests/ -q
```

The suite covers the data contract, the model forward/backward path, the loss, the trainer scheduler, and the four-card causal-cutoff embedding cache.

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
structure                # ASCII architecture diagram
USAGE.md                 # CLI reference
```

---

## Citation

A BibTeX entry will be released with the camera-ready version. For now, please cite the manuscript by its submission ID in the venue's review system.

---

## License

This artifact is released for the purpose of double-blind peer review. A permissive open-source license (e.g. Apache 2.0) will accompany the camera-ready release.
