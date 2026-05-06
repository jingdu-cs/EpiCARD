"""Enumerate the as-of dates needed by the v2 embedding cache.

The causal-cutoff Window Card requires one snapshot per forecast origin
actually used by the sliding-window evaluation. This helper loads a
dataset config, instantiates the matching :class:`EpidemicDataset`, and
writes the deterministic list of ``window_end - reporting_lag_days``
ISO dates to JSON.

The output file is consumed by ``data_processing.generate_sample_embeddings``
via ``--as-of-dates path/to/origins.json``.

Usage::

    python -m src.data.dataset_origins \\
        --config configs/covid.yaml \\
        --output data/processed/covid/as_of_dates.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from src.data import build_dataset
from src.training.trainer import load_config as _load_config_with_defaults

logger = logging.getLogger(__name__)


def enumerate_as_of_dates(cfg: dict, dataset: str) -> list[str]:
    """Return the unique sorted ISO as-of dates implied by the dataset's
    sliding-window grid and ``data.window_card.reporting_lag_days``.
    """
    ds = build_dataset(cfg, dataset)
    wc = (cfg.get("data") or {}).get("window_card") or {}
    lag_days = int(wc.get("reporting_lag_days", 14))
    seen: set[str] = set()
    for _, window_end in ds.time_windows:
        as_of = pd.Timestamp(window_end) - pd.Timedelta(days=lag_days)
        seen.add(as_of.date().isoformat())
    return sorted(seen)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--dataset", type=str, default=None,
        help=(
            "Override dataset name; default reads cfg['dataset_name'] "
            "(falls back to cfg['dataset'])."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cfg = _load_config_with_defaults(str(args.config))
    dataset = args.dataset or cfg.get("dataset_name") or cfg.get("dataset")
    if not dataset:
        raise SystemExit(
            "dataset not specified (pass --dataset or set 'dataset_name' in the config)."
        )

    iso_dates = enumerate_as_of_dates(cfg, dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(iso_dates, indent=2))
    print(f"Wrote {len(iso_dates)} as-of dates to {args.output}")


if __name__ == "__main__":
    main()
