"""Entry point for training the graph-free dual-branch forecaster.

Usage:
    python scripts/run_train.py --config configs/aiv.yaml
    python scripts/run_train.py --config configs/covid.yaml --max_epochs 2
"""

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import build_dataset
from src.data.transforms import SplitBuilder
from src.models import build_model
from src.training.trainer import Trainer, build_dataloaders, load_config
from src.utils.seed import set_global_seed

logger = logging.getLogger(__name__)


def _nan_to_none(obj: object) -> object:
    """Recursively replace float NaN with None for JSON-safe serialization."""
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nan_to_none(v) for v in obj]
    return obj


def _setup_logging(cfg: dict) -> None:
    """Configure root logger with console + file output."""
    log_dir = cfg.get("training", {}).get("log_dir", "logs")
    os.makedirs(log_dir, exist_ok=True)

    dataset_name = cfg.get("dataset_name", "unknown")
    seed = cfg.get("seed", 42)
    log_file = os.path.join(log_dir, f"train_{dataset_name}_{seed}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file),
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the graph-free dual-branch forecaster"
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to dataset config YAML")
    parser.add_argument("--max_epochs", type=int, default=None,
                        help="Override max_epochs from config")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override seed from config")
    parser.add_argument("--gpu", type=int, default=None,
                        help="GPU device index (e.g. --gpu 3 for cuda:3)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.max_epochs is not None:
        cfg["training"]["max_epochs"] = args.max_epochs
    if args.seed is not None:
        cfg["seed"] = args.seed

    _setup_logging(cfg)

    seed = cfg.get("seed", 42)
    set_global_seed(seed)

    if args.gpu is not None and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    dataset_name = cfg["dataset_name"]
    logger.info("Dataset: %s | Seed: %d", dataset_name, seed)

    dataset = build_dataset(cfg, dataset_name)
    # Get location feature dim and count from first sample
    sample0 = dataset[0]
    d_loc = sample0["location_graph"]["x"].shape[1]
    num_locations = sample0["location_graph"]["x"].shape[0]
    logger.info(
        "Dataset loaded: %d windows, d_loc=%d, num_locations=%d",
        len(dataset), d_loc, num_locations,
    )

    train_idx, val_idx, test_idx = SplitBuilder.temporal_split(
        dataset,
        train_ratio=cfg["data"]["train_ratio"],
        val_ratio=cfg["data"]["val_ratio"],
    )

    train_loader, val_loader, test_loader = build_dataloaders(
        dataset, train_idx, val_idx, test_idx, cfg,
    )

    model = build_model(cfg, d_loc, num_locations=num_locations)
    num_params = sum(p.numel() for p in model.parameters())
    logger.info("Model params: %s", f"{num_params:,}")

    trainer = Trainer(model, cfg, device)
    best_metrics = trainer.train(train_loader, val_loader)
    logger.info("Training complete. Best val metrics: %s", best_metrics)

    test_metrics, _ = trainer.evaluate(test_loader)
    logger.info("Test metrics:")
    for name, value in test_metrics.items():
        logger.info("  %s: %.4f", name, value)

    # Save results as JSON for comparison with CV runs
    results = {
        "experiment_type": "single_split",
        "dataset": dataset_name,
        "seed": seed,
        "split": {
            "train_size": len(train_idx),
            "val_size": len(val_idx),
            "test_size": len(test_idx),
        },
        "test_metrics": test_metrics,
    }
    out_dir = os.path.join("results", "single_split", dataset_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"train_{seed}.json")
    with open(out_path, "w") as f:
        json.dump(_nan_to_none(results), f, indent=2)
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
