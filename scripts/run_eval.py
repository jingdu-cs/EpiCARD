"""Entry point for evaluating a trained forecasting checkpoint.

Usage:
    python scripts/run_eval.py --config configs/aiv.yaml --checkpoint checkpoints/aiv_42/best.pt
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import build_dataset  # noqa: E402
from src.data.transforms import SplitBuilder  # noqa: E402
from src.models import build_model  # noqa: E402
from src.evaluation.metrics import (  # noqa: E402
    PER_HORIZON_REPORT_METRICS,
    REPORT_METRICS,
    compute_all_metrics,
    compute_per_horizon_metrics,
)
from src.training.trainer import build_dataloaders, load_config  # noqa: E402
from src.utils.seed import set_global_seed  # noqa: E402

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the graph-free dual-branch forecaster"
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pt)")
    parser.add_argument("--gpu", type=int, default=None,
                        help="GPU device index (e.g. --gpu 3 for cuda:3)")
    args = parser.parse_args()

    cfg = load_config(args.config)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    seed = cfg.get("seed", 42)
    set_global_seed(seed)

    if args.gpu is not None and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_name = cfg["dataset_name"]

    dataset = build_dataset(cfg, dataset_name)

    train_idx, val_idx, test_idx = SplitBuilder.temporal_split(
        dataset,
        train_ratio=cfg["data"]["train_ratio"],
        val_ratio=cfg["data"]["val_ratio"],
    )

    _, _, test_loader = build_dataloaders(dataset, train_idx, val_idx, test_idx, cfg)

    d_loc = dataset[0]["location_graph"]["x"].shape[1]
    num_locations = dataset[0]["location_graph"]["x"].shape[0]
    model = build_model(cfg, d_loc, num_locations=num_locations).to(device)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    outbreak_threshold = ckpt.get("outbreak_threshold", 0.0)
    logger.info("Loaded checkpoint: %s (epoch %d)", args.checkpoint, ckpt.get("epoch", -1))

    # Evaluate
    model.eval()
    all_pred_mean, all_pred_std, all_targets, all_mask = [], [], [], []
    all_population: list = []
    with torch.no_grad():
        for batch in test_loader:
            # Move batch to device
            def _to_device(x):
                if isinstance(x, torch.Tensor):
                    return x.to(device)
                if isinstance(x, dict):
                    return {k: _to_device(v) for k, v in x.items()}
                return x
            batch = _to_device(batch)
            output = model(batch)
            all_pred_mean.append(output["pred_mean"].cpu())
            all_pred_std.append(output["pred_std"].cpu())
            all_targets.append(batch["targets"].cpu())
            all_mask.append(batch["mask"].cpu())
            if "population" in batch:
                all_population.append(batch["population"].cpu())

    pred_mean = torch.cat(all_pred_mean, dim=0)
    pred_std = torch.cat(all_pred_std, dim=0)
    targets = torch.cat(all_targets, dim=0)
    mask = torch.cat(all_mask, dim=0)

    # Per-capita normalization: only forward population when flag is on, so
    # that COVID/AIV runs (inheriting the default false) are unchanged.
    per_capita_normalize = cfg["data"].get("per_capita_normalize", False)
    per_capita_base = float(cfg["data"].get("per_capita_base", 100_000))
    population = None
    if per_capita_normalize and all_population:
        population = torch.cat(all_population, dim=0)

    metrics = compute_all_metrics(
        pred_mean, pred_std, targets, mask, cfg, outbreak_threshold, None,
        population=population, per_capita_base=per_capita_base,
        metric_names=REPORT_METRICS,
    )
    per_horizon = compute_per_horizon_metrics(
        pred_mean, pred_std, targets, mask, cfg, outbreak_threshold, None,
        population=population, per_capita_base=per_capita_base,
        metric_names=PER_HORIZON_REPORT_METRICS,
    )

    logger.info("Evaluation results on %s:", dataset_name)
    for name in REPORT_METRICS:
        value = metrics[name]
        logger.info("  %s: %.4f", name, value)
    for horizon in ("h1", "h2", "h4"):
        if horizon in per_horizon:
            logger.info(
                "  %s RMSE: %.4f | %s MAE: %.4f",
                horizon,
                per_horizon[horizon]["RMSE"],
                horizon,
                per_horizon[horizon]["MAE"],
            )

    # Save results
    results_dir = os.path.join("results", "eval", dataset_name)
    os.makedirs(results_dir, exist_ok=True)
    results = {
        "experiment_type": "evaluation",
        "dataset": dataset_name,
        "seed": seed,
        "checkpoint": args.checkpoint,
        "metrics": metrics,
        "per_horizon_metrics": per_horizon,
    }
    out_path = os.path.join(results_dir, f"eval_{seed}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
