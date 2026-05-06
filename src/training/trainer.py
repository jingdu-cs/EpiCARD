"""Trainer, config loader, and dataloader builder for the forecasting stack.

Provides:
- Trainer: full training loop with gradient accumulation, early stopping,
  LR scheduling (warmup + cosine), checkpointing, and NaN detection.
- load_config: deep-merge dataset YAML over default.yaml.
- build_dataloaders: create train/val/test DataLoaders from index splits.
"""

from __future__ import annotations

import copy
import logging
import os
from pathlib import Path
from typing import Any, Collection

import numpy as np
import torch
import yaml
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset, Subset

from src.evaluation.metrics import compute_all_metrics, compute_per_horizon_metrics
from src.training.losses import CombinedLoss

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config utilities
# ---------------------------------------------------------------------------


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base*, returning a new dict.

    Values in *override* take precedence. Nested dicts are merged recursively;
    all other types are replaced outright.
    """
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(config_path: str) -> dict:
    """Load dataset config and deep-merge with configs/default.yaml.

    Parameters
    ----------
    config_path : str
        Path to the dataset-specific YAML config file.

    Returns
    -------
    dict
        Merged configuration with dataset values overriding defaults.
    """
    config_dir = Path(config_path).parent
    default_path = config_dir / "default.yaml"

    with open(default_path, "r") as f:
        default_cfg = yaml.safe_load(f) or {}

    with open(config_path, "r") as f:
        dataset_cfg = yaml.safe_load(f) or {}

    merged = _deep_merge(default_cfg, dataset_cfg)
    logger.info("Config loaded: default + %s", config_path)
    return merged


# ---------------------------------------------------------------------------
# DataLoader utilities
# ---------------------------------------------------------------------------


def _identity_collate(batch: list) -> Any:
    """Identity collate function — unwrap the single-item list."""
    return batch[0]


def build_dataloaders(
    dataset: Dataset,
    train_indices: list[int],
    val_indices: list[int],
    test_indices: list[int],
    cfg: dict,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build DataLoaders with Subset for each split.

    Parameters
    ----------
    dataset : Dataset
        Full EpidemicDataset.
    train_indices, val_indices, test_indices : list[int]
        Index lists from SplitBuilder.
    cfg : dict
        Full config; reads cfg["training"]["batch_size"] and
        cfg["training"]["num_workers"].

    Returns
    -------
    tuple[DataLoader, DataLoader, DataLoader]
        (train_loader, val_loader, test_loader).
    """
    train_cfg = cfg["training"]
    batch_size = train_cfg["batch_size"]
    num_workers = train_cfg["num_workers"]

    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=_identity_collate,
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_identity_collate,
    )
    test_loader = DataLoader(
        Subset(dataset, test_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_identity_collate,
    )

    logger.info(
        "DataLoaders built: train=%d, val=%d, test=%d (batch_size=%d)",
        len(train_indices), len(val_indices), len(test_indices), batch_size,
    )
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Batch device helper
# ---------------------------------------------------------------------------


from src.evaluation._utils import move_batch_to_device as _move_batch_to_device  # noqa: E402


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class Trainer:
    """Training loop for the forecasting model with early stopping and checkpointing.

    Parameters
    ----------
    model : torch.nn.Module
        Forecasting model instance.
    cfg : dict
        Full merged configuration dictionary.
    device : torch.device
        Target device (cpu or cuda).
    """

    def __init__(self, model: torch.nn.Module, cfg: dict, device: torch.device) -> None:
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device

        train_cfg = cfg["training"]

        # Per-capita target normalization flags
        self.per_capita_normalize: bool = cfg["data"].get(
            "per_capita_normalize", False
        )
        self.per_capita_base: float = float(
            cfg["data"].get("per_capita_base", 100_000)
        )
        # Loss function
        self.loss_fn = CombinedLoss(cfg)

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=train_cfg["lr"],
            weight_decay=train_cfg["weight_decay"],
        )

        # LR scheduler: warmup + cosine decay
        warmup_epochs = int(train_cfg["warmup_epochs"])
        max_epochs = int(train_cfg["max_epochs"])
        warmup_iters = max(1, min(warmup_epochs, max_epochs))

        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            total_iters=warmup_iters,
        )
        if max_epochs > warmup_epochs:
            cosine_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=max_epochs - warmup_epochs,
                eta_min=train_cfg["min_lr"],
            )
            self.scheduler: LRScheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs],
            )
        else:
            logger.info(
                "LR scheduler: warmup-only (max_epochs=%d <= warmup_epochs=%d)",
                max_epochs, warmup_epochs,
            )
            self.scheduler = warmup_scheduler

        # Early stopping state
        self.monitor_metric: str = train_cfg.get("monitor_metric", "RMSE")
        self.best_val_metric: float = float("inf")
        self.patience_counter: int = 0

        # Outbreak threshold (set during train())
        self.outbreak_threshold: float = 0.0

        # Checkpoint directory (resolved in train())
        self.checkpoint_dir: str = train_cfg["checkpoint_dir"]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> dict:
        """Run the full training loop.

        Parameters
        ----------
        train_loader : DataLoader
            Training data loader.
        val_loader : DataLoader
            Validation data loader.

        Returns
        -------
        dict
            Best validation metrics.
        """
        train_cfg = self.cfg["training"]
        max_epochs = train_cfg["max_epochs"]
        patience = train_cfg["patience"]

        # Compute outbreak threshold from training targets
        self._compute_outbreak_threshold(train_loader)

        # Resolve checkpoint directory
        dataset_name = self._infer_dataset_name(train_loader)
        seed = self.cfg.get("seed", 42)
        ckpt_dir = os.path.join(
            self.checkpoint_dir, f"{dataset_name}_{seed}"
        )
        os.makedirs(ckpt_dir, exist_ok=True)

        # Log directory
        log_dir = train_cfg.get("log_dir", "logs")
        os.makedirs(log_dir, exist_ok=True)

        best_metrics: dict = {}

        for epoch in range(1, max_epochs + 1):
            # Train one epoch
            train_losses = self._train_one_epoch(train_loader)

            # Validate
            val_losses, val_metric = self._validate(val_loader)

            # LR scheduler step
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Per-epoch logging
            logger.info(
                "Epoch %d/%d | "
                "train_loss=%.4f (pred=%.4f) | val_loss=%.4f | "
                "val_%s=%.4f | lr=%.2e",
                epoch, max_epochs,
                train_losses["loss"], train_losses["loss_pred"],
                val_losses["loss"],
                self.monitor_metric, val_metric, current_lr,
            )

            # Early stopping check
            if val_metric < self.best_val_metric:
                self.best_val_metric = val_metric
                self.patience_counter = 0
                self._save_checkpoint(
                    os.path.join(ckpt_dir, "best.pt"), epoch, val_metric,
                )
                # Compute best metrics via full evaluation
                best_metrics = self._compute_val_metrics(val_loader)
                logger.info("New best val_%s=%.4f at epoch %d", self.monitor_metric, val_metric, epoch)
            else:
                self.patience_counter += 1

            # Save latest checkpoint
            self._save_checkpoint(
                os.path.join(ckpt_dir, "latest.pt"), epoch, val_metric,
            )

            if self.patience_counter >= patience:
                logger.info(
                    "Early stopping at epoch %d (patience=%d)", epoch, patience,
                )
                break

        # Load best checkpoint for final state
        best_ckpt_path = os.path.join(ckpt_dir, "best.pt")
        if os.path.exists(best_ckpt_path):
            self._load_checkpoint(best_ckpt_path)
            logger.info("Loaded best checkpoint from %s", best_ckpt_path)

        return best_metrics

    def evaluate(
        self,
        data_loader: DataLoader,
        *,
        metric_names: Collection[str] | None = None,
        per_horizon_metric_names: Collection[str] | None = None,
    ) -> tuple[dict, dict]:
        """Run inference and compute all 11 metrics.

        Parameters
        ----------
        data_loader : DataLoader
            Data loader for evaluation.

        Returns
        -------
        tuple[dict, dict]
            (metrics_dict, predictions_dict) where predictions_dict has keys
            "pred_mean", "pred_std", "targets", and "mask".
        """
        self.model.eval()

        all_pred_mean: list[Tensor] = []
        all_pred_std: list[Tensor] = []
        all_targets: list[Tensor] = []
        all_mask: list[Tensor] = []
        all_population: list[Tensor] = []

        with torch.no_grad():
            for batch in data_loader:
                batch = _move_batch_to_device(batch, self.device)
                output = self.model(batch)

                all_pred_mean.append(output["pred_mean"].cpu())   # [N_l, H]
                all_pred_std.append(output["pred_std"].cpu())     # [N_l, H]
                all_targets.append(batch["targets"].cpu())        # [N_l, H]
                all_mask.append(batch["mask"].cpu())              # [N_l]
                if "population" in batch:
                    all_population.append(batch["population"].cpu())  # [N_l]

        # Concatenate across all batches: [total_N_l, H]
        pred_mean = torch.cat(all_pred_mean, dim=0)   # [total_N_l, H]
        pred_std = torch.cat(all_pred_std, dim=0)     # [total_N_l, H]
        targets = torch.cat(all_targets, dim=0)        # [total_N_l, H]
        mask = torch.cat(all_mask, dim=0)              # [total_N_l]

        # Forward population only when per-capita normalization is active so
        # that COVID/AIV (which take the default false branch) produce
        # bitwise-identical metrics to pre-change runs.
        population: Tensor | None = None
        if self.per_capita_normalize and all_population:
            population = torch.cat(all_population, dim=0)  # [total_N_l]

        metrics = compute_all_metrics(
            pred_mean, pred_std, targets, mask,
            self.cfg, self.outbreak_threshold, None,
            population=population,
            per_capita_base=self.per_capita_base,
            metric_names=metric_names,
        )

        # Per-horizon breakdown (h1 / h2 / h4 / all). Kept as a nested dict
        # under a dedicated key so legacy consumers of the flat metrics dict
        # are not affected.
        try:
            per_horizon = compute_per_horizon_metrics(
                pred_mean, pred_std, targets, mask,
                self.cfg, self.outbreak_threshold, None,
                population=population,
                per_capita_base=self.per_capita_base,
                metric_names=per_horizon_metric_names,
            )
            metrics["per_horizon"] = per_horizon
        except Exception as e:  # pragma: no cover — defensive
            logger.warning("compute_per_horizon_metrics failed: %s", e)

        predictions = {
            "pred_mean": pred_mean,
            "pred_std": pred_std,
            "targets": targets,
            "mask": mask,
        }
        if population is not None:
            predictions["population"] = population

        return metrics, predictions

    # ------------------------------------------------------------------
    # Private: training helpers
    # ------------------------------------------------------------------

    def _train_one_epoch(self, train_loader: DataLoader) -> dict:
        """Train for a single epoch with gradient accumulation.

        Parameters
        ----------
        train_loader : DataLoader
            Training data loader.

        Returns
        -------
        dict
            Per-step averages for ``loss``, ``loss_pred``, ``loss_decomp``.
        """
        self.model.train()
        grad_accum_steps = self.cfg["training"]["grad_accum_steps"]
        max_grad_norm = self.cfg["training"]["max_grad_norm"]

        running = {
            "loss": 0.0,
            "loss_pred": 0.0,
            "loss_decomp": 0.0,
        }
        num_steps = 0

        self.optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            batch = _move_batch_to_device(batch, self.device)

            # Forward pass
            model_output = self.model(batch)
            loss_dict = self.loss_fn(model_output, batch["targets"], batch["mask"])

            # NaN detection
            if torch.isnan(loss_dict["loss"]):
                raise RuntimeError(
                    f"NaN loss detected at step {step + 1} "
                    f"(loss_pred={loss_dict['loss_pred'].item()})"
                )

            # Scale loss for gradient accumulation
            scaled_loss = loss_dict["loss"] / grad_accum_steps  # scalar
            scaled_loss.backward()

            # Accumulate running totals (unscaled)
            running["loss"] += loss_dict["loss"].item()
            running["loss_pred"] += loss_dict["loss_pred"].item()
            running["loss_decomp"] += loss_dict.get(
                "loss_decomp", torch.tensor(0.0)
            ).item()
            num_steps += 1

            # Optimizer step at accumulation boundary or end of epoch
            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                clip_grad_norm_(self.model.parameters(), max_norm=max_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()

        # Average over steps
        avg = {k: v / max(num_steps, 1) for k, v in running.items()}
        return avg

    def _validate(self, val_loader: DataLoader) -> tuple[dict, float]:
        """Run validation and compute loss + monitored point metric.

        Parameters
        ----------
        val_loader : DataLoader
            Validation data loader.

        Returns
        -------
        tuple[dict, float]
            (loss_dict with averaged losses, monitored validation metric scalar)
        """
        self.model.eval()

        running = {"loss": 0.0, "loss_pred": 0.0, "loss_decomp": 0.0}
        num_steps = 0

        all_pred_mean: list[Tensor] = []
        all_pred_std: list[Tensor] = []
        all_targets: list[Tensor] = []
        all_mask: list[Tensor] = []
        all_population: list[Tensor] = []

        with torch.no_grad():
            for batch in val_loader:
                batch = _move_batch_to_device(batch, self.device)

                model_output = self.model(batch)
                loss_dict = self.loss_fn(model_output, batch["targets"], batch["mask"])

                running["loss"] += loss_dict["loss"].item()
                running["loss_pred"] += loss_dict["loss_pred"].item()
                running["loss_decomp"] += loss_dict.get(
                    "loss_decomp", torch.tensor(0.0)
                ).item()
                num_steps += 1

                all_pred_mean.append(model_output["pred_mean"].cpu())  # [N_l, H]
                all_pred_std.append(model_output["pred_std"].cpu())    # [N_l, H]
                all_targets.append(batch["targets"].cpu())             # [N_l, H]
                all_mask.append(batch["mask"].cpu())                   # [N_l]
                if "population" in batch:
                    all_population.append(batch["population"].cpu())   # [N_l]

        avg_losses = {k: v / max(num_steps, 1) for k, v in running.items()}

        # Compute CRPS for early stopping
        pred_mean = torch.cat(all_pred_mean, dim=0)   # [total_N_l, H]
        pred_std = torch.cat(all_pred_std, dim=0)     # [total_N_l, H]
        targets = torch.cat(all_targets, dim=0)        # [total_N_l, H]
        mask = torch.cat(all_mask, dim=0)              # [total_N_l]

        # Forward population only when per-capita normalization is active
        population: Tensor | None = None
        if self.per_capita_normalize and all_population:
            population = torch.cat(all_population, dim=0)  # [total_N_l]

        metrics = compute_all_metrics(
            pred_mean, pred_std, targets, mask,
            self.cfg, self.outbreak_threshold, None,
            population=population,
            per_capita_base=self.per_capita_base,
        )
        val_metric = metrics[self.monitor_metric]

        return avg_losses, val_metric

    def _compute_val_metrics(self, val_loader: DataLoader) -> dict:
        """Compute full validation metrics (wrapper around evaluate)."""
        metrics, _ = self.evaluate(val_loader)
        return metrics

    # ------------------------------------------------------------------
    # Private: checkpoint helpers
    # ------------------------------------------------------------------

    def _save_checkpoint(self, path: str, epoch: int, val_metric: float) -> None:
        """Save a training checkpoint.

        Parameters
        ----------
        path : str
            File path to save the checkpoint.
        epoch : int
            Current epoch number.
        val_metric : float
            Validation value of the configured monitor metric at this checkpoint.
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "epoch": epoch,
            "best_val_metric": self.best_val_metric,
            "outbreak_threshold": self.outbreak_threshold,
            "cfg": self.cfg,
        }
        torch.save(checkpoint, path)
        logger.debug("Checkpoint saved to %s (epoch=%d, val_%s=%.4f)", path, epoch, self.monitor_metric, val_metric)

    def _load_checkpoint(self, path: str) -> dict:
        """Load a training checkpoint.

        Parameters
        ----------
        path : str
            File path to the checkpoint.

        Returns
        -------
        dict
            The full checkpoint dictionary.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best_val_metric = checkpoint.get("best_val_metric", checkpoint.get("best_val_crps", float("inf")))
        self.outbreak_threshold = checkpoint["outbreak_threshold"]
        logger.info(
            "Checkpoint loaded from %s (epoch=%d, best_val_%s=%.4f)",
            path, checkpoint["epoch"], self.monitor_metric, self.best_val_metric,
        )
        return checkpoint

    # ------------------------------------------------------------------
    # Private: utility helpers
    # ------------------------------------------------------------------

    def _compute_outbreak_threshold(self, train_loader: DataLoader) -> None:
        """Compute outbreak threshold from non-zero training targets.

        Uses the configured percentile (cfg["evaluation"]["outbreak_percentile"])
        over non-zero training target values only. This avoids degenerate
        threshold=0 in sparse datasets (e.g. AIV where ~95% of targets are zero),
        producing a meaningful outbreak vs non-outbreak distinction.
        """
        all_targets: list[Tensor] = []
        for batch in train_loader:
            all_targets.append(batch["targets"])  # [N_l, H]

        train_targets = torch.cat(all_targets, dim=0).numpy().ravel()  # [total]
        nonzero_targets = train_targets[train_targets > 0]
        percentile = self.cfg["evaluation"]["outbreak_percentile"]

        if len(nonzero_targets) == 0:
            # Fallback: all targets are zero, use 0.0 (metrics will be nan)
            self.outbreak_threshold = 0.0
            logger.warning(
                "All training targets are zero. "
                "Outbreak threshold set to 0.0000 (outbreak metrics will be nan)."
            )
        else:
            self.outbreak_threshold = float(
                np.percentile(nonzero_targets, percentile)
            )
            logger.info(
                "Outbreak threshold set to %.4f "
                "(percentile=%d of %d non-zero targets, %.1f%% of total)",
                self.outbreak_threshold, percentile,
                len(nonzero_targets),
                len(nonzero_targets) / len(train_targets) * 100,
            )

    @staticmethod
    def _infer_dataset_name(data_loader: DataLoader) -> str:
        """Infer dataset name from the first batch metadata."""
        for batch in data_loader:
            return batch["metadata"]["dataset_name"]
        return "unknown"
