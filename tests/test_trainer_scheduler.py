"""Tests for Trainer learning-rate scheduler edge cases."""

from __future__ import annotations

import torch

from src.training.trainer import Trainer


def _make_cfg(max_epochs: int, warmup_epochs: int) -> dict:
    return {
        "data": {
            "per_capita_normalize": False,
            "per_capita_base": 100_000,
        },
        "model": {
            "fusion_horizon": {"head": {"decomp_loss_weight": 1.0e-4}},
        },
        "training": {
            "lr": 1.0e-3,
            "min_lr": 1.0e-5,
            "weight_decay": 1.0e-4,
            "warmup_epochs": warmup_epochs,
            "max_epochs": max_epochs,
            "monitor_metric": "RMSE",
            "checkpoint_dir": "checkpoints",
        },
    }


def test_trainer_scheduler_warmup_only_when_max_epochs_equals_warmup() -> None:
    model = torch.nn.Linear(1, 1)
    trainer = Trainer(
        model,
        _make_cfg(max_epochs=5, warmup_epochs=5),
        torch.device("cpu"),
    )

    for _ in range(5):
        trainer.optimizer.step()
        trainer.scheduler.step()


def test_trainer_scheduler_warmup_only_when_max_epochs_below_warmup() -> None:
    model = torch.nn.Linear(1, 1)
    trainer = Trainer(
        model,
        _make_cfg(max_epochs=3, warmup_epochs=5),
        torch.device("cpu"),
    )

    for _ in range(3):
        trainer.optimizer.step()
        trainer.scheduler.step()
