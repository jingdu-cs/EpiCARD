"""Tests for point-only training losses."""

from __future__ import annotations

import math
import os
import sys
import importlib.util

import pytest
import torch

_spec = importlib.util.spec_from_file_location(
    "src.training.losses",
    os.path.join(os.path.dirname(__file__), "..", "src", "training", "losses.py"),
)
_losses = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _losses
_spec.loader.exec_module(_losses)

CombinedLoss = _losses.CombinedLoss


def _make_cfg() -> dict:
    return {
        "model": {
            "fusion_horizon": {
                "head": {
                    "decomp_loss_weight": 1.0e-4,
                }
            },
        }
    }


def test_combined_loss_forward() -> None:
    cfg = _make_cfg()
    criterion = CombinedLoss(cfg)

    pred_log1p = torch.tensor([[0.0, 0.5]], requires_grad=True)
    pred_mean = torch.expm1(pred_log1p.detach()).clamp_min(0.0)
    pred_std = torch.ones_like(pred_mean)
    targets = torch.tensor([[0.0, math.e - 1.0]])
    mask = torch.tensor([1.0])

    output = criterion(
        {
            "pred_mean": pred_mean,
            "pred_std": pred_std,
            "pred_log1p": pred_log1p,
        },
        targets,
        mask,
    )

    assert set(output.keys()) == {"loss", "loss_pred", "loss_decomp"}
    for value in output.values():
        assert isinstance(value, torch.Tensor)
        assert value.dim() == 0


def test_combined_loss_point_mode_uses_log1p_huber() -> None:
    cfg = _make_cfg()
    criterion = CombinedLoss(cfg)

    pred_log1p = torch.tensor([[0.0, 0.8]], requires_grad=True)
    pred_mean = torch.expm1(pred_log1p.detach()).clamp_min(0.0)
    pred_std = torch.ones_like(pred_mean)
    targets = torch.tensor([[0.0, math.e - 1.0]])
    mask = torch.tensor([1.0])

    output = criterion(
        {
            "pred_mean": pred_mean,
            "pred_std": pred_std,
            "pred_log1p": pred_log1p,
        },
        targets,
        mask,
    )

    assert torch.isfinite(output["loss"])
    output["loss"].backward()
    assert pred_log1p.grad is not None
    assert not torch.all(pred_log1p.grad == 0.0)


def test_mask_zeros_out_invalid() -> None:
    cfg = _make_cfg()
    criterion = CombinedLoss(cfg)

    pred_log1p = torch.log1p(torch.tensor([[1.0, 2.0], [100.0, 100.0]]))
    pred_mean = torch.expm1(pred_log1p).clamp_min(0.0)
    pred_std = torch.ones_like(pred_mean)
    targets = torch.tensor([[1.0, 2.0], [0.0, 0.0]])
    mask_partial = torch.tensor([1.0, 0.0])
    mask_single = torch.tensor([1.0])

    output_partial = criterion(
        {
            "pred_mean": pred_mean,
            "pred_std": pred_std,
            "pred_log1p": pred_log1p,
        },
        targets,
        mask_partial,
    )
    output_single = criterion(
        {
            "pred_mean": pred_mean[:1],
            "pred_std": pred_std[:1],
            "pred_log1p": pred_log1p[:1],
        },
        targets[:1],
        mask_single,
    )
    assert output_partial["loss_pred"].item() == pytest.approx(
        output_single["loss_pred"].item(), abs=1e-6
    )


def test_combined_loss_shared_private_decomp_penalty() -> None:
    cfg = _make_cfg()
    cfg["model"]["fusion_horizon"]["head"]["decomp_loss_weight"] = 1.0e-3
    criterion = CombinedLoss(cfg)

    pred_log1p = torch.tensor([[0.5, 1.0]], requires_grad=True)
    pred_mean = torch.expm1(pred_log1p.detach()).clamp_min(0.0)
    pred_std = torch.ones_like(pred_mean)
    targets = torch.tensor([[1.0, 2.0]])
    mask = torch.tensor([1.0])
    case_shared = torch.tensor([[1.0, 0.0]], requires_grad=True)
    case_private = torch.tensor([[1.0, 0.0]], requires_grad=True)
    loc_shared = torch.tensor([[0.0, 1.0]], requires_grad=True)
    loc_private = torch.tensor([[0.0, 1.0]], requires_grad=True)

    output = criterion(
        {
            "pred_mean": pred_mean,
            "pred_std": pred_std,
            "pred_log1p": pred_log1p,
            "fusion_shared_case": case_shared,
            "fusion_private_case": case_private,
            "fusion_shared_loc": loc_shared,
            "fusion_private_loc": loc_private,
        },
        targets,
        mask,
    )

    assert output["loss_decomp"].item() == pytest.approx(2.0, abs=1e-6)
    assert output["loss"].item() > output["loss_pred"].item()
    output["loss"].backward()
    assert case_shared.grad is not None
    assert loc_shared.grad is not None
