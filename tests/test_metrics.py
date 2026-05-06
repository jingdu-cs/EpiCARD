"""Tests for point-only evaluation metrics."""

import math

import pytest
import torch

from src.evaluation.metrics import (
    compute_all_metrics,
    compute_ood_calibration_profile,
    compute_per_horizon_metrics,
)


def _make_cfg():
    return {"model": {}}


EXPECTED_KEYS = {
    "MAE", "RMSE", "MAPE", "sMAPE", "PearsonR", "SpearmanR",
    "OutbreakAUROC", "OutbreakAUPRC", "CRPS", "Coverage50", "Coverage90",
}


def test_all_metrics_computed():
    pred_mean = torch.tensor([[1.0, 2.0, 3.0]])
    pred_std = torch.tensor([[1.0, 1.0, 1.0]])
    targets = torch.tensor([[1.0, 2.0, 3.0]])
    mask = torch.tensor([1.0])

    result = compute_all_metrics(pred_mean, pred_std, targets, mask, _make_cfg(), 50.0)
    assert set(result.keys()) == EXPECTED_KEYS


def test_mae_rmse_known_values():
    cfg = _make_cfg()
    mask = torch.tensor([1.0])

    r = compute_all_metrics(
        torch.tensor([[1.0, 2.0, 3.0]]),
        torch.ones(1, 3),
        torch.tensor([[2.0, 3.0, 4.0]]),
        mask,
        cfg,
        50.0,
    )
    assert r["MAE"] == pytest.approx(1.0)
    assert r["RMSE"] == pytest.approx(1.0)


def test_crps_is_abs_error_in_point_mode():
    r = compute_all_metrics(
        torch.tensor([[2.0]]),
        torch.tensor([[1.0]]),
        torch.tensor([[5.0]]),
        torch.tensor([1.0]),
        _make_cfg(),
        50.0,
    )
    assert r["CRPS"] == pytest.approx(3.0, abs=1e-6)
    assert math.isnan(r["Coverage50"])
    assert math.isnan(r["Coverage90"])


def test_outbreak_and_correlations():
    pred_mean = torch.tensor([[0.0], [0.0], [0.0], [100.0]])
    pred_std = torch.ones(4, 1)
    targets = torch.tensor([[0.0], [0.0], [0.0], [100.0]])
    mask = torch.ones(4)

    r = compute_all_metrics(pred_mean, pred_std, targets, mask, _make_cfg(), 50.0)
    assert r["OutbreakAUROC"] == pytest.approx(1.0)
    assert r["OutbreakAUPRC"] == pytest.approx(1.0)

    vals = torch.arange(1.0, 101.0).unsqueeze(1)
    r2 = compute_all_metrics(vals, torch.ones_like(vals), vals, torch.ones(100), _make_cfg(), 50.0)
    assert r2["PearsonR"] == pytest.approx(1.0, abs=0.02)
    assert r2["SpearmanR"] == pytest.approx(1.0, abs=1e-5)


def test_mask_handling():
    pred_mean = torch.tensor([[5.0, 5.0, 5.0], [999.0, 999.0, 999.0]])
    pred_std = torch.ones_like(pred_mean)
    targets = torch.tensor([[5.0, 5.0, 5.0], [0.0, 0.0, 0.0]])
    mask = torch.tensor([1.0, 0.0])
    r = compute_all_metrics(pred_mean, pred_std, targets, mask, _make_cfg(), 50.0)
    assert r["MAE"] == pytest.approx(0.0)


def test_per_horizon_metrics():
    n = 5
    pred_mean = torch.randn(n, 3)
    pred_std = torch.ones(n, 3)
    targets = torch.randn(n, 3)
    mask = torch.ones(n)

    result = compute_per_horizon_metrics(pred_mean, pred_std, targets, mask, _make_cfg(), 50.0)
    assert set(result.keys()) == {"h1", "h2", "h4", "all"}
    for key in result:
        assert set(result[key].keys()) == EXPECTED_KEYS


def test_ood_calibration_profile_not_applicable():
    pred_mean = torch.tensor([[1.0, 2.0, 3.0]])
    pred_std = torch.tensor([[1.0, 1.0, 1.0]])
    targets = torch.tensor([[1.5, 2.5, 3.5]])
    mask = torch.tensor([1.0])

    profile = compute_ood_calibration_profile(
        pred_mean, pred_std, targets, mask, _make_cfg(),
    )

    assert profile["applicable"] is False
    assert profile["distribution"] == "point"
    assert "not applicable" in profile["reason"]
    assert profile["overall"] is None
    assert profile["per_horizon"] == {}
    assert profile["by_target_quantile"] == {}
