"""Unit tests for PersistenceAnchor (PR2, R5)."""

from __future__ import annotations

import pytest
import torch

from src.models.fusion_horizon import PersistenceAnchor


def test_last_value_init_reproduces_persistence():
    """With init_mode='last_value', baseline equals y_hist[:, -1] for every h."""
    horizons = [1, 2, 4]
    K = 8
    anchor = PersistenceAnchor(horizons, K=K, init_mode="last_value")
    anchor.eval()
    y_hist = torch.randn(5, K)
    out = anchor(y_hist)  # [5, H]
    assert out.shape == (5, len(horizons))
    for h in range(len(horizons)):
        assert torch.allclose(out[:, h], y_hist[:, -1], atol=1e-5)


def test_last_plus_trend_constant_series():
    """Constant series → baseline = that constant for every horizon."""
    horizons = [1, 2, 4]
    K = 8
    anchor = PersistenceAnchor(horizons, K=K, init_mode="last_plus_trend")
    anchor.eval()
    # Constant series: linear fit should be y = c, extrapolation = c.
    y_hist = torch.full((3, K), 5.0)
    out = anchor(y_hist)
    assert torch.allclose(out, torch.full_like(out, 5.0), atol=1e-4)


def test_last_plus_trend_linear_series():
    """Linear series y=2*k+3 → forecast at K-1+h is 2*(K-1+h)+3."""
    horizons = [1, 2, 4]
    K = 8
    anchor = PersistenceAnchor(horizons, K=K, init_mode="last_plus_trend")
    anchor.eval()
    x = torch.arange(K, dtype=torch.float32)
    y = (2.0 * x + 3.0).unsqueeze(0)  # [1, K]
    out = anchor(y)
    expected = torch.tensor(
        [[2.0 * (K - 1 + h) + 3.0 for h in horizons]], dtype=torch.float32
    )
    assert torch.allclose(out, expected, atol=1e-3)


def test_invalid_init_mode_raises():
    with pytest.raises(ValueError):
        PersistenceAnchor([1, 2], K=4, init_mode="bogus")


def test_gradient_flow():
    horizons = [1, 2]
    K = 4
    anchor = PersistenceAnchor(horizons, K=K)
    y_hist = torch.randn(3, K, requires_grad=True)
    out = anchor(y_hist)
    out.sum().backward()
    assert y_hist.grad is not None and torch.isfinite(y_hist.grad).all()
