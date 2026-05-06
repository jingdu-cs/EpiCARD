"""Unit tests for the point-only ``MoEHorizonHead``."""

from __future__ import annotations

import pytest
import torch

from src.models.fusion_horizon import MoEHorizonHead


def _cfg(
    D: int = 16,
    K: int = 8,
    init_mode: str = "last_plus_trend",
    D_ff: int | None = None,
) -> dict:
    return {
        "model": {
            "hidden_dim": D,
            "fusion_horizon": {
                "moe": {
                    "enabled": True,
                    "expert_hidden_dim": D_ff,
                    "expert_dropout": 0.0,
                    "shared_expert": True,
                    "per_horizon_expert": True,
                    "adaptive_mixing": {
                        "enabled": True,
                        "mode": "horizon_scalar",
                        "init_shared_logit": 1.0,
                    },
                },
                "persistence": {
                    "enabled": True,
                    "history_window": K,
                    "init_mode": init_mode,
                },
            },
        },
    }


def test_shapes_full():
    D, H, N_l, K = 16, 3, 5, 8
    mod = MoEHorizonHead(_cfg(D=D, K=K), num_horizons=H, horizons=[1, 2, 4])
    s_h = torch.randn(N_l, H, D)
    y_hist = torch.randn(N_l, K)
    pred_mean, pred_std, pred_log1p = mod(s_h, y_hist)
    assert pred_mean.shape == (N_l, H)
    assert pred_std.shape == (N_l, H)
    assert pred_log1p.shape == (N_l, H)
    assert (pred_std > 0).all()


def test_zero_init_pred_log1p_equals_baseline():
    """At init, zero point heads recover the persistence baseline exactly."""
    D, H, N_l, K = 16, 3, 4, 8
    horizons = [1, 2, 4]
    mod = MoEHorizonHead(
        _cfg(D=D, K=K, init_mode="last_value"),
        num_horizons=H,
        horizons=horizons,
    )
    mod.eval()
    s_h = torch.randn(N_l, H, D)
    y_hist = torch.rand(N_l, K) + 0.1  # positive per-capita-like values
    _, _, pred_log1p = mod(s_h, y_hist)
    expected = torch.log1p(y_hist[:, -1].clamp_min(0.0))
    for h in range(H):
        assert torch.allclose(pred_log1p[:, h], expected, atol=1e-5)


def test_persistence_missing_y_hist_raises():
    D, H, K = 16, 3, 8
    mod = MoEHorizonHead(_cfg(D=D, K=K), num_horizons=H, horizons=[1, 2, 4])
    with pytest.raises(ValueError):
        mod(torch.randn(2, H, D), y_hist=None)


def test_gradient_flow():
    D, H, N_l, K = 16, 3, 4, 8
    mod = MoEHorizonHead(_cfg(D=D, K=K), num_horizons=H, horizons=[1, 2, 4])
    s_h = torch.randn(N_l, H, D, requires_grad=True)
    y_hist = torch.randn(N_l, K, requires_grad=True)
    pred_mean, pred_std, pred_log1p = mod(s_h, y_hist)
    (pred_mean.sum() + pred_std.sum() + pred_log1p.sum()).backward()
    assert s_h.grad is not None and torch.isfinite(s_h.grad).all()
    assert y_hist.grad is not None and torch.isfinite(y_hist.grad).all()


def test_d_ff_2d_variant():
    """expert_hidden_dim = 2D widens fc1 in the shared expert."""
    D, H, K = 16, 3, 8
    mod = MoEHorizonHead(
        _cfg(D=D, K=K, D_ff=2 * D),
        num_horizons=H,
        horizons=[1, 2, 4],
    )
    assert mod.shared_expert.fc1.out_features == 2 * D
    s_h = torch.randn(2, H, D)
    y_hist = torch.randn(2, K)
    pred_mean, pred_std, pred_log1p = mod(s_h, y_hist)
    assert pred_mean.shape == (2, H)
    assert pred_std.shape == (2, H)
    assert pred_log1p.shape == (2, H)


def test_adaptive_mixing_gate_controls_shared_vs_horizon_balance():
    D, H, K = 4, 3, 8
    mod = MoEHorizonHead(_cfg(D=D, K=K), num_horizons=H, horizons=[1, 2, 4])
    s_h = torch.randn(2, H, D)
    shared = torch.full_like(s_h, 2.0)
    horizon = torch.full_like(s_h, -3.0)

    class _ConstShared(torch.nn.Module):
        def __init__(self, out: torch.Tensor) -> None:
            super().__init__()
            self.register_buffer("out", out)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.out.reshape(-1, self.out.shape[-1])

    class _ConstH(torch.nn.Module):
        def __init__(self, out: torch.Tensor) -> None:
            super().__init__()
            self.register_buffer("out", out)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.out

    mod.shared_expert = _ConstShared(shared)
    mod.horizon_experts = torch.nn.ModuleList(
        [_ConstH(horizon[:, h, :]) for h in range(H)]
    )

    with torch.no_grad():
        mod.shared_mix_logits.fill_(12.0)
    shared_heavy = mod._mix_horizon_features(s_h)
    assert torch.allclose(shared_heavy, shared, atol=1e-4)

    with torch.no_grad():
        mod.shared_mix_logits.fill_(-12.0)
    horizon_heavy = mod._mix_horizon_features(s_h)
    assert torch.allclose(horizon_heavy, horizon, atol=1e-4)
