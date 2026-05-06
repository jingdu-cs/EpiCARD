"""Point-only contract tests for ``MoEHorizonHead``."""

from __future__ import annotations

import torch

from src.models.fusion_horizon import MoEHorizonHead


def _cfg(D: int = 16, K: int = 8) -> dict:
    return {
        "model": {
            "hidden_dim": D,
            "fusion_horizon": {
                "moe": {
                    "enabled": True,
                    "expert_hidden_dim": None,
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
                    "init_mode": "last_plus_trend",
                },
            },
        },
    }


def test_forward_returns_point_only_contract() -> None:
    D, H, N_l, K = 16, 3, 4, 8
    mod = MoEHorizonHead(_cfg(D=D, K=K), num_horizons=H, horizons=[1, 2, 4])
    s_h = torch.randn(N_l, H, D)
    y_hist = torch.rand(N_l, K) * 10.0

    pred_mean, pred_std, pred_log1p = mod(s_h, y_hist)

    assert pred_mean.shape == (N_l, H)
    assert pred_std.shape == (N_l, H)
    assert pred_log1p.shape == (N_l, H)
    assert torch.isfinite(pred_mean).all()
    assert torch.isfinite(pred_std).all()
    assert torch.isfinite(pred_log1p).all()
    assert (pred_mean >= 0.0).all()
    assert torch.equal(pred_std, torch.ones_like(pred_mean))
