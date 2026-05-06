"""Unit tests for HorizonCrossAttentionFusion (PR2, R5)."""

from __future__ import annotations

import pytest
import torch

from src.models.fusion_horizon import HorizonCrossAttentionFusion


def _cfg(D: int = 16, num_heads: int = 1, dropout: float = 0.0) -> dict:
    return {
        "model": {
            "hidden_dim": D,
            "fusion_horizon": {
                "cross_attn": {"num_heads": num_heads, "dropout": dropout},
            },
        },
    }


def test_output_shape_single_head():
    D, H, N_l, N_c = 16, 3, 4, 7
    mod = HorizonCrossAttentionFusion(_cfg(D, num_heads=1), num_horizons=H)
    s = torch.randn(N_l, D)
    tokens = torch.randn(N_c, D)
    batch = torch.tensor([0, 0, 1, 1, 2, 3, 3], dtype=torch.long)
    out = mod(s, tokens, batch)
    assert out.shape == (N_l, H, D)
    assert torch.isfinite(out).all()


def test_output_shape_multi_head():
    D, H, N_l, N_c = 32, 2, 3, 5
    mod = HorizonCrossAttentionFusion(_cfg(D, num_heads=4), num_horizons=H)
    s = torch.randn(N_l, D)
    tokens = torch.randn(N_c, D)
    batch = torch.tensor([0, 0, 1, 2, 2], dtype=torch.long)
    out = mod(s, tokens, batch)
    assert out.shape == (N_l, H, D)


def test_empty_case_set():
    D, H, N_l = 16, 3, 4
    mod = HorizonCrossAttentionFusion(_cfg(D), num_horizons=H)
    mod.eval()
    s = torch.randn(N_l, D)
    tokens = torch.zeros(0, D)
    batch = torch.zeros(0, dtype=torch.long)
    out = mod(s, tokens, batch)
    assert out.shape == (N_l, H, D)
    # With no cases, every horizon slice should equal LN(s), identical across H.
    for h in range(1, H):
        assert torch.allclose(out[:, 0], out[:, h], atol=1e-6)


def test_empty_location_mask():
    """Locations with no cases produce c=0 → LN(s) broadcast across H."""
    D, H, N_l = 16, 3, 4
    mod = HorizonCrossAttentionFusion(_cfg(D), num_horizons=H)
    mod.eval()
    s = torch.randn(N_l, D)
    # Only locations 0 and 2 have cases; 1 and 3 are empty.
    tokens = torch.randn(3, D)
    batch = torch.tensor([0, 0, 2], dtype=torch.long)
    out = mod(s, tokens, batch)
    # Empty locations → horizon slices should all equal LN(s_l)
    for loc in (1, 3):
        for h in range(1, H):
            assert torch.allclose(out[loc, 0], out[loc, h], atol=1e-6)


def test_gradient_flow():
    D, H, N_l, N_c = 16, 3, 4, 6
    mod = HorizonCrossAttentionFusion(_cfg(D), num_horizons=H)
    s = torch.randn(N_l, D, requires_grad=True)
    tokens = torch.randn(N_c, D, requires_grad=True)
    batch = torch.tensor([0, 0, 1, 2, 3, 3], dtype=torch.long)
    out = mod(s, tokens, batch)
    out.sum().backward()
    assert s.grad is not None and torch.isfinite(s.grad).all()
    assert tokens.grad is not None and torch.isfinite(tokens.grad).all()


def test_softmax_sums_to_one_per_location():
    """Attention weights α per (location, horizon, head) must sum to 1."""
    from src.models.fusion_horizon import _scatter_softmax_nd

    scores = torch.randn(10, 3, 2)  # [N_c, H, nh]
    index = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2, 3], dtype=torch.long)
    alpha = _scatter_softmax_nd(scores, index, dim_size=4)
    # Per group sum to 1 on each (H, nh) slice
    for g in range(4):
        mask = index == g
        if mask.any():
            sums = alpha[mask].sum(dim=0)  # [H, nh]
            assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
