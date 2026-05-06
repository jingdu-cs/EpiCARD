"""Fusion-horizon modules for the point-only forecasting mainline.

- ``HorizonCrossAttentionFusion``: horizon-conditioned cross-attention
  over per-case LLM tokens.
- ``Expert2LayerMLP``: pre-LN 2-layer MLP expert.
- ``MoEHorizonHead``: shared + per-horizon MoE point head with adaptive
  per-horizon mixing and a persistence anchor baseline.
- ``PersistenceAnchor``: ``baseline_h = Linear_bl_h(y_hist)`` with
  closed-form "last + linear trend" initialization.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _scatter_softmax_nd(
    scores: torch.Tensor, index: torch.Tensor, dim_size: int
) -> torch.Tensor:
    """Softmax over the first axis within groups defined by ``index``.

    Parameters
    ----------
    scores : Tensor[N, ...]
        Scores to softmax. Any trailing dims are broadcast over groups.
    index : Tensor[N]
        Group id in ``[0, dim_size)`` per row.
    dim_size : int
        Number of groups.

    Returns
    -------
    Tensor[N, ...] — same shape as ``scores``.
    """
    N = scores.shape[0]
    extra = scores.shape[1:]
    flat = scores.reshape(N, -1)           # [N, P]
    P = flat.shape[1]

    idx = index.unsqueeze(-1).expand(N, P)  # [N, P]
    # Group max for numerical stability
    group_max = torch.full(
        (dim_size, P), float("-inf"), device=scores.device, dtype=scores.dtype
    )
    group_max = group_max.scatter_reduce(
        0, idx, flat, reduce="amax", include_self=True
    )
    # Replace -inf with 0 for empty groups (gathered rows won't be used anyway)
    group_max = torch.where(
        torch.isfinite(group_max), group_max, torch.zeros_like(group_max)
    )
    shifted = flat - group_max[index]       # [N, P]
    exp = shifted.exp()

    group_sum = torch.zeros(
        dim_size, P, device=scores.device, dtype=scores.dtype
    )
    group_sum.scatter_add_(0, idx, exp)
    denom = group_sum[index].clamp(min=1e-12)  # [N, P]
    alpha = exp / denom                     # [N, P]
    return alpha.reshape(N, *extra)


def _scatter_add_nd(
    values: torch.Tensor, index: torch.Tensor, dim_size: int
) -> torch.Tensor:
    """Scatter-add ``values`` along dim 0 into an output of size ``dim_size``.

    Parameters
    ----------
    values : Tensor[N, ...]
    index  : Tensor[N]
    dim_size : int

    Returns
    -------
    Tensor[dim_size, ...]
    """
    N = values.shape[0]
    extra = values.shape[1:]
    flat = values.reshape(N, -1)  # [N, P]
    P = flat.shape[1]
    out = torch.zeros(dim_size, P, device=values.device, dtype=values.dtype)
    out.scatter_add_(0, index.unsqueeze(-1).expand(N, P), flat)
    return out.reshape(dim_size, *extra)


# ---------------------------------------------------------------------------
# HorizonCrossAttentionFusion (Step 2)
# ---------------------------------------------------------------------------

class HorizonCrossAttentionFusion(nn.Module):
    """Horizon-conditioned cross-attention over per-case LLM tokens.

    For each location ``l`` and horizon ``h``:
        q_{l,h} = W_q · [s_l ; H_h]
        K, V    = W_k(E_cl), W_v(E_cl)      (per-case LLM tokens)
        c_{l,h} = softmax(q · K^T / sqrt(d_h)) · V     (masked to cases of l)
        s_{l,h} = LayerNorm(s_l + W_o(c_{l,h}))

    Locations with no cases receive ``c_{l,h} = 0`` → output is ``LN(s_l)``.
    """

    def __init__(self, cfg: Dict[str, Any], num_horizons: int) -> None:
        super().__init__()
        D: int = cfg["model"]["hidden_dim"]
        ca_cfg = cfg["model"]["fusion_horizon"]["cross_attn"]
        num_heads: int = int(ca_cfg.get("num_heads", 1))
        dropout: float = float(ca_cfg.get("dropout", 0.1))
        if D % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({D}) must be divisible by num_heads ({num_heads})"
            )

        self.D = D
        self.H = num_horizons
        self.num_heads = num_heads
        self.head_dim = D // num_heads

        self.horizon_emb = nn.Embedding(num_horizons, D)
        self.W_q = nn.Linear(2 * D, D)
        self.W_k = nn.Linear(D, D)
        self.W_v = nn.Linear(D, D)
        self.W_o = nn.Linear(D, D)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(D)

    def forward(
        self,
        s: torch.Tensor,
        case_tokens: torch.Tensor,
        case_batch: torch.Tensor,
    ) -> torch.Tensor:
        """Compute horizon-conditioned cross-attention output.

        Parameters
        ----------
        s : Tensor[N_l, D]
            Base fused state from ``SharedPrivateFusionHead``.
        case_tokens : Tensor[N_c, D]
            Per-case tokens (e.g. LLM projections from ``CaseEventBranch``).
        case_batch : Tensor[N_c]
            Case-to-location mapping, values in ``[0, N_l)``.

        Returns
        -------
        s_h : Tensor[N_l, H, D]
            Horizon-conditioned fused state.
        """
        N_l, D = s.shape
        H = self.H
        N_c = case_tokens.shape[0]
        device = s.device
        nh, hd = self.num_heads, self.head_dim

        # Build horizon-conditioned queries: [N_l, H, D]
        horizon_ids = torch.arange(H, device=device)
        H_emb = self.horizon_emb(horizon_ids)                       # [H, D]
        s_exp = s.unsqueeze(1).expand(N_l, H, D)                    # [N_l, H, D]
        H_exp = H_emb.unsqueeze(0).expand(N_l, H, D)                # [N_l, H, D]
        q = self.W_q(torch.cat([s_exp, H_exp], dim=-1))             # [N_l, H, D]

        if N_c == 0:
            # No cases at all → c = 0; output is LN(s_l) broadcast over H.
            c_out = torch.zeros(N_l, H, D, device=device, dtype=s.dtype)
            return self.norm(s_exp + c_out)                         # [N_l, H, D]

        K_ = self.W_k(case_tokens)                                  # [N_c, D]
        V_ = self.W_v(case_tokens)                                  # [N_c, D]

        # Multi-head reshape
        q_heads = q.view(N_l, H, nh, hd)                            # [N_l, H, nh, hd]
        k_heads = K_.view(N_c, nh, hd)                              # [N_c, nh, hd]
        v_heads = V_.view(N_c, nh, hd)                              # [N_c, nh, hd]

        # Per-case scores: gather query by case location
        q_per_case = q_heads[case_batch]                            # [N_c, H, nh, hd]
        scores = (q_per_case * k_heads.unsqueeze(1)).sum(-1)        # [N_c, H, nh]
        scores = scores / (hd ** 0.5)

        # Masked softmax over cases belonging to the same location
        alpha = _scatter_softmax_nd(scores, case_batch, dim_size=N_l)  # [N_c, H, nh]
        alpha = self.attn_dropout(alpha)

        # Weighted sum of values
        weighted = alpha.unsqueeze(-1) * v_heads.unsqueeze(1)       # [N_c, H, nh, hd]
        pooled = _scatter_add_nd(weighted, case_batch, dim_size=N_l)  # [N_l, H, nh, hd]
        pooled = pooled.reshape(N_l, H, D)                          # [N_l, H, D]

        c_out = self.out_dropout(self.W_o(pooled))                  # [N_l, H, D]
        return self.norm(s_exp + c_out)                             # [N_l, H, D]


# ---------------------------------------------------------------------------
# Expert2LayerMLP (Step 3)
# ---------------------------------------------------------------------------

class Expert2LayerMLP(nn.Module):
    """Pre-LN 2-layer MLP expert: LN → Linear(D, D_ff) → GELU → Dropout → Linear(D_ff, D)."""

    def __init__(self, D: int, D_ff: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(D)
        self.fc1 = nn.Linear(D, D_ff)
        self.fc2 = nn.Linear(D_ff, D)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [*, D] → [*, D]."""
        h = self.norm(x)
        h = F.gelu(self.fc1(h))
        h = self.dropout(h)
        return self.fc2(h)


# ---------------------------------------------------------------------------
# PersistenceAnchor (Step 5)
# ---------------------------------------------------------------------------

class PersistenceAnchor(nn.Module):
    """Residual persistence anchor: ``baseline_h = Linear_bl_h(y_hist)``.

    Weight matrix is initialized analytically so the network begins training
    from a reasonable persistence baseline:

    - ``last_value``   → ``baseline_h = y_hist[-1]`` for every horizon.
    - ``last_plus_trend`` → least-squares linear fit on the K history points
      extrapolated to the horizon's future step.
    """

    def __init__(
        self,
        horizons: List[int],
        K: int,
        init_mode: str = "last_plus_trend",
    ) -> None:
        super().__init__()
        self.K = int(K)
        self.horizons = list(horizons)
        H = len(self.horizons)
        self.linear_bl = nn.Linear(self.K, H, bias=True)
        self._init_baseline(init_mode)

    def _init_baseline(self, mode: str) -> None:
        K = self.K
        H = len(self.horizons)
        with torch.no_grad():
            if mode == "last_value":
                w = torch.zeros(H, K)
                w[:, K - 1] = 1.0
            elif mode == "last_plus_trend":
                x = torch.arange(K, dtype=torch.float32)            # [K]
                X = torch.stack([torch.ones(K), x], dim=1)          # [K, 2]
                XtX_inv = torch.linalg.inv(X.T @ X)                 # [2, 2]
                proj = XtX_inv @ X.T                                # [2, K]
                w = torch.zeros(H, K)
                for h_idx, h_step in enumerate(self.horizons):
                    # Forecast point: last observed is x=K-1, future at K-1+h_step.
                    x_new = torch.tensor([1.0, float(K - 1 + h_step)])  # [2]
                    w[h_idx] = x_new @ proj                         # [K]
            else:
                raise ValueError(
                    f"Unknown persistence init_mode: {mode!r}. "
                    "Use 'last_value' or 'last_plus_trend'."
                )
            self.linear_bl.weight.copy_(w)
            self.linear_bl.bias.zero_()

    def forward(self, y_hist: torch.Tensor) -> torch.Tensor:
        """y_hist: [N_l, K] → baseline: [N_l, H]."""
        return self.linear_bl(y_hist)


# ---------------------------------------------------------------------------
# MoEHorizonHead (Steps 3–5)
# ---------------------------------------------------------------------------

class MoEHorizonHead(nn.Module):
    """Point-only MoE horizon head with shared + per-horizon experts."""

    def __init__(
        self,
        cfg: Dict[str, Any],
        num_horizons: int,
        horizons: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        model_cfg = cfg["model"]
        D: int = model_cfg["hidden_dim"]
        fh = model_cfg["fusion_horizon"]
        moe = fh.get("moe", {})
        adaptive_mixing_cfg = moe.get("adaptive_mixing", {})
        pers_cfg = fh.get("persistence", {})

        if not bool(moe.get("enabled", True)):
            raise ValueError("model.fusion_horizon.moe.enabled must be true.")
        if not bool(moe.get("shared_expert", True)):
            raise ValueError("model.fusion_horizon.moe.shared_expert must be true.")
        if not bool(moe.get("per_horizon_expert", True)):
            raise ValueError("model.fusion_horizon.moe.per_horizon_expert must be true.")
        if not bool(adaptive_mixing_cfg.get("enabled", True)):
            raise ValueError(
                "model.fusion_horizon.moe.adaptive_mixing.enabled must be true."
            )
        adaptive_mode = str(adaptive_mixing_cfg.get("mode", "horizon_scalar"))
        if adaptive_mode != "horizon_scalar":
            raise ValueError(
                "fusion_horizon.moe.adaptive_mixing.mode must be "
                f"'horizon_scalar', got {adaptive_mode!r}."
            )

        D_ff_raw = moe.get("expert_hidden_dim", None)
        D_ff: int = D if D_ff_raw is None else int(D_ff_raw)
        dropout: float = float(moe.get("expert_dropout", 0.2))

        self.D = D
        self.H = num_horizons

        init_shared_logit = float(adaptive_mixing_cfg.get("init_shared_logit", 1.0))
        self.shared_mix_logits = nn.Parameter(
            torch.full((num_horizons,), init_shared_logit)
        )

        self.shared_expert = Expert2LayerMLP(D, D_ff, dropout)
        self.horizon_experts = nn.ModuleList(
            [Expert2LayerMLP(D, D_ff, dropout) for _ in range(num_horizons)]
        )

        self.point_heads = nn.ModuleList(
            [nn.Linear(D, 1) for _ in range(num_horizons)]
        )
        for ph in self.point_heads:
            nn.init.zeros_(ph.weight)
            nn.init.zeros_(ph.bias)

        if not bool(pers_cfg.get("enabled", True)):
            raise ValueError("model.fusion_horizon.persistence.enabled must be true.")
        resolved_horizons = horizons or list(range(1, num_horizons + 1))
        self.anchor = PersistenceAnchor(
            horizons=resolved_horizons,
            K=int(pers_cfg.get("history_window", 8)),
            init_mode=str(pers_cfg.get("init_mode", "last_plus_trend")),
        )

    def _mix_horizon_features(self, s_h: torch.Tensor) -> torch.Tensor:
        """Build per-horizon features by adaptively mixing shared + per-horizon experts."""
        N_l, H, D = s_h.shape
        shared_out = self.shared_expert(s_h.reshape(-1, D)).reshape(N_l, H, D)
        horizon_out = torch.stack(
            [self.horizon_experts[h](s_h[:, h, :]) for h in range(H)],
            dim=1,
        )
        alpha = torch.sigmoid(self.shared_mix_logits).view(1, H, 1)
        return alpha * shared_out + (1.0 - alpha) * horizon_out

    def forward(
        self,
        s_h: torch.Tensor,
        y_hist: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Produce point predictions per horizon."""
        return self._forward_point(s_h, y_hist)

    def _forward_point(
        self,
        s_h: torch.Tensor,
        y_hist: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Deterministic point-prediction forward in log1p space."""
        N_l, H, D = s_h.shape
        if H != self.H:
            raise ValueError(f"Expected H={self.H}, got {H}")

        v = self._mix_horizon_features(s_h)

        point_cols = []
        for h in range(H):
            point_cols.append(self.point_heads[h](v[:, h, :]).squeeze(-1))
        point_delta = torch.stack(point_cols, dim=1)  # [N_l, H]

        if y_hist is None:
            raise ValueError(
                "MoEHorizonHead requires y_hist for the persistence anchor."
            )
        baseline = self.anchor(torch.log1p(y_hist.clamp_min(0.0)))
        pred_log1p = baseline + point_delta

        pred_mean = torch.expm1(pred_log1p).clamp_min(0.0)
        pred_std = torch.ones_like(pred_mean)
        return pred_mean, pred_std, pred_log1p
