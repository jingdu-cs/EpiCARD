"""Graph-free dual-branch epidemic forecaster.

The model has four stages:
1. CaseEventBranch: MLP encoder + attention pooling over case events
2. LocationTemporalBranch: STID-style identity + temporal + residual MLP
3. SharedPrivateFusionHead: shared/private branch decomposition
4. Fusion-horizon head: cross-attention + MoE + persistence anchor
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import torch
import torch.nn as nn

from src.models.fusion_horizon import (
    HorizonCrossAttentionFusion,
    MoEHorizonHead,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scatter utilities (avoid torch_scatter dependency)
# ---------------------------------------------------------------------------

def _scatter_softmax(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Per-group softmax via scatter.

    Parameters
    ----------
    src : Tensor[N]
        Raw scores.
    index : Tensor[N]
        Group indices in [0, dim_size).
    dim_size : int
        Number of groups.

    Returns
    -------
    Tensor[N]
        Softmax-normalized scores within each group.
    """
    # Compute max per group for numerical stability
    max_vals = torch.full((dim_size,), -1e9, device=src.device, dtype=src.dtype)
    max_vals.scatter_reduce_(0, index, src, reduce="amax", include_self=True)
    src_shifted = src - max_vals[index]  # [N]

    exp_src = torch.exp(src_shifted)  # [N]
    sum_exp = torch.zeros(dim_size, device=src.device, dtype=src.dtype)
    sum_exp.scatter_add_(0, index, exp_src)
    return exp_src / sum_exp[index].clamp(min=1e-12)  # [N]


def _scatter_add(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Scatter-add along dim=0.

    Parameters
    ----------
    src : Tensor[N, D]
        Source values.
    index : Tensor[N]
        Target indices in [0, dim_size).
    dim_size : int
        Output size along dim 0.

    Returns
    -------
    Tensor[dim_size, D]
    """
    out = torch.zeros(dim_size, src.shape[1], device=src.device, dtype=src.dtype)
    idx = index.unsqueeze(1).expand_as(src)  # [N, D]
    out.scatter_add_(0, idx, src)
    return out


# ---------------------------------------------------------------------------
# CaseEventBranch
# ---------------------------------------------------------------------------

class CaseEventBranch(nn.Module):
    """Graph-free case event encoder with attention pooling.

    Encodes per-case features via MLP, then pools to per-location embeddings
    using learned attention-weighted aggregation.  No graph construction,
    message passing, or pairwise edge scoring.

    Parameters
    ----------
    cfg : dict
        Full config dict. Reads cfg["model"]["hidden_dim"], cfg["model"]["dropout"],
        cfg["data"]["strain_embedding_dim"], cfg["data"]["strain_embedding_file"].
    d_case : int
        Raw case feature dimension after dropping genetic_placeholder.
        Default 40 = temporal(16) + host(8) + strain(16).
    """

    def __init__(self, cfg: Dict[str, Any], d_case: int = 40) -> None:
        super().__init__()
        model_cfg = cfg["model"]
        D: int = model_cfg["hidden_dim"]
        dropout: float = model_cfg["dropout"]

        # Strain LLM embedding projection (optional)
        strain_emb_dim: int = cfg["data"].get("strain_embedding_dim", 4096)
        has_strain = cfg["data"].get("strain_embedding_file") is not None
        if has_strain:
            self.strain_proj: nn.Module | None = nn.Linear(strain_emb_dim, D)
        else:
            self.strain_proj = None
        encoder_in = d_case

        self.d_case = d_case
        self.hidden_dim = D

        # 2-layer MLP event encoder
        self.event_encoder = nn.Sequential(
            nn.Linear(encoder_in, D),
            nn.GELU(),
            nn.LayerNorm(D),
            nn.Dropout(dropout),
            nn.Linear(D, D),
            nn.GELU(),
            nn.LayerNorm(D),
            nn.Dropout(dropout),
        )

        # Learned attention query vector for pooling
        self.attn_query = nn.Parameter(torch.randn(D) * 0.02)

        # Learned fallback for locations with no cases
        self.null_embedding = nn.Parameter(torch.zeros(D))

        logger.info(
            "CaseEventBranch: d_case=%d, D=%d, strain_proj=%s",
            d_case, D, has_strain,
        )

    def forward(
        self,
        case_x: torch.Tensor,
        case_batch: torch.Tensor,
        N_l: int,
        strain_emb: torch.Tensor | None = None,
        return_case_tokens: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[
        torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """Encode case events and pool to per-location embeddings.

        Parameters
        ----------
        case_x : Tensor[N_c, D_raw]
            Raw case features (104-dim from pipeline; first 40 used).
        case_batch : Tensor[N_c]
            Case-to-location mapping, values in [0, N_l).
        N_l : int
            Number of locations.
        strain_emb : Tensor[N_c, strain_dim] or None
            LLM strain embeddings per case.
        return_case_tokens : bool
            When True, also return the per-case LLM projection
            ``strain_h ∈ R^{N_c × D}`` (zeros when the strain path is not
            active). Used by the fusion-horizon cross-attention module.

        Returns
        -------
        z_case : Tensor[N_l, D]
            Per-location case embedding.
        has_cases : Tensor[N_l]
            Boolean mask — True for locations with at least one case.
        case_tokens : Tensor[N_c, D]  (only when ``return_case_tokens=True``)
            Per-case LLM projection pre-pooling.
        """
        device = case_x.device
        D = self.hidden_dim

        # Slice to meaningful features, drop genetic_placeholder (last 64 zeros)
        x = case_x[:, : self.d_case]  # [N_c, 40]
        N_c = x.shape[0]

        # Handle empty case set
        if N_c == 0:
            z = self.null_embedding.unsqueeze(0).expand(N_l, -1).clone()  # [N_l, D]
            has_cases = torch.zeros(N_l, dtype=torch.bool, device=device)
            if return_case_tokens:
                empty_tokens = torch.zeros(0, D, device=device)  # [0, D]
                return z, has_cases, empty_tokens
            return z, has_cases

        # Optional: project strain embeddings for downstream semantic tokens.
        if self.strain_proj is not None and strain_emb is not None:
            strain_h = self.strain_proj(strain_emb)  # [N_c, D]
        else:
            strain_h = None

        # Encode events
        h = self.event_encoder(x)  # [N_c, D]

        # Attention pooling: per-location softmax over cases
        scores = h @ self.attn_query  # [N_c]
        alpha = _scatter_softmax(scores, case_batch, dim_size=N_l)  # [N_c]
        weighted = alpha.unsqueeze(-1) * h  # [N_c, D]
        z = _scatter_add(weighted, case_batch, dim_size=N_l)  # [N_l, D]

        # Compute has_cases mask
        ones = torch.ones(N_c, device=device)
        counts = torch.zeros(N_l, device=device)
        counts.scatter_add_(0, case_batch, ones)
        has_cases = counts > 0  # [N_l]

        # Fill locations with no cases using null embedding
        z[~has_cases] = self.null_embedding

        if return_case_tokens:
            # Fall back to the encoder output h when the strain path is off,
            # so downstream cross-attention always has per-case tokens.
            case_tokens = strain_h if strain_h is not None else h  # [N_c, D]
            return z, has_cases, case_tokens
        return z, has_cases


# ---------------------------------------------------------------------------
# ResidualMLPBlock
# ---------------------------------------------------------------------------

class ResidualMLPBlock(nn.Module):
    """Pre-norm residual MLP block: LayerNorm → Linear → GELU → Dropout → Linear → Dropout + skip.

    Parameters
    ----------
    D : int
        Hidden dimension.
    dropout : float
        Dropout rate.
    """

    def __init__(self, D: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(D)
        self.mlp = nn.Sequential(
            nn.Linear(D, D),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(D, D),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [*, D] → [*, D]."""
        return x + self.mlp(self.norm(x))


# ---------------------------------------------------------------------------
# LocationTemporalBranch
# ---------------------------------------------------------------------------

class LocationTemporalBranch(nn.Module):
    """Graph-free location temporal encoder (STID-style).

    Combines location identity embedding, temporal position embedding,
    and historical feature projection through residual MLP blocks.
    No adjacency, graph convolution, or neighbor aggregation.

    Parameters
    ----------
    cfg : dict
        Full config dict.
    d_loc : int
        Location feature dimension (68 for AIV, 8 for COVID).
    num_locations : int
        Number of locations (for the ID embedding table).
    """

    def __init__(self, cfg: Dict[str, Any], d_loc: int, num_locations: int) -> None:
        super().__init__()
        model_cfg = cfg["model"]
        D: int = model_cfg["hidden_dim"]
        dropout: float = model_cfg["dropout"]

        gf_cfg = model_cfg.get("graph_free", {})
        n_blocks: int = gf_cfg.get("num_residual_blocks", 3)
        max_time: int = gf_cfg.get("max_time_windows", 512)

        self.hidden_dim = D
        self.num_locations = num_locations

        # Embeddings
        self.loc_id_emb = nn.Embedding(num_locations, D)
        self.time_emb = nn.Embedding(max_time, D)

        # History projection
        self.history_proj = nn.Linear(d_loc, D)

        # Input projection: concat of 3 embeddings → D
        self.input_proj = nn.Sequential(
            nn.Linear(3 * D, D),
            nn.GELU(),
            nn.LayerNorm(D),
            nn.Dropout(dropout),
        )

        # Residual MLP blocks
        self.blocks = nn.ModuleList([
            ResidualMLPBlock(D, dropout) for _ in range(n_blocks)
        ])

        # Output normalization
        self.out_norm = nn.LayerNorm(D)

        logger.info(
            "LocationTemporalBranch: d_loc=%d, D=%d, num_locations=%d, "
            "n_blocks=%d, max_time=%d",
            d_loc, D, num_locations, n_blocks, max_time,
        )

    def forward(
        self,
        loc_x: torch.Tensor,
        loc_ids: torch.Tensor,
        time_index: int,
    ) -> torch.Tensor:
        """Encode location-level historical features.

        Parameters
        ----------
        loc_x : Tensor[N_l, D_loc]
            Location features (historical incidence + covariates).
        loc_ids : Tensor[N_l]
            Location indices for identity embedding.
        time_index : int
            Temporal position index for time embedding.

        Returns
        -------
        Tensor[N_l, D]
            Location temporal embedding.
        """
        N_l = loc_x.shape[0]
        device = loc_x.device

        # Project historical features
        h_hist = self.history_proj(loc_x)  # [N_l, D]

        # Location identity embedding
        h_loc = self.loc_id_emb(loc_ids)  # [N_l, D]

        # Temporal position embedding (shared across locations in this window)
        t_idx = min(time_index, self.time_emb.num_embeddings - 1)
        t_idx = max(t_idx, 0)
        h_time = self.time_emb(
            torch.tensor(t_idx, device=device, dtype=torch.long)
        )  # [D]
        h_time = h_time.unsqueeze(0).expand(N_l, -1)  # [N_l, D]

        # Concatenate and project
        h = self.input_proj(
            torch.cat([h_hist, h_loc, h_time], dim=-1)
        )  # [N_l, D]

        # Residual MLP blocks
        for block in self.blocks:
            h = block(h)  # [N_l, D]

        return self.out_norm(h)  # [N_l, D]


# ---------------------------------------------------------------------------
# SharedPrivateFusionHead
# ---------------------------------------------------------------------------

class SharedPrivateFusionHead(nn.Module):
    """Lightweight shared/private branch decomposition for coarse fusion.

    Produces a fused state ``s`` plus intermediate shared/private tensors for
    inspection. Downstream modules continue to consume only ``s``.
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__()
        model_cfg = cfg["model"]
        D: int = model_cfg["hidden_dim"]
        dropout: float = model_cfg["dropout"]

        def _proj(in_dim: int, out_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(in_dim, D),
                nn.GELU(),
                nn.LayerNorm(D),
                nn.Dropout(dropout),
                nn.Linear(D, out_dim),
            )

        self.shared_proj_case = _proj(D, D)
        self.private_proj_case = _proj(D, D)
        self.shared_proj_loc = _proj(D, D)
        self.private_proj_loc = _proj(D, D)
        self.gate_mlp = _proj(2 * D, 1)
        self.private_projector = _proj(2 * D, D)
        self.output_projector = _proj(2 * D, D)

    def forward(
        self, z_case: torch.Tensor, z_loc: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Fuse branch embeddings with shared/private decomposition."""
        case_shared = self.shared_proj_case(z_case)
        case_private = self.private_proj_case(z_case)
        loc_shared = self.shared_proj_loc(z_loc)
        loc_private = self.private_proj_loc(z_loc)

        shared_gate = torch.sigmoid(
            self.gate_mlp(torch.cat([case_shared, loc_shared], dim=-1))
        )  # [N_l, 1]
        shared_fused = (
            shared_gate * case_shared + (1.0 - shared_gate) * loc_shared
        )  # [N_l, D]
        private_fused = self.private_projector(
            torch.cat([case_private, loc_private], dim=-1)
        )  # [N_l, D]
        s = self.output_projector(
            torch.cat([shared_fused, private_fused], dim=-1)
        )  # [N_l, D]
        return {
            "s": s,
            "fusion_shared_case": case_shared,
            "fusion_shared_loc": loc_shared,
            "fusion_private_case": case_private,
            "fusion_private_loc": loc_private,
            "fusion_shared_gate": shared_gate,
            "fusion_shared_fused": shared_fused,
            "fusion_private_fused": private_fused,
        }


# ---------------------------------------------------------------------------
# GraphFreeDualBranchForecaster
# ---------------------------------------------------------------------------

class GraphFreeDualBranchForecaster(nn.Module):
    """Graph-free dual-branch epidemic forecaster.

    Pipeline:
        z_case  = CaseEventBranch(case events per location)
        z_loc   = LocationTemporalBranch(location history + ID + time)
        s       = SharedPrivateFusionHead(z_case, z_loc)
        s_h     = HorizonCrossAttentionFusion(s, per-case tokens)
        y_hat   = MoEHorizonHead(s_h, y_hist)

    Parameters
    ----------
    cfg : dict
        Full config dict with "model" and "data" sections.
    d_loc : int
        Location feature dimension (68 for AIV, 8 for COVID).
    num_locations : int
        Number of locations (for location ID embedding).
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
        d_loc: int,
        num_locations: int,
    ) -> None:
        super().__init__()
        model_cfg = cfg["model"]

        self.hidden_dim: int = model_cfg["hidden_dim"]
        self.forecast_horizons: list[int] = list(cfg["data"]["forecast_horizons"])
        self.num_horizons: int = len(self.forecast_horizons)

        fh_cfg = model_cfg.get("fusion_horizon", {})
        if not bool(fh_cfg.get("enabled", True)):
            raise ValueError("model.fusion_horizon.enabled must be true.")

        head_mode = str(fh_cfg.get("head", {}).get("mode", "shared_private"))
        if head_mode != "shared_private":
            raise ValueError(
                "model.fusion_horizon.head.mode must be 'shared_private', "
                f"got {head_mode!r}."
            )
        self.fusion_head_mode: str = head_mode

        if not bool(fh_cfg.get("cross_attn", {}).get("enabled", True)):
            raise ValueError("model.fusion_horizon.cross_attn.enabled must be true.")

        self.case_branch = CaseEventBranch(cfg)
        self.loc_branch = LocationTemporalBranch(cfg, d_loc, num_locations)
        self.fusion = SharedPrivateFusionHead(cfg)
        self.horizon_cross_attn = HorizonCrossAttentionFusion(
            cfg, num_horizons=self.num_horizons
        )
        self.moe_horizon_head = MoEHorizonHead(
            cfg,
            num_horizons=self.num_horizons,
            horizons=self.forecast_horizons,
        )

        num_params = sum(p.numel() for p in self.parameters())
        logger.info(
            "GraphFreeDualBranchForecaster: D=%d, H=%d, d_loc=%d, "
            "num_locations=%d, params=%s",
            self.hidden_dim, self.num_horizons, d_loc,
            num_locations, f"{num_params:,}",
        )

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Run the full graph-free forward pass.

        Parameters
        ----------
        batch : dict
            Dataset sample with keys: case_graph, location_graph, targets,
            mask, metadata, and optional y_hist.

        Returns
        -------
        dict
            Compatible with point-only CombinedLoss. Keys include
            ``pred_mean``, ``pred_std`` placeholder, and ``pred_log1p``.
        """
        case_x = batch["case_graph"]["x"]  # [N_c, 104]
        case_batch_idx = batch["case_graph"]["batch"]  # [N_c]
        strain_emb = batch["case_graph"].get("strain_emb")  # [N_c, E_s] or None

        loc_x = batch["location_graph"]["x"]  # [N_l, D_loc]

        N_l = loc_x.shape[0]
        device = loc_x.device

        time_index: int = batch.get("metadata", {}).get("time_index", 0)
        loc_ids = torch.arange(N_l, device=device)  # [N_l]

        z_case, _, case_tokens = self.case_branch(
            case_x, case_batch_idx, N_l, strain_emb,
            return_case_tokens=True,
        )  # [N_l, D], [N_l], [N_c, D]
        z_loc = self.loc_branch(loc_x, loc_ids, time_index)  # [N_l, D]

        fusion_out = self.fusion(z_case, z_loc)
        s = fusion_out["s"]

        s_h = self.horizon_cross_attn(s, case_tokens, case_batch_idx)  # [N_l, H, D]
        y_hist = batch.get("y_hist")  # [N_l, K] or None
        pred_mean, pred_std, pred_log1p = self.moe_horizon_head(s_h, y_hist)

        output: Dict[str, Any] = {
            "location_embeddings": s,
            "pred_mean": pred_mean,    # [N_l, H]
            "pred_std": pred_std,      # [N_l, H]
            "pred_log1p": pred_log1p,  # [N_l, H]
        }
        for key, value in fusion_out.items():
            if key != "s":
                output[key] = value
        return output
