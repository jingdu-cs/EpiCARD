"""Tests for the point-only GraphFreeDualBranchForecaster."""

from __future__ import annotations

from typing import Any, Dict

import pytest
import torch
import yaml


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _load_default_cfg() -> Dict[str, Any]:
    """Load default config from configs/default.yaml."""
    with open("configs/default.yaml", "r") as f:
        return yaml.safe_load(f)


def _make_cfg(**overrides: Any) -> Dict[str, Any]:
    """Create a config dict with optional model-level overrides."""
    cfg = _load_default_cfg()
    for k, v in overrides.items():
        cfg["model"][k] = v
    return cfg


def _make_batch(
    N_c: int = 20,
    N_l: int = 5,
    D_loc: int = 68,
    H: int = 3,
    K: int = 12,
    seed: int = 42,
    strain_emb_dim: int | None = None,
) -> Dict[str, Any]:
    """Create a synthetic batch matching the Dataset Output Contract.

    Parameters
    ----------
    N_c : int
        Number of cases.
    N_l : int
        Number of locations.
    D_loc : int
        Location feature dimension (68 for AIV, 8 for COVID).
    H : int
        Number of forecast horizons.
    K : int
        Persistence-anchor history window length.
    seed : int
        Random seed.
    strain_emb_dim : int or None
        If set, include strain_emb tensor with this dimension.
    """
    torch.manual_seed(seed)

    case_x = torch.randn(N_c, 104)
    case_batch = torch.randint(0, max(N_l, 1), (N_c,))
    for loc in range(min(N_l, N_c)):
        case_batch[loc] = loc

    strain_emb = None
    if strain_emb_dim is not None and N_c > 0:
        strain_emb = torch.randn(N_c, strain_emb_dim)

    return {
        "case_graph": {
            "x": case_x,
            "strain_emb": strain_emb,
            "batch": case_batch,
        },
        "location_graph": {
            "x": torch.randn(N_l, D_loc),
        },
        "targets": torch.rand(N_l, H).clamp(min=0.01),
        "mask": torch.ones(N_l),
        "population": torch.full((N_l,), 100000.0),
        "y_hist": torch.rand(N_l, K).clamp(min=0.0),
        "metadata": {
            "time_index": 0,
            "dataset_name": "aiv" if D_loc == 68 else "covid",
        },
    }


class TestSharedPrivateFusionHead:
    def test_contract_and_shapes(self) -> None:
        from src.models.graph_free_model import SharedPrivateFusionHead

        cfg = _make_cfg()
        mod = SharedPrivateFusionHead(cfg)
        N_l = 5
        D = cfg["model"]["hidden_dim"]
        z_case = torch.randn(N_l, D)
        z_loc = torch.randn(N_l, D)

        out = mod(z_case, z_loc)

        assert out["s"].shape == (N_l, D)
        assert out["fusion_shared_case"].shape == (N_l, D)
        assert out["fusion_shared_loc"].shape == (N_l, D)
        assert out["fusion_private_case"].shape == (N_l, D)
        assert out["fusion_private_loc"].shape == (N_l, D)
        assert out["fusion_shared_gate"].shape == (N_l, 1)
        assert out["fusion_shared_fused"].shape == (N_l, D)
        assert out["fusion_private_fused"].shape == (N_l, D)
        for value in out.values():
            assert torch.isfinite(value).all()

    def test_gate_behavior_changes_with_each_branch(self) -> None:
        from src.models.graph_free_model import SharedPrivateFusionHead

        cfg = _make_cfg()
        mod = SharedPrivateFusionHead(cfg)
        mod.eval()
        N_l = 4
        D = cfg["model"]["hidden_dim"]
        z_case = torch.randn(N_l, D)
        z_loc = torch.randn(N_l, D)

        with torch.no_grad():
            out_base = mod(z_case, z_loc)
            out_case = mod(z_case + 0.5, z_loc)
            out_loc = mod(z_case, z_loc + 0.5)

        gate = out_base["fusion_shared_gate"]
        assert ((gate >= 0.0) & (gate <= 1.0)).all()
        assert not torch.allclose(
            out_base["fusion_shared_fused"], out_case["fusion_shared_fused"]
        )
        assert not torch.allclose(
            out_base["fusion_shared_fused"], out_loc["fusion_shared_fused"]
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestForwardShapes:
    """Verify output shapes for GraphFreeDualBranchForecaster."""

    @pytest.mark.parametrize("D_loc,dataset", [(68, "aiv"), (8, "covid")])
    def test_forward_shapes(self, D_loc: int, dataset: str) -> None:
        from src.models import build_model

        cfg = _make_cfg()
        N_c, N_l, H = 20, 5, 3
        batch = _make_batch(N_c=N_c, N_l=N_l, D_loc=D_loc, H=H)
        model = build_model(cfg, d_loc=D_loc, num_locations=N_l)
        model.eval()

        with torch.no_grad():
            out = model(batch)

        D = cfg["model"]["hidden_dim"]
        assert out["pred_mean"].shape == (N_l, H)
        assert out["pred_std"].shape == (N_l, H)
        assert out["pred_log1p"].shape == (N_l, H)
        assert out["location_embeddings"].shape == (N_l, D)

    def test_empty_events(self) -> None:
        """N_c=0: model handles empty case set without NaN."""
        from src.models import build_model

        cfg = _make_cfg()
        N_l, H = 5, 3
        batch = _make_batch(N_c=0, N_l=N_l, D_loc=68, H=H)
        model = build_model(cfg, d_loc=68, num_locations=N_l)
        model.eval()

        with torch.no_grad():
            out = model(batch)

        assert out["pred_mean"].shape == (N_l, H)
        assert not torch.isnan(out["pred_mean"]).any()
        assert not torch.isnan(out["pred_std"]).any()
        assert not torch.isnan(out["pred_log1p"]).any()

    def test_single_location(self) -> None:
        """N_l=1: single location works."""
        from src.models import build_model

        cfg = _make_cfg()
        batch = _make_batch(N_c=5, N_l=1, D_loc=68)
        model = build_model(cfg, d_loc=68, num_locations=1)
        model.eval()

        with torch.no_grad():
            out = model(batch)

        assert out["pred_mean"].shape == (1, 3)


class TestLossCompatibility:
    """Verify model output is compatible with CombinedLoss."""

    def test_output_keys_for_loss(self) -> None:
        """Model output has all keys CombinedLoss needs."""
        from src.models import build_model

        cfg = _make_cfg()
        batch = _make_batch()
        model = build_model(cfg, d_loc=68, num_locations=5)
        model.eval()

        with torch.no_grad():
            out = model(batch)

        # CombinedLoss reads these keys
        assert "pred_mean" in out
        assert "pred_std" in out
        assert "pred_log1p" in out

    def test_loss_forward(self) -> None:
        """CombinedLoss computes without error on model output."""
        from src.models import build_model
        from src.training.losses import CombinedLoss

        cfg = _make_cfg()
        batch = _make_batch()
        model = build_model(cfg, d_loc=68, num_locations=5)
        loss_fn = CombinedLoss(cfg)
        model.eval()

        with torch.no_grad():
            out = model(batch)
            loss_dict = loss_fn(out, batch["targets"], batch["mask"])

        assert "loss" in loss_dict
        assert "loss_pred" in loss_dict
        assert not torch.isnan(loss_dict["loss"])

    def test_backward_no_nan(self) -> None:
        """Backward pass produces no NaN gradients."""
        from src.models import build_model
        from src.training.losses import CombinedLoss

        cfg = _make_cfg()
        batch = _make_batch()
        model = build_model(cfg, d_loc=68, num_locations=5)
        loss_fn = CombinedLoss(cfg)
        model.train()

        out = model(batch)
        loss_dict = loss_fn(out, batch["targets"], batch["mask"])
        loss_dict["loss"].backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any(), f"NaN grad in {name}"


class TestPointMode:
    def test_fusion_horizon_outputs_pred_log1p(self) -> None:
        from src.models import build_model
        from src.training.losses import CombinedLoss

        cfg = _make_cfg()
        cfg["model"]["fusion_horizon"]["enabled"] = True
        cfg["model"]["fusion_horizon"]["cross_attn"]["enabled"] = True
        cfg["model"]["fusion_horizon"]["moe"]["enabled"] = True
        cfg["model"]["fusion_horizon"]["moe"]["adaptive_mixing"]["enabled"] = True
        cfg["model"]["fusion_horizon"]["persistence"]["enabled"] = True

        batch = _make_batch()
        batch["y_hist"] = torch.rand(5, 12) * 10.0

        model = build_model(cfg, d_loc=68, num_locations=5)
        loss_fn = CombinedLoss(cfg)
        model.train()

        out = model(batch)
        assert "pred_log1p" in out
        assert out["pred_mean"].shape == (5, 3)
        assert out["pred_std"].shape == (5, 3)
        assert out["pred_log1p"].shape == (5, 3)
        assert torch.isfinite(out["pred_mean"]).all()
        assert torch.isfinite(out["pred_log1p"]).all()
        assert (out["pred_mean"] >= 0.0).all()

        loss_dict = loss_fn(out, batch["targets"], batch["mask"])
        assert torch.isfinite(loss_dict["loss"])


class TestStrainEmbeddings:
    """Test strain embedding flow through CaseEventBranch."""

    def test_with_strain_embeddings(self) -> None:
        """Strain embeddings still flow through the fusion-horizon path."""
        from src.models import build_model

        cfg = _make_cfg()
        cfg["data"]["strain_embedding_file"] = "dummy.pt"
        cfg["data"]["strain_embedding_dim"] = 4096
        cfg["model"]["fusion_horizon"]["enabled"] = True
        cfg["model"]["fusion_horizon"]["cross_attn"]["enabled"] = True

        batch = _make_batch(strain_emb_dim=4096)
        batch["y_hist"] = torch.rand(5, 12)
        model = build_model(cfg, d_loc=68, num_locations=5)
        model.eval()

        with torch.no_grad():
            out = model(batch)

        assert out["pred_mean"].shape == (5, 3)
        assert not torch.isnan(out["pred_mean"]).any()

    def test_strain_embeddings_with_backward(self) -> None:
        """Strain embedding path has valid gradients through cross-attention."""
        from src.models import build_model
        from src.training.losses import CombinedLoss

        cfg = _make_cfg()
        cfg["data"]["strain_embedding_file"] = "dummy.pt"
        cfg["data"]["strain_embedding_dim"] = 4096
        cfg["model"]["fusion_horizon"]["enabled"] = True
        cfg["model"]["fusion_horizon"]["cross_attn"]["enabled"] = True

        batch = _make_batch(strain_emb_dim=4096)
        batch["y_hist"] = torch.rand(5, 12)
        model = build_model(cfg, d_loc=68, num_locations=5)
        loss_fn = CombinedLoss(cfg)
        model.train()

        out = model(batch)
        loss_dict = loss_fn(out, batch["targets"], batch["mask"])
        loss_dict["loss"].backward()

        # Check strain_proj has gradients
        assert model.case_branch.strain_proj is not None
        assert model.case_branch.strain_proj.weight.grad is not None

    def test_case_embeddings_do_not_change_z_case_when_only_strain_emb_changes(self) -> None:
        """Semantic embeddings should affect case_tokens, not z_case."""
        from src.models.graph_free_model import CaseEventBranch

        cfg = _make_cfg()
        cfg["data"]["strain_embedding_file"] = "dummy.pt"
        cfg["data"]["strain_embedding_dim"] = 4096

        branch = CaseEventBranch(cfg)
        branch.eval()

        N_c, N_l = 6, 3
        case_x = torch.randn(N_c, 104)
        case_batch = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
        strain_a = torch.randn(N_c, 4096)
        strain_b = torch.randn(N_c, 4096)

        with torch.no_grad():
            z_a, has_a, tokens_a = branch(
                case_x, case_batch, N_l, strain_a, return_case_tokens=True
            )
            z_b, has_b, tokens_b = branch(
                case_x, case_batch, N_l, strain_b, return_case_tokens=True
            )

        assert torch.allclose(z_a, z_b)
        assert torch.equal(has_a, has_b)
        assert not torch.allclose(tokens_a, tokens_b)


def test_shared_private_fusion_point_mode_exposes_internals() -> None:
    from src.models import build_model

    cfg = _make_cfg()
    cfg["model"]["fusion_horizon"]["enabled"] = True
    cfg["model"]["fusion_horizon"]["head"]["mode"] = "shared_private"
    cfg["model"]["fusion_horizon"]["cross_attn"]["enabled"] = True
    cfg["model"]["fusion_horizon"]["moe"]["enabled"] = True
    cfg["model"]["fusion_horizon"]["moe"]["adaptive_mixing"]["enabled"] = True
    cfg["model"]["fusion_horizon"]["persistence"]["enabled"] = True

    batch = _make_batch(strain_emb_dim=4096)
    batch["case_graph"]["strain_emb"] = torch.randn(
        batch["case_graph"]["x"].shape[0], 4096
    )
    batch["y_hist"] = torch.rand(5, 12)

    model = build_model(cfg, d_loc=68, num_locations=5)
    model.eval()
    with torch.no_grad():
        out = model(batch)

    for key in (
        "fusion_shared_case",
        "fusion_shared_loc",
        "fusion_private_case",
        "fusion_private_loc",
        "fusion_shared_gate",
        "fusion_shared_fused",
        "fusion_private_fused",
    ):
        assert key in out
        assert torch.isfinite(out[key]).all()
    assert out["location_embeddings"].shape == (5, cfg["model"]["hidden_dim"])


def test_shared_private_fusion_loss_backward_with_decomp_penalty() -> None:
    from src.models import build_model
    from src.training.losses import CombinedLoss

    cfg = _make_cfg()
    cfg["model"]["fusion_horizon"]["enabled"] = True
    cfg["model"]["fusion_horizon"]["head"]["mode"] = "shared_private"
    cfg["model"]["fusion_horizon"]["head"]["decomp_loss_weight"] = 1.0e-3
    cfg["model"]["fusion_horizon"]["cross_attn"]["enabled"] = True
    cfg["model"]["fusion_horizon"]["moe"]["enabled"] = True
    cfg["model"]["fusion_horizon"]["moe"]["adaptive_mixing"]["enabled"] = True
    cfg["model"]["fusion_horizon"]["persistence"]["enabled"] = True

    batch = _make_batch(strain_emb_dim=4096)
    batch["case_graph"]["strain_emb"] = torch.randn(
        batch["case_graph"]["x"].shape[0], 4096
    )
    batch["y_hist"] = torch.rand(5, 12)

    model = build_model(cfg, d_loc=68, num_locations=5)
    loss_fn = CombinedLoss(cfg)
    model.train()

    out = model(batch)
    loss_dict = loss_fn(out, batch["targets"], batch["mask"])
    assert loss_dict["loss_decomp"].item() >= 0.0
    loss_dict["loss"].backward()

    if model.fusion_head_mode == "shared_private":
        assert model.fusion.shared_proj_case[0].weight.grad is not None


