"""Tests for HierEpiGNN Phase 1 data pipeline.

Uses REAL data files to validate actual parsing and graph construction.
Fixtures are module-scoped to avoid expensive reloading per test.
"""

from __future__ import annotations

import copy

import pytest
import torch
import yaml

from src.data import build_dataset
from src.data.transforms import SplitBuilder
from src.utils.seed import set_global_seed


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cfg():
    """Load default config."""
    with open("configs/default.yaml", "r") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def aiv_dataset(cfg):
    """Load AIV dataset (expensive, cache at module level)."""
    set_global_seed(42)
    return build_dataset(cfg, "aiv")


@pytest.fixture(scope="module")
def covid_dataset(cfg):
    """Load COVID dataset (expensive, cache at module level)."""
    set_global_seed(42)
    return build_dataset(cfg, "covid")


@pytest.fixture(scope="module")
def aiv_sample(aiv_dataset):
    """Get first sample from AIV dataset."""
    return aiv_dataset[0]


@pytest.fixture(scope="module")
def covid_sample(covid_dataset):
    """Get first sample from COVID dataset."""
    return covid_dataset[0]


# ---------------------------------------------------------------------------
# TestInterfaceContract
# ---------------------------------------------------------------------------


class TestInterfaceContract:
    """Verify output matches the Dataset Output Contract."""

    REQUIRED_TOP_KEYS = {
        "case_graph", "location_graph", "targets", "mask",
        "population", "metadata", "y_hist",
    }

    def test_aiv_dataset_schema(self, aiv_sample):
        assert set(aiv_sample.keys()) == self.REQUIRED_TOP_KEYS

    def test_covid_dataset_schema(self, covid_sample):
        assert set(covid_sample.keys()) == self.REQUIRED_TOP_KEYS

    def test_case_graph_keys(self, aiv_sample):
        expected = {"x", "strain_emb", "batch"}
        assert set(aiv_sample["case_graph"].keys()) == expected

    def test_location_graph_keys(self, aiv_sample):
        expected = {"x"}
        assert set(aiv_sample["location_graph"].keys()) == expected

    def test_metadata_fields(self, aiv_sample):
        meta = aiv_sample["metadata"]
        assert "time_index" in meta
        assert "dataset_name" in meta
        assert isinstance(meta["time_index"], int)
        assert isinstance(meta["dataset_name"], str)


# ---------------------------------------------------------------------------
# TestCaseGraph
# ---------------------------------------------------------------------------


class TestCaseGraph:
    """Verify case graph structure and shapes."""

    def test_case_graph_shapes(self, aiv_sample):
        cg = aiv_sample["case_graph"]
        x = cg["x"]
        batch = cg["batch"]

        # x is 2D: [N_cases, feat_dim]
        assert x.dim() == 2
        # batch is 1D: [N_cases]
        assert batch.dim() == 1
        assert batch.shape[0] == x.shape[0]

    def test_batch_valid_range(self, aiv_sample):
        cg = aiv_sample["case_graph"]
        batch = cg["batch"]
        num_locations = aiv_sample["location_graph"]["x"].shape[0]
        if batch.numel() > 0:
            assert batch.min().item() >= 0
            assert batch.max().item() < num_locations

    def test_case_feat_dim(self, aiv_sample, cfg):
        """x.shape[1] == temporal + host + strain + genetic_placeholder = 104."""
        data_cfg = cfg["data"]
        expected_dim = (
            data_cfg["temporal_encoding_dim"]
            + data_cfg["host_encoding_dim"]
            + data_cfg["strain_encoding_dim"]
            + data_cfg["genetic_feat_placeholder_dim"]
        )
        assert expected_dim == 104
        cg = aiv_sample["case_graph"]
        if cg["x"].shape[0] > 0:
            assert cg["x"].shape[1] == expected_dim


# ---------------------------------------------------------------------------
# TestLocationGraph
# ---------------------------------------------------------------------------


class TestLocationGraph:
    """Verify location graph structure."""

    def test_location_graph_shapes(self, aiv_sample):
        lg = aiv_sample["location_graph"]
        x = lg["x"]
        assert x.dim() == 2

    def test_aiv_loc_feat_dim(self, aiv_sample):
        """AIV location features = 2*8 (incidence lookback) + 52 (abundance) = 68."""
        lg = aiv_sample["location_graph"]
        assert lg["x"].shape[1] == 68

    def test_covid_loc_feat_dim(self, covid_sample):
        """COVID location features = 8 (incidence lookback)."""
        lg = covid_sample["location_graph"]
        assert lg["x"].shape[1] == 8


# ---------------------------------------------------------------------------
# TestTargetsAndMask
# ---------------------------------------------------------------------------


class TestTargetsAndMask:
    """Verify targets and mask tensors."""

    def test_targets_shape(self, aiv_sample, cfg):
        num_locations = aiv_sample["location_graph"]["x"].shape[0]
        num_horizons = len(cfg["data"]["forecast_horizons"])
        assert aiv_sample["targets"].shape == (num_locations, num_horizons)

    def test_mask_shape(self, aiv_sample):
        num_locations = aiv_sample["location_graph"]["x"].shape[0]
        assert aiv_sample["mask"].shape == (num_locations,)

    def test_mask_dtype(self, aiv_sample):
        assert aiv_sample["mask"].dtype == torch.float32

    def test_mask_values(self, aiv_sample):
        mask = aiv_sample["mask"]
        unique_vals = mask.unique()
        for v in unique_vals:
            assert v.item() in (0.0, 1.0), f"Unexpected mask value: {v.item()}"


# ---------------------------------------------------------------------------
# TestTemporalSplit
# ---------------------------------------------------------------------------


class TestTemporalSplit:
    """Verify temporal split correctness."""

    def test_no_leakage(self, aiv_dataset, cfg):
        train_idx, val_idx, test_idx = SplitBuilder.temporal_split(
            aiv_dataset,
            train_ratio=cfg["data"]["train_ratio"],
            val_ratio=cfg["data"]["val_ratio"],
        )
        if train_idx and val_idx:
            assert max(train_idx) < min(val_idx)
        if val_idx and test_idx:
            assert max(val_idx) < min(test_idx)

    def test_coverage(self, aiv_dataset, cfg):
        train_idx, val_idx, test_idx = SplitBuilder.temporal_split(
            aiv_dataset,
            train_ratio=cfg["data"]["train_ratio"],
            val_ratio=cfg["data"]["val_ratio"],
        )
        assert len(train_idx) + len(val_idx) + len(test_idx) == len(aiv_dataset)


# ---------------------------------------------------------------------------
# TestReproducibility
# ---------------------------------------------------------------------------


class TestReproducibility:
    """Verify deterministic outputs with same seed."""

    def test_same_seed_same_output(self, cfg):
        set_global_seed(42)
        ds1 = build_dataset(cfg, "aiv")
        sample1 = ds1[0]

        set_global_seed(42)
        ds2 = build_dataset(cfg, "aiv")
        sample2 = ds2[0]

        assert torch.equal(sample1["case_graph"]["x"], sample2["case_graph"]["x"])


# ---------------------------------------------------------------------------
# TestBothDatasets
# ---------------------------------------------------------------------------


class TestBothDatasets:
    """Cross-dataset sanity checks."""

    def test_aiv_loads(self, aiv_dataset):
        assert len(aiv_dataset) > 0

    def test_covid_loads(self, covid_dataset):
        assert len(covid_dataset) > 0

    def test_same_output_structure(self, aiv_sample, covid_sample):
        assert set(aiv_sample.keys()) == set(covid_sample.keys())


class TestFusionHorizonHistory:
    """Verify real y_hist emission for persistence-enabled datasets."""

    def test_aiv_emits_nonzero_y_hist_when_persistence_enabled(self, cfg):
        fh_cfg = copy.deepcopy(cfg)
        fh_cfg["model"].setdefault("fusion_horizon", {})
        fh_cfg["model"]["fusion_horizon"]["enabled"] = True
        fh_cfg["model"]["fusion_horizon"].setdefault("persistence", {})
        fh_cfg["model"]["fusion_horizon"]["persistence"]["enabled"] = True
        fh_cfg["model"]["fusion_horizon"]["persistence"]["history_window"] = 8

        set_global_seed(42)
        ds = build_dataset(fh_cfg, "aiv")
        found_nonzero = False
        for idx in range(len(ds)):
            sample = ds[idx]
            assert "y_hist" in sample
            assert sample["y_hist"].shape[1] == 8
            assert torch.isfinite(sample["y_hist"]).all()
            if sample["y_hist"].sum().item() > 0.0:
                found_nonzero = True
                break

        assert found_nonzero

    def test_covid_emits_nonzero_y_hist_when_persistence_enabled(self, cfg):
        fh_cfg = copy.deepcopy(cfg)
        fh_cfg["model"].setdefault("fusion_horizon", {})
        fh_cfg["model"]["fusion_horizon"]["enabled"] = True
        fh_cfg["model"]["fusion_horizon"].setdefault("persistence", {})
        fh_cfg["model"]["fusion_horizon"]["persistence"]["enabled"] = True
        fh_cfg["model"]["fusion_horizon"]["persistence"]["history_window"] = 8

        set_global_seed(42)
        ds = build_dataset(fh_cfg, "covid")
        sample = ds[0]

        assert "y_hist" in sample
        assert sample["y_hist"].shape[1] == 8
        assert torch.isfinite(sample["y_hist"]).all()
        assert sample["y_hist"].sum().item() > 0.0
