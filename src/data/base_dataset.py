"""Base dataset class for epidemic graph datasets.

Defines the template method pattern: subclasses implement data-loading hooks,
this class handles time windowing and output assembly to satisfy the
Dataset Output Contract (see codebase-invariants.md).
"""

from __future__ import annotations

import abc
import logging
from datetime import datetime, timedelta
from typing import Any, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.graph_builder import (
    GraphBuilder,
    HostEncoder,
    LocationNormalizer,
    StrainEncoder,
)
from src.utils.seed import set_global_seed

logger = logging.getLogger(__name__)


class EpidemicDataset(Dataset):
    """Base class for epidemic graph datasets.

    Subclasses MUST implement:
        _load_raw_data() -> None
            Populates: self.cases_df (with 'parsed_date' and 'loc_key' columns),
                       self.location_index (dict mapping (state, county) -> int),
                       and any dataset-specific data.

        _build_location_features(window_start, window_end) -> Tensor[N_locs, D_loc]

        _get_targets(window_end, horizons) -> Tensor[N_locs, H]
            Incidence counts for each location at each horizon.

        _get_mask(window_end, horizons) -> Tensor[N_locs]
            1.0 for valid locations, 0.0 for invalid.
    """

    def __init__(self, cfg: dict, dataset_name: str) -> None:
        """Initialize the epidemic dataset.

        Args:
            cfg: Full config dict (from default.yaml).
            dataset_name: Either "aiv" or "covid".
        """
        super().__init__()
        self.cfg = cfg
        self.dataset_name = dataset_name
        self.data_cfg = cfg["data"]
        self.dataset_cfg = cfg["data"][dataset_name]

        # Build shared helpers
        self.location_normalizer = LocationNormalizer()
        self.graph_builder = GraphBuilder(self.data_cfg)
        self.host_encoder = HostEncoder(dim=self.data_cfg["host_encoding_dim"])
        self.strain_encoder = StrainEncoder(dim=self.data_cfg["strain_encoding_dim"])

        # Populated by subclass _load_raw_data
        self.cases_df: pd.DataFrame
        self.location_index: dict[tuple[str, str], int]

        # Per-capita target normalization (opt-in via data.per_capita_normalize)
        self.per_capita_normalize: bool = self.data_cfg.get(
            "per_capita_normalize", False
        )
        self.per_capita_base: float = float(
            self.data_cfg.get("per_capita_base", 100_000)
        )
        # Subclass populates this dict if it has population data available.
        self.location_population: dict[tuple[str, str], int] = {}
        # Slice of location feature columns that are count-valued (default: all).
        # Subclasses may override (e.g. AIV to exclude abundance columns).
        self._count_feature_slice: slice = slice(None)

        # Resolve time-unit before subclass _load_raw_data (which may need it)
        time_unit = self.data_cfg.get("time_unit", "weeks")
        self.days_per_step: int = 1 if time_unit == "days" else 7

        # Template method sequence
        self._load_raw_data()
        self._build_time_windows()
        self._fit_encoders()

        if self.per_capita_normalize and not self.location_population:
            logger.warning(
                "per_capita_normalize=True but dataset '%s' has no "
                "location_population data. Normalization will be a no-op.",
                self.dataset_name,
            )

        logger.info(
            "Dataset '%s' initialized: %d locations, %d cases, %d time windows",
            self.dataset_name,
            len(self.location_index),
            len(self.cases_df),
            len(self.time_windows),
        )

    # ------------------------------------------------------------------
    # Abstract methods (subclasses MUST implement)
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def _load_raw_data(self) -> None:
        """Load raw data and populate self.cases_df, self.location_index,
        and any dataset-specific attributes."""

    @abc.abstractmethod
    def _build_location_features(
        self, window_start: datetime, window_end: datetime
    ) -> torch.Tensor:
        """Build location node features for a given time window.

        Returns:
            Tensor of shape [num_locations, loc_feat_dim].
        """

    @abc.abstractmethod
    def _get_targets(
        self, window_end: datetime, horizons: list[int]
    ) -> torch.Tensor:
        """Compute incidence targets for each location at each horizon.

        Returns:
            Tensor of shape [num_locations, len(horizons)].
        """

    @abc.abstractmethod
    def _get_mask(
        self, window_end: datetime, horizons: list[int]
    ) -> torch.Tensor:
        """Compute validity mask for each location.

        Returns:
            Tensor of shape [num_locations] with 1.0 for valid, 0.0 for invalid.
        """

    def _get_y_hist(self, window_end: datetime, K: int) -> torch.Tensor:
        """Return recent K observed incidence counts per location.

        Used as input to the persistence anchor in the fusion-horizon head.
        The K most recent steps are arranged chronologically: index K-1 is the
        most recent step (ending at `window_end`). Respects the
        no-future-leakage rule: only steps strictly before `window_end` are
        used.

        Default returns zeros; dataset subclasses should override when
        `model.fusion_horizon.persistence.enabled=true`.

        Returns:
            Tensor of shape [num_locations, K].
        """
        n_locs = len(self.location_index)
        if not getattr(self, "_y_hist_override_warned", False):
            logger.warning(
                "Dataset '%s' uses default zero y_hist fallback; override "
                "_get_y_hist to enable persistence anchor.",
                self.dataset_name,
            )
            self._y_hist_override_warned = True
        return torch.zeros(n_locs, K, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Time windowing
    # ------------------------------------------------------------------

    def _build_time_windows(self) -> None:
        """Compute sliding time windows from the cases dataframe.

        A window [start, start + window_size) is valid when:
          - At least 1 case exists within the window
          - Enough future data exists for max(forecast_horizons) steps
            after the window end
        """
        window_size: int = self.data_cfg.get(
            "window_size", self.data_cfg.get("window_size_weeks", 8)
        )
        stride: int = self.data_cfg.get(
            "stride", self.data_cfg.get("stride_weeks", 1)
        )
        forecast_horizons: list[int] = self.data_cfg["forecast_horizons"]

        window_size_days = timedelta(days=window_size * self.days_per_step)
        stride_days = timedelta(days=stride * self.days_per_step)
        max_horizon_days = timedelta(
            days=max(forecast_horizons) * self.days_per_step
        )

        dates = self.cases_df["parsed_date"]
        min_date: datetime = dates.min()
        max_date: datetime = self._get_effective_end_date()

        time_unit = self.data_cfg.get("time_unit", "weeks")
        logger.debug(
            "Building time windows: date range %s to %s, "
            "window=%d %s, stride=%d %s",
            min_date.date(),
            max_date.date(),
            window_size,
            time_unit,
            stride,
            time_unit,
        )

        self.time_windows: list[tuple[datetime, datetime]] = []
        window_start = min_date

        while True:
            window_end = window_start + window_size_days

            # Check: enough future data for the longest forecast horizon
            if window_end + max_horizon_days > max_date + timedelta(days=1):
                break

            # Check: at least 1 case in this window
            cases_in_window = (dates >= window_start) & (dates < window_end)
            if cases_in_window.any():
                self.time_windows.append((window_start, window_end))

            window_start += stride_days

        logger.info(
            "Built %d valid time windows for '%s'",
            len(self.time_windows),
            self.dataset_name,
        )

    def _get_effective_end_date(self) -> datetime:
        """Return the effective end date for time window construction.

        By default uses the case data's max date.  Subclasses may override
        to account for other data sources (e.g. incidence data) whose date
        range may be shorter than the case data.
        """
        return self.cases_df["parsed_date"].max()

    # ------------------------------------------------------------------
    # Encoder fitting
    # ------------------------------------------------------------------

    def _fit_encoders(self) -> None:
        """Fit the HostEncoder on all cases' Category and Subcategory.

        StrainEncoder is hash-based and requires no fitting.
        """
        categories = self.cases_df["Category"].tolist()
        subcategories = self.cases_df["Subcategory"].tolist()
        self.host_encoder.fit(categories, subcategories)
        # Also fit the graph_builder's internal host_encoder
        self.graph_builder.host_encoder.fit(categories, subcategories)
        logger.debug(
            "HostEncoder fitted on %d unique (Category, Subcategory) pairs",
            len(set(zip(categories, subcategories))),
        )

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.time_windows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return a single sample matching the Dataset Output Contract.

        Returns dict with keys: case_graph, location_graph, targets, mask, metadata.
        """
        # 1. Get window bounds
        window_start, window_end = self.time_windows[idx]

        # 2. Filter cases to window
        mask = (self.cases_df["parsed_date"] >= window_start) & (
            self.cases_df["parsed_date"] < window_end
        )
        window_cases = self.cases_df[mask].copy()

        # 3. Cap at max_cases_per_window (deterministic subsample)
        max_cases: int = self.data_cfg["max_cases_per_window"]
        if len(window_cases) > max_cases:
            window_cases = window_cases.sample(
                n=max_cases, random_state=self.cfg["seed"] + idx
            )
            logger.debug(
                "Window %d: subsampled cases from %d to %d",
                idx,
                mask.sum(),
                max_cases,
            )

        # 3b. Stamp the as-of date on every row when the v2 causal-cutoff
        # embedding cache is in use. The forecast origin is window_end and
        # the per-record as-of date is window_end - reporting_lag_days. The
        # graph builder consumes the ``as_of_date`` column when its loaded
        # cache is the v2_as_of format; on legacy caches the column is
        # ignored.
        wc_cfg = self.data_cfg.get("window_card") or {}
        if wc_cfg.get("causal_cutoff", False):
            lag_days = int(wc_cfg.get("reporting_lag_days", 14))
            as_of_ts = pd.Timestamp(window_end) - pd.Timedelta(days=lag_days)
            window_cases = window_cases.copy()
            window_cases["as_of_date"] = as_of_ts

        # 4. Build case node features
        # Returns: {x: [N_cases, case_feat_dim], strain_emb: ..., batch: [N_cases]}
        case_graph = self.graph_builder.build_case_graph(
            cases=window_cases,
            window_start=pd.Timestamp(window_start),
            location_index=self.location_index,
        )

        # 5. Build location features via subclass method (raw counts)
        # [num_locations, loc_feat_dim]
        location_features = self._build_location_features(window_start, window_end)

        # 5b. Build population tensor and optionally normalize count-valued
        # feature columns BEFORE the location graph is constructed, so the
        # graph sees the same (normalized) feature values that flow to the
        # model. When the flag is off, `population` is a no-op tensor of
        # per_capita_base so downstream code can be uniform.
        n_locs = len(self.location_index)
        if self.per_capita_normalize and self.location_population:
            pop_list = [
                float(self.location_population.get(loc_key, self.per_capita_base))
                for loc_key, _loc_idx in sorted(
                    self.location_index.items(), key=lambda kv: kv[1]
                )
            ]
            population = torch.tensor(pop_list, dtype=torch.float32)  # [N_locs]
            # Defensive: guard against zero populations
            scale = (population / self.per_capita_base).clamp(min=1e-6)  # [N_locs]
            location_features = location_features.clone()
            feat_slice = self._count_feature_slice
            location_features[:, feat_slice] = (
                location_features[:, feat_slice] / scale.unsqueeze(1)
            )
        else:
            population = torch.full(
                (n_locs,), self.per_capita_base, dtype=torch.float32
            )
            scale = None

        # 6. Wrap location features for the model
        location_graph = self.graph_builder.build_location_graph(
            location_features=location_features,
        )

        # 7. Get targets and mask via subclass methods (raw counts)
        forecast_horizons: list[int] = self.data_cfg["forecast_horizons"]
        # [num_locations, len(forecast_horizons)]
        targets = self._get_targets(window_end, forecast_horizons)
        # [num_locations]
        valid_mask = self._get_mask(window_end, forecast_horizons)

        # 7b. Normalize targets to rate space when per-capita flag is on
        if scale is not None:
            targets = targets / scale.unsqueeze(1)

        # 7c. Optional y_hist emission for fusion-horizon persistence anchor.
        # Kept OUT of the batch dict unless both the master flag and the
        # persistence sub-flag are enabled, so the legacy schema
        # (Dataset Output Contract) stays bit-exact for existing experiments.
        fh_cfg = self.cfg.get("model", {}).get("fusion_horizon", {})
        emit_y_hist = bool(fh_cfg.get("enabled", False)) and bool(
            fh_cfg.get("persistence", {}).get("enabled", False)
        )
        y_hist: Optional[torch.Tensor] = None
        if emit_y_hist:
            K = int(fh_cfg["persistence"].get("history_window", 8))
            y_hist = self._get_y_hist(window_end, K)  # [N_locs, K]
            if scale is not None:
                y_hist = y_hist / scale.unsqueeze(1)

        # 8. Return contract-compliant dict
        batch: dict[str, Any] = {
            "case_graph": {
                "x": case_graph["x"],                       # [N_cases, case_feat_dim]
                "strain_emb": case_graph.get("strain_emb"),  # [N_cases, E_strain] or None
                "batch": case_graph["batch"],                # [N_cases]
            },
            "location_graph": {
                "x": location_graph["x"],                    # [N_locs, loc_feat_dim]
            },
            "targets": targets,        # [N_locs, H]
            "mask": valid_mask,        # [N_locs]
            "population": population,  # [N_locs] (no-op tensor when flag is off)
            "metadata": {
                "time_index": idx,
                "dataset_name": self.dataset_name,
            },
        }
        if y_hist is not None:
            batch["y_hist"] = y_hist  # [N_locs, K]
        return batch
