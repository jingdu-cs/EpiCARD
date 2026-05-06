"""AIV (Avian Influenza Virus) dataset loader for HierEpiGNN.

Loads and preprocesses HPAI surveillance data from multiple sources:
- case_data.csv: individual genomic samples with host/strain metadata
- hpai_backyard.csv: domestic poultry outbreak incidence (with coords)
- hpai-wild-birds.csv: wild bird detection incidence (no coords)
- abundance/*.json: species abundance by location (52 features)

Targets are derived from incidence data (poultry + wild bird), NOT
from case_data sample counts.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import torch
from torch import Tensor

from src.data.base_dataset import EpidemicDataset

logger = logging.getLogger(__name__)

# Number of abundance feature files
_NUM_ABUNDANCE_FILES = 52


class AIVDataset(EpidemicDataset):
    """Avian Influenza dataset implementing the EpidemicDataset interface.

    Attributes:
        poultry_weekly: DataFrame with columns [loc_key, week_start, count]
            Weekly aggregated poultry outbreak counts per location.
        wildbird_weekly: DataFrame with columns [loc_key, week_start, count]
            Weekly aggregated wild bird detection counts per location.
        abundance_matrix: Tensor[num_locations, 52]
            Species abundance features per location.
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg, dataset_name="aiv")

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    def _load_raw_data(self) -> None:
        """Load all AIV data sources and build location index."""
        aiv_cfg = self.dataset_cfg

        # 1. Load case_data.csv
        self._load_case_data(aiv_cfg["case_data"])

        # 2. Load and aggregate poultry incidence
        self._load_poultry_incidence(aiv_cfg["incidence_poultry"])

        # 3. Load and aggregate wild bird incidence
        self._load_wildbird_incidence(aiv_cfg["incidence_wildbird"])

        # 4. Load abundance data
        self._load_abundance(aiv_cfg["abundance_dir"])

        # 5. Build unified location index from all sources
        self._build_location_index()

        # 6. Rebuild abundance matrix with final location index
        self._build_abundance_matrix()

        logger.info(
            "AIV data loaded: %d cases, %d locations, %d poultry weekly records, "
            "%d wildbird weekly records",
            len(self.cases_df),
            len(self.location_index),
            len(self.poultry_weekly),
            len(self.wildbird_weekly),
        )

    def _build_location_features(
        self, window_start: datetime, window_end: datetime
    ) -> Tensor:
        """Build location feature vectors for a time window.

        Features per location:
        - Poultry weekly incidence over lookback period: [lookback_weeks]
        - Wild bird weekly incidence over lookback period: [lookback_weeks]
        - Abundance values: [52]

        Returns:
            Tensor of shape [num_locations, 2 * lookback_weeks + 52].
        """
        lookback_weeks: int = self.data_cfg["incidence_lookback_weeks"]
        n_locs = len(self.location_index)
        feat_dim = 2 * lookback_weeks + _NUM_ABUNDANCE_FILES

        features = torch.zeros(
            n_locs, feat_dim, dtype=torch.float32
        )  # [N_locs, feat_dim]

        # Compute weekly incidence for each week in lookback period
        for w in range(lookback_weeks):
            week_start = window_end - timedelta(days=(lookback_weeks - w) * 7)
            week_end = week_start + timedelta(days=7)

            # Poultry incidence for this week
            poultry_counts = self._get_weekly_counts(
                self.poultry_weekly, week_start, week_end
            )  # dict[loc_key -> count]

            # Wild bird incidence for this week
            wildbird_counts = self._get_weekly_counts(
                self.wildbird_weekly, week_start, week_end
            )  # dict[loc_key -> count]

            for loc_key, loc_idx in self.location_index.items():
                features[loc_idx, w] = poultry_counts.get(loc_key, 0.0)
                features[loc_idx, lookback_weeks + w] = wildbird_counts.get(
                    loc_key, 0.0
                )

        # Abundance features (already aligned to location_index)
        features[:, 2 * lookback_weeks :] = self.abundance_matrix  # [N_locs, 52]

        return features  # [N_locs, feat_dim]

    def _get_targets(
        self, window_end: datetime, horizons: list[int]
    ) -> Tensor:
        """Compute incidence targets for each location at each horizon.

        For each horizon h, counts total incidence (poultry + wild bird) in
        [window_end, window_end + h*7 days).

        Returns:
            Tensor of shape [num_locations, len(horizons)].
        """
        n_locs = len(self.location_index)
        n_horizons = len(horizons)
        targets = torch.zeros(
            n_locs, n_horizons, dtype=torch.float32
        )  # [N_locs, H]

        for h_idx, h in enumerate(horizons):
            horizon_end = window_end + timedelta(days=h * 7)

            poultry_counts = self._get_period_counts(
                self.poultry_weekly, window_end, horizon_end
            )
            wildbird_counts = self._get_period_counts(
                self.wildbird_weekly, window_end, horizon_end
            )

            for loc_key, loc_idx in self.location_index.items():
                total = poultry_counts.get(loc_key, 0.0) + wildbird_counts.get(
                    loc_key, 0.0
                )
                targets[loc_idx, h_idx] = total

        return targets  # [N_locs, H]

    def _get_mask(
        self, window_end: datetime, horizons: list[int]
    ) -> Tensor:
        """Compute validity mask for each location.

        A location is valid (1.0) if:
        - It has any incidence data at all (in either poultry or wild bird)
        - Future data exists for all horizons

        Returns:
            Tensor of shape [num_locations] with 1.0 for valid, 0.0 for invalid.
        """
        n_locs = len(self.location_index)
        mask = torch.zeros(n_locs, dtype=torch.float32)  # [N_locs]

        # Locations that have ANY incidence data
        locs_with_data: set[tuple[str, str]] = set()
        if len(self.poultry_weekly) > 0:
            locs_with_data.update(self.poultry_weekly["loc_key"].unique())
        if len(self.wildbird_weekly) > 0:
            locs_with_data.update(self.wildbird_weekly["loc_key"].unique())

        # Check that future data exists for all horizons
        max_horizon = max(horizons)
        max_horizon_end = window_end + timedelta(days=max_horizon * 7)

        # Get combined incidence dates
        future_dates: set[tuple[str, str]] = set()
        for df in [self.poultry_weekly, self.wildbird_weekly]:
            if len(df) > 0:
                future_mask = (df["week_start"] >= window_end) & (
                    df["week_start"] < max_horizon_end
                )
                future_locs = df.loc[future_mask, "loc_key"].unique()
                future_dates.update(future_locs)

        for loc_key, loc_idx in self.location_index.items():
            if loc_key in locs_with_data and loc_key in future_dates:
                mask[loc_idx] = 1.0

        return mask  # [N_locs]

    def _get_y_hist(self, window_end: datetime, K: int) -> Tensor:
        """Recent K observed weekly counts per location (no future leakage).

        AIV targets are built from weekly poultry + wild-bird incidence tables.
        This method exposes the same recent observed signal to the persistence
        anchor: slot ``K-1`` is the most recent weekly bin ending at
        ``window_end`` and slot 0 is the oldest.

        Returns:
            Tensor of shape [num_locations, K].
        """
        n_locs = len(self.location_index)
        hist = torch.zeros(n_locs, K, dtype=torch.float32)

        hist_start = window_end - timedelta(days=K * self.days_per_step)
        for weekly_df in (self.poultry_weekly, self.wildbird_weekly):
            if len(weekly_df) == 0:
                continue

            mask = (
                (weekly_df["week_start"] >= hist_start)
                & (weekly_df["week_start"] < window_end)
            )
            window_inc = weekly_df.loc[mask]
            if window_inc.empty:
                continue

            for _, row in window_inc.iterrows():
                loc_idx = self.location_index.get(row["loc_key"])
                if loc_idx is None:
                    continue
                days_before_end = (window_end - row["week_start"]).days
                slot = K - 1 - (days_before_end // self.days_per_step)
                if 0 <= slot < K:
                    hist[loc_idx, slot] += float(row["count"])

        return hist  # [N_locs, K]

    # ------------------------------------------------------------------
    # Private helpers: data loading
    # ------------------------------------------------------------------

    def _load_case_data(self, path: str) -> None:
        """Load case_data.csv with date parsing and location normalization."""
        logger.info("Loading AIV case data from %s", path)
        df = pd.read_csv(path)

        # Parse Collection_Date (mixed formats: DD/M/YYYY and partial ISO)
        df["parsed_date"] = pd.to_datetime(
            df["Collection_Date"], format="mixed", dayfirst=True, errors="coerce"
        )

        # Drop rows with unparseable dates (NaT)
        n_nat = df["parsed_date"].isna().sum()
        if n_nat > 0:
            logger.warning("Dropped %d rows with unparseable Collection_Date", n_nat)
            df = df.dropna(subset=["parsed_date"]).copy()

        # Strip whitespace from State
        df["State"] = df["State"].astype(str).str.strip()

        # Replace "Nan" string with None
        df["State"] = df["State"].replace("Nan", None)
        df["County"] = df["County"].astype(str).replace("Nan", None)

        # Normalize locations
        normalized = df.apply(
            lambda row: self.location_normalizer.normalize(
                row["State"], row["County"]
            ),
            axis=1,
            result_type="expand",
        )
        df["state"] = normalized[0]
        df["county"] = normalized[1]

        # Build loc_key as tuple for indexing
        df["loc_key"] = list(
            zip(
                df["state"].fillna("unknown"),
                df["county"].fillna("unknown"),
            )
        )

        # Filter out excluded categories (e.g., mammal, human)
        excluded = self.dataset_cfg.get("excluded_categories", [])
        if excluded:
            mask = df["Category"].isin(excluded)
            n_excluded = mask.sum()
            if n_excluded > 0:
                logger.info(
                    "Excluding %d cases with Category in %s",
                    n_excluded, excluded,
                )
                df = df[~mask].copy()

        self.cases_df = df
        logger.info(
            "Loaded %d cases, date range: %s to %s",
            len(df),
            df["parsed_date"].min().date(),
            df["parsed_date"].max().date(),
        )

    def _load_poultry_incidence(self, path: str) -> None:
        """Load hpai_backyard.csv, aggregate weekly, extract coords."""
        logger.info("Loading poultry incidence from %s", path)
        df = pd.read_csv(path, encoding="utf-8-sig")

        # Parse date (MM/DD/YYYY HH:MM:SS AM/PM format)
        df["parsed_date"] = pd.to_datetime(
            df["Confirmed"], format="%m/%d/%Y %I:%M:%S %p"
        )

        # Normalize locations
        normalized = df.apply(
            lambda row: self.location_normalizer.normalize(
                row["State"], row["County Name"]
            ),
            axis=1,
            result_type="expand",
        )
        df["state"] = normalized[0]
        df["county"] = normalized[1]
        df["loc_key"] = list(
            zip(
                df["state"].fillna("unknown"),
                df["county"].fillna("unknown"),
            )
        )

        # Aggregate to weekly counts per location
        df["week_start"] = df["parsed_date"].dt.to_period("W-SAT").apply(
            lambda p: p.start_time
        )
        weekly = (
            df.groupby(["loc_key", "week_start"])
            .size()
            .reset_index(name="count")
        )
        weekly["count"] = weekly["count"].astype(float)
        self.poultry_weekly = weekly

        logger.info(
            "Poultry incidence: %d records -> %d weekly aggregates",
            len(df),
            len(weekly),
        )

    def _load_wildbird_incidence(self, path: str) -> None:
        """Load hpai-wild-birds.csv and aggregate weekly."""
        logger.info("Loading wild bird incidence from %s", path)
        df = pd.read_csv(path)

        # Parse date (MM/DD/YYYY format, with possible non-date values like "Unknown")
        df["parsed_date"] = pd.to_datetime(
            df["Collection Date"], format="mixed", dayfirst=False, errors="coerce"
        )

        # Drop rows with unparseable dates (NaT)
        n_nat = df["parsed_date"].isna().sum()
        if n_nat > 0:
            logger.warning(
                "Dropped %d wild bird rows with unparseable Collection Date", n_nat
            )
            df = df.dropna(subset=["parsed_date"]).copy()

        # Normalize locations
        normalized = df.apply(
            lambda row: self.location_normalizer.normalize(
                row["State"], row["County"]
            ),
            axis=1,
            result_type="expand",
        )
        df["state"] = normalized[0]
        df["county"] = normalized[1]
        df["loc_key"] = list(
            zip(
                df["state"].fillna("unknown"),
                df["county"].fillna("unknown"),
            )
        )

        # Aggregate to weekly counts per location
        df["week_start"] = df["parsed_date"].dt.to_period("W-SAT").apply(
            lambda p: p.start_time
        )
        weekly = (
            df.groupby(["loc_key", "week_start"])
            .size()
            .reset_index(name="count")
        )
        weekly["count"] = weekly["count"].astype(float)
        self.wildbird_weekly = weekly

        logger.info(
            "Wild bird incidence: %d records -> %d weekly aggregates",
            len(df),
            len(weekly),
        )

    def _load_abundance(self, abundance_dir: str) -> None:
        """Load all 52 abundance JSON files and normalize keys.

        Stores raw data in self._abundance_raw as list of dicts
        mapping (state, county) -> float.
        """
        logger.info("Loading abundance data from %s", abundance_dir)
        abundance_path = Path(abundance_dir)
        self._abundance_raw: list[dict[tuple[str, str], float]] = []

        for i in range(1, _NUM_ABUNDANCE_FILES + 1):
            filepath = abundance_path / f"location_abundance_{i}.json"
            with open(filepath, "r") as f:
                raw_data: dict[str, float] = json.load(f)

            # Normalize keys: "state|county" -> (normalized_state, normalized_county)
            normalized: dict[tuple[str, str], float] = {}
            for key, value in raw_data.items():
                loc_key = self.location_normalizer.from_abundance_key(key)
                normalized[loc_key] = value

            self._abundance_raw.append(normalized)

        logger.info("Loaded %d abundance feature files", len(self._abundance_raw))

    # ------------------------------------------------------------------
    # Private helpers: index and matrix building
    # ------------------------------------------------------------------

    def _build_location_index(self) -> None:
        """Build unified location index from all data sources."""
        all_locations: set[tuple[str, str]] = set()

        # From case_data
        all_locations.update(self.cases_df["loc_key"].unique())

        # From poultry incidence
        all_locations.update(self.poultry_weekly["loc_key"].unique())

        # From wild bird incidence
        all_locations.update(self.wildbird_weekly["loc_key"].unique())

        # From abundance data
        for abundance_dict in self._abundance_raw:
            all_locations.update(abundance_dict.keys())

        # Sort for deterministic ordering
        sorted_locations = sorted(all_locations)
        self.location_index = {
            loc: idx for idx, loc in enumerate(sorted_locations)
        }

        logger.info(
            "Built location index with %d locations", len(self.location_index)
        )

    def _build_abundance_matrix(self) -> None:
        """Build abundance feature matrix aligned to location_index.

        Shape: [num_locations, 52]
        """
        n_locs = len(self.location_index)
        self.abundance_matrix = torch.zeros(
            n_locs, _NUM_ABUNDANCE_FILES, dtype=torch.float32
        )  # [N_locs, 52]

        for feat_idx, abundance_dict in enumerate(self._abundance_raw):
            for loc_key, value in abundance_dict.items():
                loc_idx = self.location_index.get(loc_key)
                if loc_idx is not None:
                    self.abundance_matrix[loc_idx, feat_idx] = value

        # Clean up raw data
        del self._abundance_raw

        logger.info(
            "Built abundance matrix: %s", list(self.abundance_matrix.shape)
        )

    # ------------------------------------------------------------------
    # Private helpers: incidence queries
    # ------------------------------------------------------------------

    @staticmethod
    def _get_weekly_counts(
        weekly_df: pd.DataFrame,
        week_start: datetime,
        week_end: datetime,
    ) -> dict[tuple[str, str], float]:
        """Get incidence counts for a single week from weekly aggregated data.

        Returns dict mapping loc_key -> total count in [week_start, week_end).
        """
        if len(weekly_df) == 0:
            return {}

        mask = (weekly_df["week_start"] >= week_start) & (
            weekly_df["week_start"] < week_end
        )
        filtered = weekly_df.loc[mask]

        if len(filtered) == 0:
            return {}

        return (
            filtered.groupby("loc_key")["count"]
            .sum()
            .to_dict()
        )

    @staticmethod
    def _get_period_counts(
        weekly_df: pd.DataFrame,
        period_start: datetime,
        period_end: datetime,
    ) -> dict[tuple[str, str], float]:
        """Get total incidence counts over a multi-week period.

        Returns dict mapping loc_key -> total count in [period_start, period_end).
        """
        if len(weekly_df) == 0:
            return {}

        mask = (weekly_df["week_start"] >= period_start) & (
            weekly_df["week_start"] < period_end
        )
        filtered = weekly_df.loc[mask]

        if len(filtered) == 0:
            return {}

        return (
            filtered.groupby("loc_key")["count"]
            .sum()
            .to_dict()
        )
