"""Japan COVID-19 dataset loader for HierEpiGNN.

Loads case-level genomic data, genetic similarity (chunked),
and prefecture-level cumulative confirmed counts (wide format),
converting to incidence (daily or weekly) for targets and features.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta

import pandas as pd
import torch
from torch import Tensor

from src.data.base_dataset import EpidemicDataset

logger = logging.getLogger(__name__)



# ---------------------------------------------------------------------------
# 2020 Japan census figures (Statistics Bureau of Japan). Used for per-capita
# target normalization when data.per_capita_normalize is enabled. Keys match
# the normalized lowercase prefecture names in location_index.
# ---------------------------------------------------------------------------
_JAPAN_PREFECTURE_POPULATION: dict[str, int] = {
    "hokkaido": 5_224_614,
    "aomori": 1_237_984,
    "iwate": 1_210_534,
    "miyagi": 2_301_996,
    "akita": 959_502,
    "yamagata": 1_068_027,
    "fukushima": 1_833_152,
    "ibaraki": 2_867_009,
    "tochigi": 1_933_146,
    "gunma": 1_939_110,
    "saitama": 7_344_765,
    "chiba": 6_284_480,
    "tokyo": 14_047_594,
    "kanagawa": 9_237_337,
    "niigata": 2_201_272,
    "toyama": 1_034_814,
    "ishikawa": 1_132_526,
    "fukui": 766_863,
    "yamanashi": 809_974,
    "nagano": 2_048_011,
    "gifu": 1_978_742,
    "shizuoka": 3_633_202,
    "aichi": 7_542_415,
    "mie": 1_770_254,
    "shiga": 1_413_610,
    "kyoto": 2_578_087,
    "osaka": 8_837_685,
    "hyogo": 5_465_002,
    "nara": 1_324_473,
    "wakayama": 922_584,
    "tottori": 553_407,
    "shimane": 671_126,
    "okayama": 1_888_432,
    "hiroshima": 2_799_702,
    "yamaguchi": 1_342_059,
    "tokushima": 719_559,
    "kagawa": 950_244,
    "ehime": 1_334_841,
    "kochi": 691_527,
    "fukuoka": 5_135_214,
    "saga": 811_442,
    "nagasaki": 1_312_317,
    "kumamoto": 1_738_301,
    "oita": 1_123_852,
    "miyazaki": 1_069_576,
    "kagoshima": 1_588_256,
    "okinawa": 1_467_480,
}


class JapanDataset(EpidemicDataset):
    """Japan COVID-19 epidemic dataset.

    Data sources:
    - case_data.csv: individual sequenced samples with lineage/location
    - japan_confirmed.csv: cumulative prefecture-level confirmed counts (wide format)
    """

    def __init__(self, cfg: dict) -> None:
        # Populated by _load_raw_data before base class calls other methods
        self.incidence: pd.DataFrame  # columns: loc_key, date, count
        super().__init__(cfg, dataset_name="japan")

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    def _load_raw_data(self) -> None:
        """Load and preprocess all Japan COVID-19 data sources."""
        japan_cfg = self.dataset_cfg

        # 1. Load case_data.csv
        self._load_case_data(japan_cfg["case_data"])

        # 2. Load japan_confirmed.csv (wide -> long format incidence)
        self._load_incidence(japan_cfg["incidence"])

        # 4. Build location_index from union of case and incidence locations
        all_locs: set[tuple[str, str]] = set()

        case_loc_keys = self.cases_df["loc_key"].dropna().unique()
        for lk in case_loc_keys:
            if isinstance(lk, tuple) and len(lk) == 2:
                all_locs.add(lk)

        incidence_loc_keys = self.incidence["loc_key"].unique()
        for lk in incidence_loc_keys:
            if isinstance(lk, tuple) and len(lk) == 2:
                all_locs.add(lk)

        sorted_locs = sorted(all_locs)
        self.location_index = {loc: i for i, loc in enumerate(sorted_locs)}

        logger.info(
            "Japan location_index built: %d locations", len(self.location_index)
        )

        # 4b. Populate location_population from prefecture census figures
        self.location_population = {
            loc_key: _JAPAN_PREFECTURE_POPULATION[loc_key[0]]
            for loc_key in self.location_index
            if loc_key[0] in _JAPAN_PREFECTURE_POPULATION
        }
        missing = [
            loc
            for loc in self.location_index
            if loc not in self.location_population
        ]
        if missing:
            logger.warning(
                "Japan: %d location(s) without population data: %s — will use "
                "per_capita_base as default (no normalization effect)",
                len(missing), missing[:5],
            )
        logger.info(
            "Japan location_population built: %d entries",
            len(self.location_population),
        )

        # 5. Log date range diagnostics for debugging
        case_max = self.cases_df["parsed_date"].max()
        case_min = self.cases_df["parsed_date"].min()
        if not self.incidence.empty:
            inc_min = self.incidence["date"].min()
            inc_max = self.incidence["date"].max()
        else:
            inc_min = inc_max = None
        logger.info(
            "Japan date ranges: case_data=[%s, %s], incidence=[%s, %s]",
            case_min.date() if case_min is not None else "N/A",
            case_max.date() if case_max is not None else "N/A",
            inc_min.date() if inc_min is not None else "N/A",
            inc_max.date() if inc_max is not None else "N/A",
        )

        # 7. Log location overlap diagnostics
        case_locs = set(
            lk for lk in self.cases_df["loc_key"].dropna().unique()
            if isinstance(lk, tuple) and len(lk) == 2
        )
        inc_locs = set(
            lk for lk in self.incidence["loc_key"].unique()
            if isinstance(lk, tuple) and len(lk) == 2
        ) if not self.incidence.empty else set()
        overlap = case_locs & inc_locs
        logger.info(
            "Japan location overlap: case=%d, incidence=%d, overlap=%d",
            len(case_locs), len(inc_locs), len(overlap),
        )

    def _get_effective_end_date(self) -> datetime:
        """Bound time windows by the incidence data's date range.

        If the case data extends beyond the incidence data, time windows
        built from case data alone would produce windows with no future
        incidence data, leading to all-zero masks and NaN metrics.
        """
        case_max = self.cases_df["parsed_date"].max()
        if self.incidence.empty:
            return case_max
        inc_max = self.incidence["date"].max()
        effective = min(case_max, inc_max)
        if effective < case_max:
            logger.info(
                "Clamping time window end from %s (case_data) to %s (incidence) "
                "to ensure forecast targets are available.",
                case_max.date(), effective.date(),
            )
        return effective

    def _build_location_features(
        self, window_start: datetime, window_end: datetime
    ) -> Tensor:
        """Build location features: incidence over lookback period.

        Returns:
            Tensor of shape [num_locations, incidence_lookback].
        """
        lookback: int = self.data_cfg.get(
            "incidence_lookback",
            self.data_cfg.get("incidence_lookback_weeks", 8),
        )
        n_locs = len(self.location_index)

        # Compute lookback start
        lookback_start = window_end - timedelta(
            days=lookback * self.days_per_step
        )

        # Filter incidence data to lookback window
        mask = (
            (self.incidence["date"] >= lookback_start)
            & (self.incidence["date"] < window_end)
        )
        window_inc = self.incidence.loc[mask]

        # Build feature tensor  # [num_locations, lookback]
        features = torch.zeros(
            n_locs, lookback, dtype=torch.float32
        )  # [N_locs, lookback]

        if window_inc.empty:
            return features  # [N_locs, lookback]

        for _, row in window_inc.iterrows():
            loc_key = row["loc_key"]
            loc_idx = self.location_index.get(loc_key)
            if loc_idx is None:
                continue

            inc_date = row["date"]
            days_before_end = (window_end - inc_date).days
            slot = lookback - 1 - (days_before_end // self.days_per_step)

            if 0 <= slot < lookback:
                features[loc_idx, slot] += float(row["count"])

        return features  # [N_locs, lookback]

    def _get_targets(
        self, window_end: datetime, horizons: list[int]
    ) -> Tensor:
        """Compute incidence targets for each location at each horizon.

        For each horizon h, sum new cases in
        [window_end, window_end + h * days_per_step) per location.

        Returns:
            Tensor of shape [num_locations, len(horizons)].
        """
        n_locs = len(self.location_index)
        n_horizons = len(horizons)

        targets = torch.zeros(
            n_locs, n_horizons, dtype=torch.float32
        )  # [N_locs, H]

        for h_idx, h in enumerate(horizons):
            horizon_end = window_end + timedelta(
                days=h * self.days_per_step
            )

            mask = (
                (self.incidence["date"] >= window_end)
                & (self.incidence["date"] < horizon_end)
            )
            horizon_inc = self.incidence.loc[mask]

            if horizon_inc.empty:
                continue

            # Aggregate by loc_key
            grouped = horizon_inc.groupby("loc_key")["count"].sum()
            for loc_key, count in grouped.items():
                loc_idx = self.location_index.get(loc_key)
                if loc_idx is not None:
                    targets[loc_idx, h_idx] = float(count)

        return targets  # [N_locs, H]

    def _get_y_hist(self, window_end: datetime, K: int) -> Tensor:
        """Recent K observed daily/weekly counts per location (no future leakage).

        Step size follows ``self.days_per_step``. Slot ``K-1`` is the most
        recent step (ending at ``window_end``); slot 0 is the oldest.

        Returns:
            Tensor of shape [num_locations, K].
        """
        n_locs = len(self.location_index)
        hist = torch.zeros(n_locs, K, dtype=torch.float32)  # [N_locs, K]

        hist_start = window_end - timedelta(days=K * self.days_per_step)
        mask = (
            (self.incidence["date"] >= hist_start)
            & (self.incidence["date"] < window_end)
        )
        window_inc = self.incidence.loc[mask]
        if window_inc.empty:
            return hist

        for _, row in window_inc.iterrows():
            loc_idx = self.location_index.get(row["loc_key"])
            if loc_idx is None:
                continue
            days_before_end = (window_end - row["date"]).days
            slot = K - 1 - (days_before_end // self.days_per_step)
            if 0 <= slot < K:
                hist[loc_idx, slot] += float(row["count"])
        return hist  # [N_locs, K]

    def _get_mask(
        self, window_end: datetime, horizons: list[int]
    ) -> Tensor:
        """Compute validity mask for each location.

        1.0 for locations with historical data up to window_end
        AND future data exists for at least the shortest horizon.
        0.0 otherwise.

        Returns:
            Tensor of shape [num_locations] with 1.0/0.0 values.
        """
        n_locs = len(self.location_index)
        mask = torch.zeros(n_locs, dtype=torch.float32)  # [N_locs]

        # Check which locations have historical data up to window_end
        hist_mask = self.incidence["date"] < window_end
        locs_with_history = set(
            self.incidence.loc[hist_mask, "loc_key"].unique()
        )

        # Check which locations have future data
        min_horizon = min(horizons)
        future_end = window_end + timedelta(
            days=min_horizon * self.days_per_step
        )
        future_mask = (
            (self.incidence["date"] >= window_end)
            & (self.incidence["date"] < future_end)
        )
        locs_with_future = set(
            self.incidence.loc[future_mask, "loc_key"].unique()
        )

        for loc_key, loc_idx in self.location_index.items():
            if loc_key in locs_with_history and loc_key in locs_with_future:
                mask[loc_idx] = 1.0

        return mask  # [N_locs]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_case_data(self, path: str) -> None:
        """Load case_data.csv, normalize locations, add parsed_date and loc_key.

        In the Japan dataset, locations are identified by prefecture (State column)
        with County always NaN. The loc_key is (prefecture, "unknown").
        """
        logger.info("Loading Japan case_data from %s", path)
        df = pd.read_csv(path)
        logger.info("Japan case_data loaded: %d rows", len(df))

        # Parse Collection_Date (D/M/YYYY format)
        df["parsed_date"] = pd.to_datetime(
            df["Collection_Date"], format="%d/%m/%Y", errors="coerce"
        )
        n_null_dates = df["parsed_date"].isna().sum()
        if n_null_dates > 0:
            logger.warning(
                "Dropped %d rows with unparseable Collection_Date", n_null_dates
            )
            df = df.dropna(subset=["parsed_date"]).copy()

        # Normalize locations
        norm_states = []
        norm_counties = []
        for _, row in df.iterrows():
            raw_state = str(row.get("State", ""))
            raw_county = str(row.get("County", ""))
            ns, nc = self.location_normalizer.normalize(raw_state, raw_county)
            norm_states.append(ns or "unknown")
            norm_counties.append(nc or "unknown")

        df["state"] = norm_states
        df["county"] = norm_counties
        df["loc_key"] = list(zip(df["state"], df["county"]))

        # Ensure required columns for base class
        if "Category" not in df.columns:
            df["Category"] = "human"
        if "Subcategory" not in df.columns:
            df["Subcategory"] = "Nan"
        if "Strain" not in df.columns:
            # Use lineage column if available
            strain_col = None
            for col in ("Strain", "Lineage", "lineage"):
                if col in df.columns:
                    strain_col = col
                    break
            if strain_col and strain_col != "Strain":
                df["Strain"] = df[strain_col]
            elif "Strain" not in df.columns:
                df["Strain"] = ""

        # Fill NaN in string columns
        for col in ("Category", "Subcategory", "Strain", "sequence_Id"):
            if col in df.columns:
                df[col] = df[col].fillna("").astype(str)

        self.cases_df = df
        logger.info(
            "Japan case_data processed: %d rows, %d unique locations",
            len(df),
            df["loc_key"].nunique(),
        )

    def _load_incidence(self, path: str) -> None:
        """Load japan_confirmed.csv (wide format), convert cumulative to incidence.

        The wide format has columns:
        prefectureCode, Prefecture, Country, <date_1>, <date_2>, ...
        where date columns are YYYY-MM-DD and values are cumulative counts.

        When ``days_per_step == 1`` (daily mode) every date column is used;
        when ``days_per_step == 7`` (weekly mode) every 7th date is sampled.
        """
        logger.info("Loading Japan confirmed incidence from %s", path)
        df = pd.read_csv(path)
        logger.info("Japan confirmed loaded: %d rows", len(df))

        # Identify date columns by regex YYYY-MM-DD
        date_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
        date_cols = [c for c in df.columns if date_pattern.match(c)]
        date_cols_sorted = sorted(date_cols)

        logger.info(
            "Found %d date columns spanning %s to %s",
            len(date_cols_sorted),
            date_cols_sorted[0] if date_cols_sorted else "N/A",
            date_cols_sorted[-1] if date_cols_sorted else "N/A",
        )

        # Parse date column values to datetime
        date_values = [pd.Timestamp(d) for d in date_cols_sorted]

        # Select date columns based on time resolution
        if self.days_per_step == 1:
            selected_dates = date_values
            selected_cols = date_cols_sorted
        else:
            # Weekly: pick every 7th date
            idx = list(range(0, len(date_cols_sorted), 7))
            selected_dates = [date_values[i] for i in idx]
            selected_cols = [date_cols_sorted[i] for i in idx]

        # Normalize locations per row and compute new cases by differencing
        rows_long: list[dict] = []

        for _, row in df.iterrows():
            prefecture = str(row.get("Prefecture", ""))
            loc_key = (prefecture.lower().strip(), "unknown")

            # Compute new cases by differencing consecutive cumulative values
            cumulative_vals = []
            for col in selected_cols:
                val = row.get(col, 0)
                cumulative_vals.append(float(val) if pd.notna(val) else 0.0)

            for i in range(1, len(cumulative_vals)):
                count = max(0.0, cumulative_vals[i] - cumulative_vals[i - 1])
                rows_long.append({
                    "loc_key": loc_key,
                    "date": selected_dates[i - 1],
                    "count": count,
                })

        self.incidence = pd.DataFrame(rows_long)

        # Aggregate duplicate (loc_key, date) pairs
        if not self.incidence.empty:
            self.incidence = (
                self.incidence
                .groupby(["loc_key", "date"], as_index=False)["count"]
                .sum()
            )

        time_unit = self.data_cfg.get("time_unit", "weeks")
        logger.info(
            "Japan incidence processed: %d %s records, %d unique locations",
            len(self.incidence),
            time_unit,
            self.incidence["loc_key"].nunique() if not self.incidence.empty else 0,
        )

