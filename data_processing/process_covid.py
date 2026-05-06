"""COVID data processing pipeline.

Processes:
- case_data.csv                      (state/county normalization, US filtering)
- covid_confirmed.csv                (state/county normalization)
- genetic_similarity_long_table.csv  (copy unchanged)
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from data_processing.mapping import (
    _ALL_US_FULL_NAMES,
    _ALL_US_TERRITORY_NAMES,
    classify_state,
)
from data_processing.normalization import normalize_county, normalize_state
from data_processing.utils import copy_file_unchanged, log_processing_stats

logger = logging.getLogger("data_processing")


def process_covid(
    raw_dir: Path,
    output_dir: Path,
    log_dir: Path,
    fuzzy_threshold: int = 85,
) -> dict:
    """Process all COVID data files.

    Parameters
    ----------
    raw_dir : Path
        ``data/covid/``
    output_dir : Path
        ``data/processed/covid/``
    log_dir : Path
        ``data/processed/logs/``
    fuzzy_threshold : int
        Minimum score for fuzzy state matching.

    Returns
    -------
    dict
        Combined processing report.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {}

    logger.info("=" * 60)
    logger.info("Processing COVID dataset")
    logger.info("=" * 60)

    # 1. case_data.csv
    report["case_data"] = _process_case_data(
        raw_dir / "case_data.csv",
        output_dir / "case_data.csv",
        fuzzy_threshold,
    )

    # 2. covid_confirmed.csv
    report["covid_confirmed"] = _process_covid_confirmed(
        raw_dir / "covid_confirmed.csv",
        output_dir / "covid_confirmed.csv",
    )

    # 3. genetic_similarity_long_table.csv (copy unchanged)
    gs_src = raw_dir / "genetic_similarity_long_table.csv"
    if gs_src.exists():
        copy_file_unchanged(gs_src, output_dir / "genetic_similarity_long_table.csv", logger)
        report["genetic_similarity"] = "copied_unchanged"

    return report


# ---------------------------------------------------------------------------
# case_data.csv
# ---------------------------------------------------------------------------

def _process_case_data(
    raw_path: Path, output_path: Path, fuzzy_threshold: int
) -> dict:
    """Normalize and filter COVID case_data.csv to US-only."""
    df = pd.read_csv(raw_path, dtype=str)
    original_cols = list(df.columns)
    total_rows = len(df)

    classifications: list[str] = []
    resolved_states: list[str | None] = []

    for raw_state in df["State"]:
        cls, resolved = classify_state(str(raw_state), fuzzy_threshold=fuzzy_threshold)
        classifications.append(cls)
        resolved_states.append(resolved)

    df["_classification"] = classifications
    df["_resolved_state"] = resolved_states

    cls_counts = df["_classification"].value_counts().to_dict()

    # Keep US states and territories
    mask_keep = df["_classification"].isin(["us_state", "us_territory"])
    df_kept = df[mask_keep].copy()

    # Apply normalized state
    df_kept["State"] = df_kept["_resolved_state"]

    # Normalize county (replicates LocationNormalizer._normalize_county exactly)
    df_kept["County"] = df_kept["County"].apply(normalize_county)

    # Drop rows with missing county
    county_missing = df_kept["County"].isna() | (df_kept["County"] == "")
    dropped_county = county_missing.sum()
    df_kept = df_kept[~county_missing]

    # Restore original columns
    df_out = df_kept[original_cols]

    # --- Validation ---
    _validate_us_only(df_out, "State", "case_data.csv")
    _validate_no_nan_strings(df_out, ["State", "County"], "case_data.csv")
    assert list(df_out.columns) == original_cols, "Column schema mismatch"

    df_out.to_csv(output_path, index=False)

    stats = {
        "total_rows": total_rows,
        "rows_kept": len(df_out),
        "dropped_non_us": cls_counts.get("canadian", 0) + cls_counts.get("mexican", 0),
        "dropped_unknown_state": cls_counts.get("unknown", 0),
        "dropped_missing_county": int(dropped_county),
        "unique_states_after": int(df_out["State"].nunique()),
    }
    log_processing_stats(logger, "case_data.csv (COVID)", stats)
    return stats


# ---------------------------------------------------------------------------
# covid_confirmed.csv
# ---------------------------------------------------------------------------

def _process_covid_confirmed(raw_path: Path, output_path: Path) -> dict:
    """Normalize locations in covid_confirmed.csv (wide format)."""
    df = pd.read_csv(raw_path, dtype=str)
    original_cols = list(df.columns)
    total_rows = len(df)

    # Normalize State (2-letter abbreviations)
    df["State"] = df["State"].apply(normalize_state)

    # Normalize County Name (strip suffixes, trailing spaces)
    df["County Name"] = df["County Name"].apply(normalize_county)

    # Drop rows with missing state
    valid_mask = df["State"].notna() & (df["State"] != "")
    dropped = total_rows - valid_mask.sum()
    df_out = df[valid_mask].copy()

    # --- Validation ---
    _validate_us_only(df_out, "State", "covid_confirmed.csv")
    assert list(df_out.columns) == original_cols, "Column schema mismatch"

    df_out.to_csv(output_path, index=False)

    stats = {
        "total_rows": total_rows,
        "rows_kept": len(df_out),
        "dropped_invalid": int(dropped),
        "unique_states_after": int(df_out["State"].nunique()),
    }
    log_processing_stats(logger, "covid_confirmed.csv", stats)
    return stats


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

_VALID_US = _ALL_US_FULL_NAMES | _ALL_US_TERRITORY_NAMES


def _validate_us_only(df: pd.DataFrame, state_col: str, filename: str) -> None:
    """Assert every state value is a valid US state or territory."""
    unique_states = set(df[state_col].dropna().unique())
    non_us = unique_states - _VALID_US
    if non_us:
        raise ValueError(
            f"[{filename}] Non-US states found after processing: {non_us}"
        )


def _validate_no_nan_strings(
    df: pd.DataFrame, columns: list[str], filename: str
) -> None:
    """Assert no literal 'nan'/'Nan' string values remain."""
    for col in columns:
        vals = df[col].dropna().astype(str)
        nan_mask = vals.str.lower().isin(["nan"])
        if nan_mask.any():
            count = nan_mask.sum()
            raise ValueError(
                f"[{filename}] Column '{col}' still has {count} 'nan' string values"
            )
