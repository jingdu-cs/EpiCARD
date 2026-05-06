"""AIV data processing pipeline.

Processes:
- case_data.csv          (country filtering, state/county normalization)
- hpai_backyard.csv      (state/county normalization)
- hpai-wild-birds.csv    (state/county normalization)
- abundance/*.json       (US-only filtering, key normalization)
- genetic_similarity_matrix.csv  (copy unchanged)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from data_processing.mapping import (
    _ALL_US_FULL_NAMES,
    _ALL_US_TERRITORY_NAMES,
    classify_state,
    resolve_us_state,
)
from data_processing.normalization import (
    normalize_abundance_key,
    normalize_county,
    normalize_state,
)
from data_processing.utils import copy_file_unchanged, log_processing_stats

logger = logging.getLogger("data_processing")


def process_aiv(
    raw_dir: Path,
    output_dir: Path,
    log_dir: Path,
    fuzzy_threshold: int = 85,
) -> dict:
    """Process all AIV data files.

    Parameters
    ----------
    raw_dir : Path
        ``data/aiv/``
    output_dir : Path
        ``data/processed/aiv/``
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
    logger.info("Processing AIV dataset")
    logger.info("=" * 60)

    # 1. case_data.csv
    report["case_data"] = _process_case_data(
        raw_dir / "case_data.csv",
        output_dir / "case_data.csv",
        fuzzy_threshold,
    )

    # 2. hpai_backyard.csv
    report["hpai_backyard"] = _process_hpai_backyard(
        raw_dir / "hpai_backyard.csv",
        output_dir / "hpai_backyard.csv",
    )

    # 3. hpai-wild-birds.csv
    report["hpai_wild_birds"] = _process_hpai_wildbirds(
        raw_dir / "hpai-wild-birds.csv",
        output_dir / "hpai-wild-birds.csv",
    )

    # 4. abundance/*.json
    report["abundance"] = _process_abundance(
        raw_dir / "abundance",
        output_dir / "abundance",
    )

    # 5. genetic_similarity_matrix.csv (copy unchanged)
    gs_src = raw_dir / "genetic_similarity_matrix.csv"
    if gs_src.exists():
        copy_file_unchanged(gs_src, output_dir / "genetic_similarity_matrix.csv", logger)
        report["genetic_similarity_matrix"] = "copied_unchanged"

    return report


# ---------------------------------------------------------------------------
# case_data.csv
# ---------------------------------------------------------------------------

def _process_case_data(
    raw_path: Path, output_path: Path, fuzzy_threshold: int
) -> dict:
    """Filter to US-only and normalize locations in AIV case_data.csv."""
    df = pd.read_csv(raw_path, dtype=str)
    original_cols = list(df.columns)
    total_rows = len(df)

    classifications: list[str] = []
    resolved_states: list[str | None] = []
    fuzzy_matched: list[str] = []

    for raw_state in df["State"]:
        cls, resolved = classify_state(str(raw_state), fuzzy_threshold=fuzzy_threshold)
        classifications.append(cls)
        resolved_states.append(resolved)

    df["_classification"] = classifications
    df["_resolved_state"] = resolved_states

    # Count by classification
    cls_counts = df["_classification"].value_counts().to_dict()

    # Keep only US states and territories
    mask_keep = df["_classification"].isin(["us_state", "us_territory"])
    df_kept = df[mask_keep].copy()

    # Apply normalized state
    df_kept["State"] = df_kept["_resolved_state"]

    # Normalize county
    df_kept["County"] = df_kept["County"].apply(normalize_county)

    # Drop rows with missing county
    county_missing_mask = df_kept["County"].isna() | (df_kept["County"] == "")
    dropped_county = county_missing_mask.sum()
    df_kept = df_kept[~county_missing_mask]

    # Drop helper columns and restore original column order
    df_out = df_kept[original_cols]

    # --- Validation ---
    _validate_us_only(df_out, "State", "case_data.csv")
    _validate_no_nan_strings(df_out, ["State", "County"], "case_data.csv")
    assert list(df_out.columns) == original_cols, "Column schema mismatch"

    df_out.to_csv(output_path, index=False)

    stats = {
        "total_rows": total_rows,
        "rows_kept": len(df_out),
        "dropped_canadian": cls_counts.get("canadian", 0),
        "dropped_mexican": cls_counts.get("mexican", 0),
        "dropped_unknown": cls_counts.get("unknown", 0),
        "dropped_territory": cls_counts.get("us_territory", 0),
        "dropped_missing_county": int(dropped_county),
        "unique_states_after": int(df_out["State"].nunique()),
    }
    log_processing_stats(logger, "case_data.csv", stats)
    return stats


# ---------------------------------------------------------------------------
# hpai_backyard.csv
# ---------------------------------------------------------------------------

def _process_hpai_backyard(raw_path: Path, output_path: Path) -> dict:
    """Normalize locations in hpai_backyard.csv (already US-only)."""
    df = pd.read_csv(raw_path, dtype=str)
    original_cols = list(df.columns)
    total_rows = len(df)

    # Normalize state
    df["State"] = df["State"].apply(normalize_state)

    # Normalize county (column is "County Name")
    df["County Name"] = df["County Name"].apply(normalize_county)

    # Drop rows with missing state or county
    valid_mask = (
        df["State"].notna()
        & (df["State"] != "")
        & df["County Name"].notna()
        & (df["County Name"] != "")
    )
    dropped = total_rows - valid_mask.sum()
    df_out = df[valid_mask].copy()

    # --- Validation ---
    _validate_us_only(df_out, "State", "hpai_backyard.csv")
    _validate_no_nan_strings(df_out, ["State", "County Name"], "hpai_backyard.csv")
    assert list(df_out.columns) == original_cols, "Column schema mismatch"

    df_out.to_csv(output_path, index=False)

    stats = {
        "total_rows": total_rows,
        "rows_kept": len(df_out),
        "dropped_invalid": int(dropped),
        "unique_states_after": int(df_out["State"].nunique()),
    }
    log_processing_stats(logger, "hpai_backyard.csv", stats)
    return stats


# ---------------------------------------------------------------------------
# hpai-wild-birds.csv
# ---------------------------------------------------------------------------

def _process_hpai_wildbirds(raw_path: Path, output_path: Path) -> dict:
    """Normalize locations in hpai-wild-birds.csv."""
    df = pd.read_csv(raw_path, dtype=str)
    original_cols = list(df.columns)
    total_rows = len(df)

    # Normalize state and county
    df["State"] = df["State"].apply(normalize_state)
    df["County"] = df["County"].apply(normalize_county)

    # Drop rows with missing state or county
    valid_mask = (
        df["State"].notna()
        & (df["State"] != "")
        & df["County"].notna()
        & (df["County"] != "")
    )
    dropped = total_rows - valid_mask.sum()
    df_out = df[valid_mask].copy()

    # --- Validation ---
    _validate_us_only(df_out, "State", "hpai-wild-birds.csv")
    _validate_no_nan_strings(df_out, ["State", "County"], "hpai-wild-birds.csv")
    assert list(df_out.columns) == original_cols, "Column schema mismatch"

    df_out.to_csv(output_path, index=False)

    stats = {
        "total_rows": total_rows,
        "rows_kept": len(df_out),
        "dropped_invalid": int(dropped),
        "unique_states_after": int(df_out["State"].nunique()),
    }
    log_processing_stats(logger, "hpai-wild-birds.csv", stats)
    return stats


# ---------------------------------------------------------------------------
# abundance/*.json
# ---------------------------------------------------------------------------

def _process_abundance(raw_dir: Path, output_dir: Path) -> dict:
    """Filter abundance JSONs to US-only entries and normalize keys."""
    output_dir.mkdir(parents=True, exist_ok=True)

    total_keys_original = 0
    total_keys_kept = 0
    files_processed = 0
    dropped_states: set[str] = set()

    json_files = sorted(raw_dir.glob("location_abundance_*.json"))
    for jf in json_files:
        with open(jf, "r") as f:
            data = json.load(f)

        total_keys_original += len(data)
        cleaned: dict[str, float] = {}

        for key, value in data.items():
            state, county = normalize_abundance_key(key)
            # Use classify_state to handle aliases (e.g., "us virgin islands")
            cls, resolved = classify_state(state)
            if cls in ("us_state", "us_territory") and resolved is not None:
                new_key = f"{resolved}|{county}"
                cleaned[new_key] = value
            else:
                dropped_states.add(state)

        total_keys_kept += len(cleaned)
        files_processed += 1

        with open(output_dir / jf.name, "w") as f:
            json.dump(cleaned, f)

    stats = {
        "files_processed": files_processed,
        "total_keys_original": total_keys_original,
        "total_keys_kept": total_keys_kept,
        "total_keys_dropped": total_keys_original - total_keys_kept,
        "dropped_states": sorted(dropped_states),
    }
    log_processing_stats(logger, "abundance/*.json", stats)
    return stats


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

_VALID_US_STATES_WITH_TERRITORIES = _ALL_US_FULL_NAMES | _ALL_US_TERRITORY_NAMES


def _validate_us_only(df: pd.DataFrame, state_col: str, filename: str) -> None:
    """Assert every state value is a valid US state or territory."""
    unique_states = set(df[state_col].dropna().unique())
    non_us = unique_states - _VALID_US_STATES_WITH_TERRITORIES
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
