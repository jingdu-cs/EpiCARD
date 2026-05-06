"""Japan data processing pipeline.

Processes:
- case_data.csv                      (Nan cleanup, prefecture normalization)
- japan_confirmed.csv                (prefecture normalization)
- genetic_similarity_long_table.csv  (copy unchanged)
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from data_processing.utils import copy_file_unchanged, log_processing_stats

logger = logging.getLogger("data_processing")


def process_japan(
    raw_dir: Path,
    output_dir: Path,
    log_dir: Path,
    fuzzy_threshold: int = 85,
) -> dict:
    """Process all Japan data files.

    Parameters
    ----------
    raw_dir : Path
        ``data/japan/``
    output_dir : Path
        ``data/processed/japan/``
    log_dir : Path
        ``data/processed/logs/``
    fuzzy_threshold : int
        Unused for Japan (kept for interface consistency).

    Returns
    -------
    dict
        Combined processing report.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {}

    logger.info("=" * 60)
    logger.info("Processing Japan dataset")
    logger.info("=" * 60)

    # 1. case_data.csv
    report["case_data"] = _process_case_data(
        raw_dir / "case_data.csv",
        output_dir / "case_data.csv",
    )

    # 2. japan_confirmed.csv
    report["japan_confirmed"] = _process_japan_confirmed(
        raw_dir / "japan_confirmed.csv",
        output_dir / "japan_confirmed.csv",
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

def _process_case_data(raw_path: Path, output_path: Path) -> dict:
    """Clean Japan case_data.csv: replace literal Nan strings, normalize prefectures."""
    df = pd.read_csv(raw_path, dtype=str)
    original_cols = list(df.columns)
    total_rows = len(df)

    # Replace all literal "Nan" strings with actual NA
    df = df.replace("Nan", pd.NA)

    # Normalize State (prefecture names): lowercase + strip
    state_notna = df["State"].notna()
    df.loc[state_notna, "State"] = (
        df.loc[state_notna, "State"].str.strip().str.lower()
    )

    rows_with_state = int(state_notna.sum())
    nan_state_count = total_rows - rows_with_state
    unique_prefectures = int(df["State"].dropna().nunique())

    df_out = df[original_cols]

    # --- Validation ---
    _validate_no_nan_strings(df_out, ["Unique_Identifier", "Strain"], "case_data.csv")
    assert list(df_out.columns) == original_cols, "Column schema mismatch"

    df_out.to_csv(output_path, index=False)

    stats = {
        "total_rows": total_rows,
        "rows_kept": len(df_out),
        "rows_with_state": rows_with_state,
        "nan_state_count": nan_state_count,
        "unique_prefectures": unique_prefectures,
    }
    log_processing_stats(logger, "case_data.csv (Japan)", stats)
    return stats


# ---------------------------------------------------------------------------
# japan_confirmed.csv
# ---------------------------------------------------------------------------

def _process_japan_confirmed(raw_path: Path, output_path: Path) -> dict:
    """Normalize prefecture names in japan_confirmed.csv (wide format)."""
    df = pd.read_csv(raw_path, dtype=str)
    original_cols = list(df.columns)
    total_rows = len(df)

    # Normalize Prefecture: lowercase + strip
    df["Prefecture"] = df["Prefecture"].str.strip().str.lower()

    # --- Validation ---
    missing_pref = df["Prefecture"].isna() | (df["Prefecture"] == "")
    if missing_pref.any():
        raise ValueError(
            f"[japan_confirmed.csv] {missing_pref.sum()} rows with missing Prefecture"
        )

    dup_codes = df["prefectureCode"].duplicated()
    if dup_codes.any():
        raise ValueError(
            f"[japan_confirmed.csv] Duplicate prefectureCodes: "
            f"{df.loc[dup_codes, 'prefectureCode'].tolist()}"
        )

    assert list(df.columns) == original_cols, "Column schema mismatch"

    df.to_csv(output_path, index=False)

    # Count date columns (everything after the 3 metadata columns)
    date_columns = len(original_cols) - 3

    stats = {
        "total_rows": total_rows,
        "unique_prefectures": int(df["Prefecture"].nunique()),
        "date_columns": date_columns,
    }
    log_processing_stats(logger, "japan_confirmed.csv", stats)
    return stats


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_no_nan_strings(
    df: pd.DataFrame, columns: list[str], filename: str
) -> None:
    """Assert no literal 'nan'/'Nan' string values remain in essential columns."""
    for col in columns:
        vals = df[col].dropna().astype(str)
        nan_mask = vals.str.lower().isin(["nan"])
        if nan_mask.any():
            count = nan_mask.sum()
            raise ValueError(
                f"[{filename}] Column '{col}' still has {count} 'nan' string values"
            )
