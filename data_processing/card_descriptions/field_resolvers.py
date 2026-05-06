"""Derived-field resolvers for sample records.

Each resolver is a pure function that computes one derived field from raw
CSV values.  :func:`resolve_all_fields` orchestrates them into a complete
:class:`SampleRecord`.
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime
from typing import Optional, Tuple

import pandas as pd

from data_processing.card_descriptions.schema import SampleRecord
from data_processing.mapping import resolve_us_state

logger = logging.getLogger("data_processing")


# ======================================================================
# Date parsing
# ======================================================================


def parse_date(raw: str) -> Optional[date]:
    """Parse a date string in DD/M/YYYY format.

    Falls back to :func:`pandas.to_datetime` with ``dayfirst=True`` for
    edge cases.  Returns ``None`` on failure.
    """
    if not raw or not isinstance(raw, str):
        return None
    s = raw.strip()
    if not s or s.lower() == "nan":
        return None

    # Primary: DD/M/YYYY  (day and month may be 1 or 2 digits)
    try:
        return datetime.strptime(s, "%d/%m/%Y").date()
    except ValueError:
        pass

    # Fallback: pandas flexible parser
    try:
        return pd.to_datetime(s, dayfirst=True).date()
    except Exception:
        logger.warning("Failed to parse date '%s'", s)
        return None


# ======================================================================
# Report lag
# ======================================================================


_LAG_THRESHOLDS: list[Tuple[int, str]] = [
    (7, "prompt"),
    (30, "moderate"),
    (90, "delayed"),
]


def compute_report_lag(
    collection: Optional[date], report: Optional[date],
) -> Tuple[Optional[int], Optional[str]]:
    """Compute report lag in days and classify quality.

    Returns
    -------
    (days, quality)
        *quality* is one of ``"prompt"``, ``"moderate"``, ``"delayed"``,
        ``"severely_delayed"``, or ``"negative_or_inconsistent"``.
        Returns ``(None, None)`` only when either date is missing.
    """
    if collection is None or report is None:
        return (None, None)

    days = (report - collection).days

    if days < 0:
        return (days, "negative_or_inconsistent")

    for threshold, label in _LAG_THRESHOLDS:
        if days < threshold:
            return (days, label)
    return (days, "severely_delayed")


# ======================================================================
# Strain resolution
# ======================================================================

# Regex for AIV subtype (HxNx or Hx)
_AIV_SUBTYPE_RE = re.compile(r"(H\d+N?\d*)", re.IGNORECASE)
# Regex for numeric clade (e.g., 2.3.4.4b)
_AIV_CLADE_RE = re.compile(r"\d+\.\d+")

# Known AIV clade tokens that are not numeric
_AIV_NAMED_CLADES = {"2.3.4.4b", "Am_nonGsGD", "EA_nonGsGD"}

# COVID root lineages that count as root_level
_COVID_ROOT_LINEAGES = {"A", "B", "B.1"}


def classify_strain_resolution(strain: str, dataset: str) -> str:
    """Classify how much strain information is available.

    AIV levels: ``"specific"`` | ``"partial"`` | ``"missing"``
    COVID levels: ``"specific"`` | ``"family_level"`` | ``"root_level"``
    | ``"missing"``
    """
    s = strain.strip() if isinstance(strain, str) else ""
    if not s or s.lower() in ("nan", "not reported", ""):
        return "missing"

    if dataset == "aiv":
        return _classify_aiv_strain(s)
    return _classify_covid_strain(s)


def _classify_aiv_strain(s: str) -> str:
    """AIV: specific (subtype + clade), partial (subtype only), missing."""
    has_subtype = bool(_AIV_SUBTYPE_RE.search(s))
    parts = s.replace("/", " ").split()
    has_clade = any(
        p in _AIV_NAMED_CLADES or _AIV_CLADE_RE.match(p)
        for p in parts
    )
    if has_subtype and has_clade:
        return "specific"
    if has_subtype:
        return "partial"
    return "missing"


def _classify_covid_strain(s: str) -> str:
    """COVID: specific, family_level, root_level, missing.

    - specific: >= 2 dot segments after root letter (e.g., BA.5.2.1, AY.103)
      OR exact WHO-variant lineage (B.1.1.7, B.1.617.2)
    - family_level: known family prefix with 1 segment (BA.5, AY, P.1)
    - root_level: bare root (A, B, B.1)
    - missing: empty / nan
    """
    if s in _COVID_ROOT_LINEAGES:
        return "root_level"

    # Count dot-separated parts
    parts = s.split(".")
    n_parts = len(parts)

    # Special WHO-variant exact matches -> specific
    _who_specific = {"B.1.1.7", "B.1.351", "P.1", "B.1.617.2", "B.1.1.529"}
    if s in _who_specific:
        return "specific"

    # Root letter families (AY, BA, BQ, XBB, EG, JN, etc.)
    root = parts[0]
    if root in ("AY", "BA", "BQ", "XBB", "EG", "HV", "JN", "Q"):
        if n_parts >= 2:
            return "specific"
        return "family_level"

    # Generic: 3+ parts = specific, 2 parts = family_level, 1 = root
    if n_parts >= 3:
        return "specific"
    if n_parts == 2:
        return "family_level"
    return "root_level"


# ======================================================================
# Host system mapping
# ======================================================================


def resolve_host_system(
    category: str, subcategory: str, common_name: str, dataset: str,
) -> Optional[str]:
    """Map host metadata to a system label.

    AIV returns ``"wild_birds"`` or ``"poultry"`` (or ``None`` for
    mammals/humans, which are filtered out).
    COVID always returns ``"human"``.
    """
    if dataset in ("covid", "japan"):
        return "human"

    cat = category.strip().lower() if isinstance(category, str) else ""
    subcat = subcategory.strip().lower() if isinstance(subcategory, str) else ""

    # Wild birds
    if cat == "wild_birds":
        return "wild_birds"
    if cat == "bird" and subcat == "wild":
        return "wild_birds"

    # Poultry (domestic birds)
    if cat == "bird" and subcat == "domestic":
        return "poultry"
    if cat == "poultry":
        # poultry category is always poultry regardless of subcategory
        return "poultry"

    # Mammals & humans -> None (filtered out)
    if cat in ("mammal", "human"):
        return None

    # Ambiguous — log and return None
    logger.warning(
        "Ambiguous host system: category=%r, subcategory=%r, common_name=%r",
        cat, subcat, common_name,
    )
    return None


def should_filter_sample(record: SampleRecord) -> bool:
    """Return True if this AIV sample should be excluded (mammal/human)."""
    if record.dataset != "aiv":
        return False
    return record.host_system is None


# ======================================================================
# AIV strain parsing helpers (used by card builders)
# ======================================================================


def parse_aiv_strain_parts(strain: str) -> dict[str, Optional[str]]:
    """Extract subtype and clade from an AIV strain string.

    Returns ``{"subtype": ..., "clade": ...}`` where either value may be
    ``None``.
    """
    s = strain.strip() if isinstance(strain, str) else ""
    subtype_match = _AIV_SUBTYPE_RE.search(s)
    subtype = subtype_match.group(1).upper() if subtype_match else None

    clade: Optional[str] = None
    parts = s.replace("/", " ").split()
    for p in parts:
        if p in _AIV_NAMED_CLADES:
            clade = p
            break
        if _AIV_CLADE_RE.match(p):
            clade = p
            break

    return {"subtype": subtype, "clade": clade}


# ======================================================================
# Orchestrator
# ======================================================================


def resolve_all_fields(row: pd.Series, dataset: str) -> SampleRecord:
    """Build a fully-resolved :class:`SampleRecord` from one CSV row."""
    rec = SampleRecord(
        unique_identifier=str(row.get("Unique_Identifier", "")),
        collection_date_raw=str(row.get("Collection_Date", "")),
        report_date_raw=str(row.get("Report Date", "")),
        strain=str(row.get("Strain", "")),
        common_name=str(row.get("Common_Name", "")),
        category=str(row.get("Category", "")),
        subcategory=str(row.get("Subcategory", "")),
        state=str(row.get("State", "")),
        county=str(row.get("County", "")),
        sequence_id=str(row.get("sequence_Id", "")),
        dataset=dataset,
    )

    # Dates
    rec.collection_date = parse_date(rec.collection_date_raw)
    rec.report_date = parse_date(rec.report_date_raw)
    rec.report_lag_days, rec.report_lag_quality = compute_report_lag(
        rec.collection_date, rec.report_date,
    )

    # Strain
    rec.strain_resolution = classify_strain_resolution(rec.strain, dataset)

    # Host system
    rec.host_system = resolve_host_system(
        rec.category, rec.subcategory, rec.common_name, dataset,
    )

    # State
    if dataset == "japan":
        # Japanese prefectures — use directly (already lowercase from preprocessing)
        s = rec.state.strip().lower() if rec.state and rec.state.strip() else ""
        rec.resolved_state = s if s and s != "nan" else None
    else:
        rec.resolved_state = resolve_us_state(rec.state)

    # Filtering
    rec.is_filtered_out = should_filter_sample(rec)

    return rec
