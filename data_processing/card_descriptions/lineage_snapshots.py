"""As-of-date lineage snapshots for the causal-cutoff Window Card.

A snapshot is the set of values that a real-time surveillance dashboard
could have reported on day ``as_of`` from publicly-submitted sequences:

- top-3 most-frequent Pango lineages over the past ``lineage_window_days``,
- their trend versus the prior equal-length window,
- per-lineage first-observed date and cumulative count, restricted to
  records whose effective publication date is ``<= as_of``.

All values are derived from the project's own ``case_data.csv`` — never
from external sources — so the snapshot is reproducible and free of
hidden post-hoc lookups.

Effective publication date for a row is::

    pub_eff = report_date if report_date else collection_date + Δ_lag

A row contributes to the as-of-`t` snapshot iff ``pub_eff <= t``. This is
a conservative proxy for the GISAID/GenBank submission date (which is
not present in the project's case_data files).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from functools import lru_cache
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LineageSnapshot:
    """As-of-`as_of` snapshot of lineage activity in `country`."""

    as_of: date
    country: str
    lineage_window_days: int
    reporting_lag_days: int
    # Top-3 lineages over the past `lineage_window_days`: list of (lineage, pct).
    # `pct` is rounded to one decimal place. May contain fewer than 3 entries
    # when the past window is sparse.
    top3: tuple[tuple[str, float], ...]
    # Trend tag per top-3 lineage: "rising" | "stable" | "declining" | "new".
    # "new" = lineage absent in prior window. Compared to the equal-length
    # prior window using a ±`trend_threshold` proportional-change rule.
    trend: dict[str, str]
    # First-observation date and cumulative-count-to-as-of for every lineage
    # that appears in the as-of-knowable subset of the dataframe.
    first_obs: dict[str, date]
    cum_count: dict[str, int]
    # Sample sizes for diagnostics.
    n_rows_window: int
    n_rows_prior: int
    n_rows_total: int


def compute_publication_dates(
    cases: pd.DataFrame, *, reporting_lag_days: int,
) -> pd.Series:
    """Compute the effective publication date for every row.

    Uses ``Report Date`` when present and parseable, otherwise
    ``Collection_Date + reporting_lag_days``. Returns a ``datetime64[ns]``
    Series aligned to ``cases.index``; rows where neither date is parseable
    receive ``NaT`` and will be excluded from snapshots.
    """
    coll = pd.to_datetime(
        cases.get("Collection_Date"), format="mixed", dayfirst=True, errors="coerce",
    )
    rep = pd.to_datetime(
        cases.get("Report Date"), format="mixed", dayfirst=True, errors="coerce",
    )
    fallback = coll + pd.to_timedelta(reporting_lag_days, unit="D")
    return rep.fillna(fallback)


def build_lineage_snapshot(
    cases: pd.DataFrame,
    *,
    as_of: date,
    country: str = "covid",
    lineage_window_days: int = 28,
    reporting_lag_days: int = 14,
    trend_threshold: float = 0.10,
    strain_col: str = "Strain",
) -> LineageSnapshot:
    """Build an as-of-`as_of` lineage snapshot from a case dataframe.

    Parameters
    ----------
    cases
        DataFrame with at least ``Collection_Date``, ``Strain``, and
        optionally ``Report Date`` columns. Rows with missing or unparseable
        Strain are dropped.
    as_of
        The cut-off date. Only rows whose effective publication date is
        ``<= as_of`` are considered.
    country
        Free-form label retained on the snapshot for provenance (one of
        ``"covid"``, ``"japan"``, ...). Filtering by country must be done by
        the caller before calling this function.
    lineage_window_days
        Past-window length for top-3 frequencies and trend computation.
    reporting_lag_days
        Δ_lag fallback when Report Date is missing.
    trend_threshold
        Proportional-change threshold for the trend classifier.
    strain_col
        Column name carrying the Pango/strain identifier.
    """
    if as_of is None:
        raise ValueError("build_lineage_snapshot requires a non-None as_of date.")

    pub = compute_publication_dates(cases, reporting_lag_days=reporting_lag_days)
    coll = pd.to_datetime(cases.get("Collection_Date"), dayfirst=True, errors="coerce")
    as_of_ts = pd.Timestamp(as_of)
    knowable_mask = pub.notna() & (pub <= as_of_ts) & coll.notna()

    strains = cases[strain_col].astype(str).str.strip()
    valid = strains.notna() & (strains.str.lower() != "nan") & (strains != "")
    knowable_mask = knowable_mask & valid

    knowable = cases[knowable_mask].copy()
    knowable["__strain__"] = strains[knowable_mask].values
    knowable["__coll__"] = coll[knowable_mask].values

    n_total = int(len(knowable))
    if n_total == 0:
        return LineageSnapshot(
            as_of=as_of, country=country,
            lineage_window_days=lineage_window_days,
            reporting_lag_days=reporting_lag_days,
            top3=tuple(), trend={}, first_obs={}, cum_count={},
            n_rows_window=0, n_rows_prior=0, n_rows_total=0,
        )

    # Cumulative count and first-observation date — knowable-by-as_of.
    grouped = knowable.groupby("__strain__")
    cum_count = grouped.size().astype(int).to_dict()
    first_obs_ts = grouped["__coll__"].min().dropna()
    first_obs = {k: pd.Timestamp(v).date() for k, v in first_obs_ts.items()}

    # Past `lineage_window_days` (by collection date).
    window_start = as_of_ts - pd.Timedelta(days=lineage_window_days)
    prior_start = window_start - pd.Timedelta(days=lineage_window_days)
    coll_ts = knowable["__coll__"]

    in_window = (coll_ts > window_start) & (coll_ts <= as_of_ts)
    in_prior = (coll_ts > prior_start) & (coll_ts <= window_start)

    window_counts = (
        knowable.loc[in_window, "__strain__"].value_counts()
    )
    prior_counts = (
        knowable.loc[in_prior, "__strain__"].value_counts()
    )
    n_window = int(in_window.sum())
    n_prior = int(in_prior.sum())

    top3_list: list[tuple[str, float]] = []
    if n_window > 0:
        top3 = window_counts.head(3)
        for lineage, count in top3.items():
            pct = round(100.0 * count / n_window, 1)
            top3_list.append((str(lineage), pct))

    trend: dict[str, str] = {}
    for lineage, _ in top3_list:
        cur = float(window_counts.get(lineage, 0)) / n_window if n_window else 0.0
        prev = float(prior_counts.get(lineage, 0)) / n_prior if n_prior else 0.0
        if prev == 0.0 and cur > 0.0:
            trend[lineage] = "new"
        elif prev == 0.0:
            trend[lineage] = "stable"
        else:
            delta = (cur - prev) / prev
            if delta > trend_threshold:
                trend[lineage] = "rising"
            elif delta < -trend_threshold:
                trend[lineage] = "declining"
            else:
                trend[lineage] = "stable"

    return LineageSnapshot(
        as_of=as_of, country=country,
        lineage_window_days=lineage_window_days,
        reporting_lag_days=reporting_lag_days,
        top3=tuple(top3_list),
        trend=trend,
        first_obs=first_obs,
        cum_count=cum_count,
        n_rows_window=n_window,
        n_rows_prior=n_prior,
        n_rows_total=n_total,
    )


# ---------------------------------------------------------------------------
# Time-of-year helpers (always real-time-knowable)
# ---------------------------------------------------------------------------


def hemisphere_season(month: int, *, hemisphere: str = "northern") -> str:
    """Return the meteorological season for a calendar month.

    Months 12, 1, 2 → winter; 3, 4, 5 → spring; 6, 7, 8 → summer;
    9, 10, 11 → autumn (Northern Hemisphere). Southern is the inverse.
    """
    if hemisphere not in ("northern", "southern"):
        raise ValueError(f"hemisphere must be 'northern' or 'southern', got {hemisphere!r}")
    n = ["winter", "spring", "summer", "autumn"]
    idx = ((month % 12) // 3)  # 12,1,2->0(winter); 3,4,5->1(spring); ...
    season = n[idx]
    if hemisphere == "southern":
        season = {"winter": "summer", "spring": "autumn",
                  "summer": "winter", "autumn": "spring"}[season]
    return season


def country_hemisphere(country: str) -> str:
    """Map a project-internal country label to a hemisphere."""
    c = country.strip().lower()
    if c in ("us", "united states", "covid", "japan", "japan_covid"):
        return "northern"
    return "northern"


def reporting_lag_for_record(
    *, collection_date: Optional[date], report_date: Optional[date],
    reporting_lag_days: int,
) -> int:
    """Effective reporting lag in days for one record.

    Uses ``(report_date - collection_date)`` when both are parseable,
    otherwise the configured default ``reporting_lag_days``.
    """
    if collection_date and report_date:
        delta = (report_date - collection_date).days
        if delta >= 0:
            return int(delta)
    return int(reporting_lag_days)
