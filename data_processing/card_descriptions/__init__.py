"""Four-card structured disease description system.

This package generates sample-level structured descriptions for disease
surveillance data, using a four-card design (disease, subtype, window,
source) with structured intermediate representations and rendered text.

Public API
----------
.. autofunction:: build_sample_descriptions
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any, Iterable, Optional

import pandas as pd

from data_processing.card_descriptions.card_builders import (
    CARD_NAMES,
    build_all_cards,
)
from data_processing.card_descriptions.field_resolvers import resolve_all_fields
from data_processing.card_descriptions.lineage_snapshots import (
    LineageSnapshot,
    build_lineage_snapshot,
)
from data_processing.card_descriptions.schema import CardOutput, SampleRecord

logger = logging.getLogger("data_processing")

__all__ = [
    "build_sample_descriptions",
    "build_sample_descriptions_with_as_of",
    "CARD_NAMES",
    "CardOutput",
    "SampleRecord",
    "LineageSnapshot",
]


def build_sample_descriptions(
    df: pd.DataFrame,
    dataset: str,
    *,
    include_cards: frozenset[str] | None = None,
) -> dict[str, CardOutput]:
    """Generate legacy four-card descriptions for all non-filtered samples.

    This is the legacy (causal_cutoff=False) entry point preserved for
    the V1 row (a) ablation. Returns one CardOutput per Unique_Identifier
    keyed by the sample's identifier.

    Parameters
    ----------
    df:
        DataFrame from ``case_data.csv`` with the standard 10-column schema.
    dataset:
        ``"aiv"``, ``"covid"``, or ``"japan"``.
    include_cards:
        Optional subset of :data:`CARD_NAMES` to include in the
        concatenated ``model_summary`` for each sample. ``None`` (default)
        includes all four cards.
    """
    results: dict[str, CardOutput] = {}
    n_filtered = 0

    for _, row in df.iterrows():
        record = resolve_all_fields(row, dataset)

        if record.is_filtered_out:
            n_filtered += 1
            continue

        card_output = build_all_cards(record, include_cards=include_cards)
        results[record.unique_identifier] = card_output

    logger.info(
        "Built descriptions for %d samples (%d filtered out)",
        len(results),
        n_filtered,
    )
    return results


def build_sample_descriptions_with_as_of(
    df: pd.DataFrame,
    dataset: str,
    *,
    as_of_dates: Iterable[date],
    include_cards: frozenset[str] | None = None,
    lineage_block_enabled: bool = True,
    reporting_lag_days: int = 14,
    lineage_window_days: int = 28,
    trend_threshold: float = 0.10,
    window_size_days: Optional[int] = None,
) -> dict[tuple[str, str], CardOutput]:
    """Generate causal-cutoff four-card descriptions across a set of as-of dates.

    Window membership at training time is governed by the case-window
    invariant ``window_start <= collection_date < window_end``, where
    ``window_end = as_of + reporting_lag_days``. Records collected in the
    last ``reporting_lag_days`` of a window (i.e., between ``as_of`` and
    ``window_end``) are legitimate members of the window — they are the
    most recent samples in the case graph and must have a cached
    embedding, otherwise the graph builder falls back to ``__UNKNOWN__``.

    For each ``(record, as_of)`` pair the orchestrator therefore emits a
    CardOutput when:

    * ``record.collection_date`` is parseable; AND
    * if ``window_size_days`` is given,
      ``as_of + lag - window_size_days <= collection_date < as_of + lag``.

    When ``window_size_days`` is ``None`` no temporal filter is applied
    and a card is emitted for every ``(record, as_of)`` pair — correct
    but expensive. The renderer handles records collected after
    ``as_of`` cleanly (the "no public submissions on record by as-of"
    fallback), so this is leak-safe.

    A lineage snapshot is computed once per ``as_of`` (per
    dataset/country) and reused across all records sharing that
    ``as_of``. AIV is excluded from the snapshot path because its
    Window Card does not consume one.

    Returns
    -------
    dict
        ``{(unique_identifier, as_of_iso): CardOutput}``. The composite
        key matches the v2 embedding-cache contract.
    """
    as_of_list = sorted(set(as_of_dates))
    if not as_of_list:
        return {}

    use_snapshot = dataset in ("covid", "japan")

    # Resolve all records once.
    records: list[SampleRecord] = []
    for _, row in df.iterrows():
        rec = resolve_all_fields(row, dataset)
        if rec.is_filtered_out or rec.collection_date is None:
            continue
        records.append(rec)

    # Window-membership filter bounds (relative to as_of, in days):
    #   lag - window_size_days <= collection_date - as_of < lag
    # When window_size_days is None we keep the bounds wide-open.
    if window_size_days is not None:
        offset_lo = reporting_lag_days - int(window_size_days)
        offset_hi = reporting_lag_days
    else:
        offset_lo = None
        offset_hi = None

    results: dict[tuple[str, str], CardOutput] = {}
    for as_of in as_of_list:
        snapshot: Optional[LineageSnapshot] = None
        if use_snapshot:
            snapshot = build_lineage_snapshot(
                df, as_of=as_of,
                country=dataset,
                lineage_window_days=lineage_window_days,
                reporting_lag_days=reporting_lag_days,
                trend_threshold=trend_threshold,
            )
        for rec in records:
            if offset_lo is not None and offset_hi is not None:
                delta = (rec.collection_date - as_of).days
                if not (offset_lo <= delta < offset_hi):
                    continue
            scoped = SampleRecord(**{**rec.__dict__})
            scoped.as_of_date = as_of
            scoped.causal_cutoff = True
            card_output = build_all_cards(
                scoped,
                include_cards=include_cards,
                snapshot=snapshot,
                lineage_block_enabled=lineage_block_enabled,
                reporting_lag_days=reporting_lag_days,
            )
            results[(scoped.unique_identifier, as_of.isoformat())] = card_output

    logger.info(
        "Built %d (sample, as_of) descriptions across %d as-of dates "
        "(dataset=%s, snapshot=%s, lineage_block=%s)",
        len(results), len(as_of_list), dataset, use_snapshot, lineage_block_enabled,
    )
    return results
