"""Tests for the causal-cutoff Window Card redesign (Appendix B.2.3).

Pre-registered invariants:
1. No lineage emerging *after* ``as_of_date - Δ_lag`` may appear in the
   rendered card text.
2. Every causal-cutoff card must contain the literal token
   ``as-of {YYYY-MM-DD}`` so the cut-off is machine-checkable.
3. The AIV card-rendering path must be byte-identical to the legacy
   renderer (regression guard).
4. With ``causal_cutoff=true`` the COVID Subtype Card emits the uniform
   ``"No peer-reviewed characterisation of …"`` fallback.
5. With ``causal_cutoff=false`` the legacy retrospective phase phrasing
   (``"Omicron BA.1 wave"`` etc.) is preserved unchanged.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from data_processing.card_descriptions.card_builders import (
    build_all_cards,
    render_window_card,
)
from data_processing.card_descriptions.lineage_snapshots import (
    build_lineage_snapshot,
)
from data_processing.card_descriptions.schema import SampleRecord


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _covid_record(*, collection: date, strain: str = "BA.1.1",
                  unique_id: str = "MT_TEST_001") -> SampleRecord:
    rec = SampleRecord(
        unique_identifier=unique_id,
        collection_date_raw=collection.strftime("%d/%m/%Y"),
        report_date_raw=(collection.strftime("%d/%m/%Y")),
        strain=strain,
        common_name="human",
        category="human",
        subcategory="human",
        state="California",
        county="Los Angeles County",
        sequence_id=unique_id,
        dataset="covid",
    )
    rec.collection_date = collection
    rec.report_date = collection  # zero lag for test simplicity
    rec.strain_resolution = "specific"
    rec.host_system = "human"
    rec.resolved_state = "california"
    return rec


def _aiv_record(*, collection: date) -> SampleRecord:
    rec = SampleRecord(
        unique_identifier="AIV_TEST_001",
        collection_date_raw=collection.strftime("%d/%m/%Y"),
        report_date_raw=collection.strftime("%d/%m/%Y"),
        strain="A / H5N1 2.3.4.4b",
        common_name="mallard",
        category="bird",
        subcategory="wild",
        state="Washington",
        county="King County",
        sequence_id="AIV_TEST_001",
        dataset="aiv",
    )
    rec.collection_date = collection
    rec.report_date = collection
    rec.strain_resolution = "specific"
    rec.host_system = "wild_birds"
    rec.resolved_state = "washington"
    return rec


def _make_cases_df(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    """Build a minimal case_data dataframe.

    rows: list of (collection_iso, report_iso_or_blank, strain)
    """
    records = []
    for i, (coll, rep, strain) in enumerate(rows):
        coll_d = date.fromisoformat(coll)
        rep_str = rep
        records.append({
            "Unique_Identifier": f"R{i:03d}",
            "Collection_Date": coll_d.strftime("%-d/%-m/%Y"),
            "Report Date": rep_str,
            "Strain": strain,
            "Common_Name": "human",
            "Category": "human",
            "Subcategory": "human",
            "State": "California",
            "County": "Los Angeles County",
            "sequence_Id": f"R{i:03d}",
        })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Invariant 1: no future-only lineages may appear in the card text
# ---------------------------------------------------------------------------


def test_no_lineage_emerging_after_as_of():
    # XBB.1.5 first appears in the dataset on 2022-12-01; with as_of=2022-02-28
    # it MUST NOT leak into the card text for a record collected 2022-01-15.
    cases = _make_cases_df([
        ("2022-01-10", "2022-01-12", "BA.1.1"),
        ("2022-01-12", "2022-01-15", "BA.1.1"),
        ("2022-01-20", "2022-01-25", "BA.2"),
        ("2022-12-01", "2022-12-03", "XBB.1.5"),
        ("2022-12-15", "2022-12-18", "XBB.1.5"),
    ])
    as_of = date(2022, 2, 28)
    snapshot = build_lineage_snapshot(
        cases, as_of=as_of, country="covid",
        lineage_window_days=28, reporting_lag_days=14,
    )
    rec = _covid_record(collection=date(2022, 1, 15), strain="BA.1.1")
    rec.causal_cutoff = True
    rec.as_of_date = as_of

    output = build_all_cards(rec, snapshot=snapshot)
    text = output.window_card

    assert "XBB" not in text, f"future lineage XBB leaked into card: {text!r}"
    assert "as-of 2022-02-28" in text


# ---------------------------------------------------------------------------
# Invariant 2: literal as-of token must appear
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "as_of",
    [date(2022, 1, 31), date(2022, 4, 30), date(2022, 8, 31)],
)
def test_as_of_token_present(as_of: date):
    cases = _make_cases_df([
        (as_of.isoformat(), as_of.isoformat(), "BA.1.1"),
    ])
    snapshot = build_lineage_snapshot(
        cases, as_of=as_of, country="covid",
    )
    rec = _covid_record(collection=as_of, strain="BA.1.1")
    rec.causal_cutoff = True
    rec.as_of_date = as_of

    output = build_all_cards(rec, snapshot=snapshot)
    expected = f"as-of {as_of.isoformat()}"
    assert expected in output.window_card, (
        f"missing literal token {expected!r} in:\n{output.window_card}"
    )


# ---------------------------------------------------------------------------
# Invariant 3: AIV path is byte-identical (regression guard)
# ---------------------------------------------------------------------------


def test_aiv_path_unchanged():
    rec = _aiv_record(collection=date(2023, 11, 5))

    legacy_only = build_all_cards(rec)
    # Even when the global causal_cutoff toggle is on, an AIV record's
    # ``record.causal_cutoff`` stays False — the redesign is a no-op for
    # AIV. We assert this by setting the flag on the record and verifying
    # nothing changes.
    rec_with_flag = _aiv_record(collection=date(2023, 11, 5))
    rec_with_flag.causal_cutoff = True
    rec_with_flag.as_of_date = date(2023, 11, 19)
    with_flag = build_all_cards(rec_with_flag)

    assert legacy_only.window_card == with_flag.window_card
    assert legacy_only.subtype_card == with_flag.subtype_card
    # Confirm none of the redesign tokens appear on the AIV path.
    assert "as-of" not in legacy_only.window_card
    assert "Lineage-context snapshot" not in legacy_only.window_card


# ---------------------------------------------------------------------------
# Invariant 4: COVID Subtype fallback under causal_cutoff
# ---------------------------------------------------------------------------


def test_subtype_fallback_when_causal():
    rec = _covid_record(collection=date(2022, 1, 15), strain="BA.1.1")
    rec.causal_cutoff = True
    rec.as_of_date = date(2022, 2, 28)

    output = build_all_cards(rec, snapshot=None, lineage_block_enabled=False)
    text = output.subtype_card
    assert "No peer-reviewed characterisation of" in text
    # Retrospective fitness/escape descriptors must not survive.
    assert "rapid global displacement" not in text
    assert "additional immune evasion" not in text
    assert "became globally dominant" not in text


# ---------------------------------------------------------------------------
# Invariant 5: legacy mode preserves retrospective phrasing
# ---------------------------------------------------------------------------


def test_records_collected_after_as_of_get_cards():
    """Records collected during the lag window (between as_of and window_end)
    must still receive a CardOutput — they are legitimate window members at
    training time and would otherwise miss in the v2 cache lookup.
    """
    from data_processing.card_descriptions import build_sample_descriptions_with_as_of

    # Record collected on 2022-03-10, as_of = 2022-02-28 (10 days into the lag).
    # window_end = as_of + lag = 2022-03-14, so this record is in the window.
    cases = _make_cases_df([
        ("2022-02-15", "2022-02-17", "BA.1.1"),  # before as_of
        ("2022-03-10", "2022-03-12", "BA.1.1"),  # AFTER as_of, within lag
        ("2022-03-13", "2022-03-15", "BA.2"),    # AFTER as_of, within lag
    ])
    as_of = date(2022, 2, 28)
    out = build_sample_descriptions_with_as_of(
        cases, "covid", as_of_dates=[as_of], reporting_lag_days=14,
        window_size_days=56,
    )
    keys = {sid for (sid, _) in out.keys()}
    assert "R000" in keys, "before-as_of record missing"
    assert "R001" in keys, "in-lag-after-as_of record missing — would 404 at training"
    assert "R002" in keys, "in-lag-after-as_of record missing — would 404 at training"


def test_window_size_filter_excludes_far_past():
    """With window_size_days=56 and lag=14, records collected more than
    (window_size - lag) = 42 days before as_of are excluded.
    """
    from data_processing.card_descriptions import build_sample_descriptions_with_as_of

    cases = _make_cases_df([
        ("2021-01-15", "2021-01-17", "B.1"),     # 409 days before as_of → excluded
        ("2022-01-20", "2022-01-22", "BA.1.1"),  # 39 days before as_of → kept
        ("2022-02-25", "2022-02-27", "BA.1.1"),  # 3 days before as_of → kept
    ])
    as_of = date(2022, 2, 28)
    out = build_sample_descriptions_with_as_of(
        cases, "covid", as_of_dates=[as_of], reporting_lag_days=14,
        window_size_days=56,
    )
    keys = {sid for (sid, _) in out.keys()}
    assert "R000" not in keys, "far-past record should be excluded"
    assert "R001" in keys
    assert "R002" in keys


def test_legacy_mode_still_works():
    rec = _covid_record(collection=date(2022, 1, 15), strain="BA.1")
    # Default record.causal_cutoff is False → legacy path.
    output = build_all_cards(rec)
    text = output.window_card
    assert "Omicron BA.1 wave" in text, (
        f"legacy phase label missing from window card:\n{text}"
    )
    # And no causal-cutoff tokens leak in.
    assert "as-of" not in text
    assert "Lineage-context snapshot" not in text
