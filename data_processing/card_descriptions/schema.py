"""Data structures for the four-card sample description system.

Provides :class:`SampleRecord` (one CSV row with raw + derived fields) and
:class:`CardOutput` (structured intermediate dicts, rendered text, and
canonical dedup key for one sample).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Optional


# ---------------------------------------------------------------------------
# SampleRecord – one row from case_data.csv
# ---------------------------------------------------------------------------


@dataclass
class SampleRecord:
    """One row from ``case_data.csv`` with raw CSV fields and derived values.

    Derived fields are populated by the resolvers in
    :mod:`field_resolvers`.
    """

    # --- Raw fields (verbatim from CSV) ---
    unique_identifier: str = ""
    collection_date_raw: str = ""
    report_date_raw: str = ""
    strain: str = ""
    common_name: str = ""
    category: str = ""
    subcategory: str = ""
    state: str = ""
    county: str = ""
    sequence_id: str = ""

    # --- Derived fields (set by resolvers) ---
    dataset: str = ""  # "aiv" | "covid"
    collection_date: Optional[date] = None
    report_date: Optional[date] = None
    report_lag_days: Optional[int] = None
    report_lag_quality: Optional[str] = None
    # AIV: "specific"|"partial"|"missing"
    # COVID: "specific"|"family_level"|"root_level"|"missing"
    strain_resolution: str = "missing"
    host_system: Optional[str] = None  # "wild_birds"|"poultry"|None (AIV); "human" (COVID)
    resolved_state: Optional[str] = None
    is_filtered_out: bool = False

    # --- Causal-cutoff context (None on legacy code paths) ---
    # Set when the record is rendered for a particular forecast origin.
    # Knowable-by-`as_of_date` is the redesign's leak-free invariant.
    as_of_date: Optional[date] = None
    causal_cutoff: bool = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def raw_fields_dict(self) -> dict[str, str]:
        """Return raw CSV fields as a plain dict."""
        return {
            "Unique_Identifier": self.unique_identifier,
            "Collection_Date": self.collection_date_raw,
            "Report Date": self.report_date_raw,
            "Strain": self.strain,
            "Common_Name": self.common_name,
            "Category": self.category,
            "Subcategory": self.subcategory,
            "State": self.state,
            "County": self.county,
            "sequence_Id": self.sequence_id,
        }

    def derived_fields_dict(self) -> dict[str, Any]:
        """Return derived fields as a JSON-safe dict."""
        return {
            "dataset": self.dataset,
            "collection_date": self.collection_date.isoformat() if self.collection_date else None,
            "report_date": self.report_date.isoformat() if self.report_date else None,
            "report_lag_days": self.report_lag_days,
            "report_lag_quality": self.report_lag_quality,
            "strain_resolution": self.strain_resolution,
            "host_system": self.host_system,
            "resolved_state": self.resolved_state,
            "is_filtered_out": self.is_filtered_out,
            "as_of_date": self.as_of_date.isoformat() if self.as_of_date else None,
            "causal_cutoff": self.causal_cutoff,
        }


# ---------------------------------------------------------------------------
# CardOutput – complete four-card output for one sample
# ---------------------------------------------------------------------------


@dataclass
class CardOutput:
    """Structured intermediate + rendered text for all four cards.

    The ``canonical_key`` is derived from structural fields (not rendered
    text) and used for deduplication before LLM encoding.
    """

    raw_fields: dict[str, str] = field(default_factory=dict)
    derived_fields: dict[str, Any] = field(default_factory=dict)

    # Structured intermediate representations
    disease_card_struct: dict[str, Any] = field(default_factory=dict)
    subtype_card_struct: dict[str, Any] = field(default_factory=dict)
    window_card_struct: dict[str, Any] = field(default_factory=dict)
    source_card_struct: dict[str, Any] = field(default_factory=dict)

    # Rendered text
    disease_card: str = ""
    subtype_card: str = ""
    window_card: str = ""
    source_card: str = ""
    model_summary: str = ""

    # Canonical dedup key
    canonical_key: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dictionary."""
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Canonical key builder
# ---------------------------------------------------------------------------


def build_canonical_key(record: SampleRecord) -> str:
    """Build a deterministic hash from the structural fields that determine
    card content.

    Two samples with the same canonical key will produce identical
    ``model_summary`` text and thus identical embeddings.
    """
    key_parts = (
        record.dataset,
        record.strain.strip().lower(),
        record.strain_resolution,
        record.host_system or "__none__",
        record.collection_date.isoformat() if record.collection_date else "__no_date__",
        record.report_lag_quality or "__no_lag__",
        record.resolved_state or "__no_state__",
        record.county.strip().lower() if record.county else "__no_county__",
        # As-of date splits the cache when causal_cutoff is on. On legacy
        # paths (causal_cutoff=False, as_of_date=None) the sentinel keeps
        # canonical keys identical to pre-redesign hashes.
        record.as_of_date.isoformat() if record.as_of_date else "__no_as_of__",
    )
    raw = json.dumps(key_parts, sort_keys=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()[:16]
