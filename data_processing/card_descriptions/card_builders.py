"""Four-card struct builders and text renderers.

Each card has a pair of functions:

* ``build_<card>_struct(record, ...) -> dict``  — structured intermediate
* ``render_<card>(struct) -> str``              — text from struct

:func:`build_all_cards` orchestrates both stages and returns a complete
:class:`CardOutput`.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any, Optional

from data_processing.card_descriptions.field_resolvers import (
    parse_aiv_strain_parts,
)
from data_processing.card_descriptions.knowledge_tables import (
    AIV_CLADE_KNOWLEDGE,
    AIV_SUBTYPE_NOTES,
    get_aiv_seasonal_label,
    get_covid_pandemic_phase,
    get_covid_variant_context,
)
from data_processing.card_descriptions.lineage_snapshots import (
    LineageSnapshot,
    country_hemisphere,
    hemisphere_season,
    reporting_lag_for_record,
)
from data_processing.card_descriptions.schema import (
    CardOutput,
    SampleRecord,
    build_canonical_key,
)

# Card identifiers used by ablation toggles. The order here is also the
# concatenation order in build_model_summary.
CARD_NAMES: tuple[str, ...] = ("disease", "subtype", "window", "source")


# ======================================================================
# Text canonicalization
# ======================================================================


def _canonicalize_text(text: str) -> str:
    """Lightweight text canonicalization before concatenation.

    - Strip leading/trailing whitespace
    - Collapse multiple spaces
    - Remove double periods
    - Remove trailing commas before periods
    - Normalise date references to ISO (YYYY-MM-DD) where they appear
      in DD/M/YYYY format
    """
    s = text.strip()
    s = re.sub(r"  +", " ", s)
    s = s.replace("..", ".")
    s = s.replace(",.", ".")
    # Normalise inline DD/M/YYYY -> YYYY-MM-DD
    s = re.sub(
        r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b",
        lambda m: f"{m.group(3)}-{int(m.group(2)):02d}-{int(m.group(1)):02d}",
        s,
    )
    return s


def _iso(d: Optional[date]) -> str:
    """Format a date as ISO string or placeholder."""
    return d.isoformat() if d else "date unknown"


# ======================================================================
# Disease Card
# ======================================================================


def build_disease_card_struct(record: SampleRecord) -> dict[str, Any]:
    """Build structured disease-card fields."""
    if record.dataset == "aiv":
        return {
            "disease": "avian influenza",
            "pathogen_type": "Influenza A virus",
            "scope_note": (
                "This module considers wild-bird and poultry host systems "
                "only. Avian influenza A viruses circulate among wild bird "
                "populations and can amplify within domestic poultry systems."
            ),
        }
    # covid / japan — both SARS-CoV-2
    country = "Japan" if record.dataset == "japan" else "United States"
    return {
        "disease": "COVID-19",
        "pathogen_type": "SARS-CoV-2",
        "country": country,
        "dynamics_note": (
            "SARS-CoV-2 is a respiratory pathogen characterised by "
            "repeated variant emergence driven by immune selection, with "
            "surveillance systems subject to variable testing intensity "
            "and sequencing coverage."
        ),
    }


def render_disease_card(struct: dict[str, Any]) -> str:
    """Render disease card to text."""
    if "scope_note" in struct:
        # AIV
        return (
            f"This sample was collected during a highly pathogenic avian "
            f"influenza (HPAI) surveillance effort in the United States. "
            f"{struct['scope_note']}"
        )
    # COVID / Japan
    country = struct.get("country", "United States")
    country_phrase = f"the {country}" if country == "United States" else country
    return (
        f"This sample was collected during the SARS-CoV-2 pandemic in "
        f"{country_phrase}. {struct['dynamics_note']}"
    )


# ======================================================================
# Subtype Card
# ======================================================================


def build_subtype_card_struct(record: SampleRecord) -> dict[str, Any]:
    """Build structured subtype-card fields."""
    if record.dataset == "aiv":
        return _build_aiv_subtype_struct(record)
    return _build_covid_subtype_struct(record)


def _build_aiv_subtype_struct(record: SampleRecord) -> dict[str, Any]:
    parts = parse_aiv_strain_parts(record.strain)
    subtype = parts["subtype"]
    clade = parts["clade"]
    clade_knowledge = AIV_CLADE_KNOWLEDGE.get(clade) if clade else None
    subtype_note = AIV_SUBTYPE_NOTES.get(subtype, "") if subtype else ""

    date_str = _iso(record.collection_date)
    host = record.host_system or "unknown"

    if clade_knowledge:
        date_anchored = (
            f"At the time of collection ({date_str}), clade {clade} was "
            f"circulating in North American {host} populations. "
            f"{clade_knowledge['emergence']}."
        )
    elif subtype:
        date_anchored = (
            f"At the time of collection ({date_str}), {subtype} was "
            f"detected in North American {host} populations."
        )
    else:
        date_anchored = (
            f"Strain information not reported for this sample collected "
            f"on {date_str}."
        )

    return {
        "strain_raw": record.strain,
        "strain_resolution": record.strain_resolution,
        "subtype": subtype,
        "subtype_note": subtype_note,
        "clade": clade,
        "clade_knowledge": clade_knowledge,
        "host_system": host,
        "collection_date": date_str,
        "date_anchored_context": date_anchored,
    }


def _build_covid_subtype_struct(record: SampleRecord) -> dict[str, Any]:
    lineage = record.strain.strip() if record.strain else ""
    date_str = _iso(record.collection_date)
    as_of_str = (
        record.as_of_date.isoformat() if record.as_of_date else None
    )

    # Causal-cutoff path (B.2.3): retrospective fitness/escape descriptors
    # come from a curated knowledge base whose entries lack publication-date
    # metadata in this codebase, so per the redesign protocol we emit the
    # uniform fallback string and treat the lost descriptor as the
    # documented cost of the leak-free design.
    if record.causal_cutoff:
        as_of_clause = f"as-of {as_of_str}" if as_of_str else "as-of date unknown"
        strain_id = lineage if lineage else "unspecified"
        fitness_stmt = (
            f"No peer-reviewed characterisation of {strain_id} available "
            f"by {as_of_str or 'as-of date unknown'}."
        )
        escape_stmt = (
            f"No peer-reviewed characterisation of {strain_id} available "
            f"by {as_of_str or 'as-of date unknown'}."
        )
        return {
            "lineage_raw": lineage,
            "strain_resolution": record.strain_resolution,
            "variant_context": None,
            "collection_date": date_str,
            "as_of_date": as_of_str,
            "as_of_clause": as_of_clause,
            "causal_cutoff": True,
            "date_anchored_fitness": fitness_stmt,
            "date_anchored_escape": escape_stmt,
        }

    variant_ctx = get_covid_variant_context(
        lineage, record.collection_date or date(2020, 1, 1),
    )

    # Build dated fitness statement, hedged by confidence
    confidence = variant_ctx["confidence"]
    family = variant_ctx["family"]
    who = variant_ctx.get("who_label")

    if confidence == "uncharacterised":
        fitness_stmt = (
            f"Lineage {lineage} collected on {date_str}; fitness relative "
            f"to contemporaneous variants is not characterised."
        )
        escape_stmt = (
            f"Immune escape profile not characterised for this lineage."
        )
    elif variant_ctx["fallback_level"] >= 2:
        fitness_stmt = (
            f"Lineage {lineage} collected on {date_str}; interpreted in "
            f"the context of the broader {family} family. "
            f"{variant_ctx['fitness_context']}."
        )
        escape_stmt = (
            f"Immune escape inferred from the broader {family} family: "
            f"{variant_ctx['immune_escape']}. "
            f"Sub-lineage-specific characterisation unavailable."
        )
    else:
        who_clause = f" ({who})" if who else ""
        fitness_stmt = (
            f"Lineage {lineage}{who_clause} collected on {date_str}. "
            f"{variant_ctx['fitness_context']}."
        )
        escape_stmt = f"{variant_ctx['immune_escape']}."

    return {
        "lineage_raw": lineage,
        "strain_resolution": record.strain_resolution,
        "variant_context": variant_ctx,
        "collection_date": date_str,
        "as_of_date": as_of_str,
        "causal_cutoff": False,
        "date_anchored_fitness": fitness_stmt,
        "date_anchored_escape": escape_stmt,
    }


def render_subtype_card(struct: dict[str, Any]) -> str:
    """Render subtype card to text."""
    if "subtype" in struct:
        return _render_aiv_subtype(struct)
    return _render_covid_subtype(struct)


def _render_aiv_subtype(s: dict[str, Any]) -> str:
    parts: list[str] = []

    # Strain resolution
    res = s["strain_resolution"]
    if res == "specific":
        parts.append(
            f"Strain characterised at specific resolution: "
            f"{s['subtype']} clade {s['clade']}."
        )
    elif res == "partial":
        parts.append(
            f"Strain partially characterised: subtype {s['subtype']}, "
            f"clade unspecified."
        )
    else:
        parts.append("Strain information not reported.")

    # Subtype note
    if s.get("subtype_note"):
        parts.append(s["subtype_note"] + ".")

    # Date-anchored context
    parts.append(s["date_anchored_context"])

    # Clade-specific knowledge (no virulence/adaptation/transmissibility claims)
    ck = s.get("clade_knowledge")
    if ck:
        parts.append(f"Pathogenicity: {ck['pathogenicity']}.")
        parts.append(ck["avian_drift_note"] + ".")
        parts.append(ck["reassortment_note"] + ".")

        # Host-specific semantics (no spillover)
        host = s["host_system"]
        if host == "wild_birds":
            parts.append(ck["wild_bird_circulation"] + ".")
        elif host == "poultry":
            parts.append(ck["poultry_amplification"] + ".")

    return " ".join(parts)


def _render_covid_subtype(s: dict[str, Any]) -> str:
    parts: list[str] = []

    # Resolution level
    res = s["strain_resolution"]
    lineage = s["lineage_raw"]
    if res == "specific":
        parts.append(f"SARS-CoV-2 Pango lineage {lineage}.")
    elif res == "family_level":
        parts.append(
            f"SARS-CoV-2 Pango lineage {lineage} "
            f"(family-level classification)."
        )
    elif res == "root_level":
        parts.append(
            f"SARS-CoV-2 root lineage {lineage}; limited subtype resolution."
        )
    else:
        parts.append("SARS-CoV-2 lineage unspecified.")

    # Dated fitness and escape (or fallback strings on the causal-cutoff path)
    parts.append(s["date_anchored_fitness"])
    parts.append(s["date_anchored_escape"])

    if s.get("causal_cutoff"):
        # Stamp the as-of date so the cut-off is machine-checkable in cards.
        as_of_clause = s.get("as_of_clause") or "as-of date unknown"
        parts.append(f"Subtype description {as_of_clause}.")
        return " ".join(parts)

    # Confidence caveat for high fallback (legacy path only)
    ctx = s.get("variant_context") or {}
    if ctx.get("fallback_level", 0) >= 2:
        parts.append(
            "Note: lineage-specific context was inferred via family-level "
            "fallback; interpret with caution."
        )

    return " ".join(parts)


# ======================================================================
# Window Card
# ======================================================================


def build_window_card_struct(
    record: SampleRecord,
    *,
    snapshot: Optional["LineageSnapshot"] = None,
    lineage_block_enabled: bool = True,
    reporting_lag_days: int = 14,
) -> dict[str, Any]:
    """Build structured window-card fields.

    Pure temporal + location + data quality.  No lineage interpretation
    on the legacy path; on the causal-cutoff path the optional
    ``snapshot`` carries an as-of-`as_of_date` lineage activity summary
    derived from the project's own case dataframe.

    Parameters
    ----------
    record:
        The :class:`SampleRecord`. ``record.causal_cutoff`` and
        ``record.as_of_date`` toggle the causal-cutoff path.
    snapshot:
        Optional :class:`LineageSnapshot` for the COVID/Japan
        causal-cutoff path. Required when ``record.causal_cutoff`` is
        True, ``record.dataset in {"covid", "japan"}``, and
        ``lineage_block_enabled`` is True. Ignored otherwise.
    lineage_block_enabled:
        When False, the lineage-snapshot block is omitted (V1 row (c)
        date-only ablation).
    reporting_lag_days:
        Δ_lag fallback used for the per-record reporting-lag field when
        Report Date is missing.
    """
    struct: dict[str, Any] = {
        "collection_date": _iso(record.collection_date),
        "report_date": _iso(record.report_date),
        "report_lag_days": record.report_lag_days,
        "report_lag_quality": record.report_lag_quality,
        "state": record.resolved_state or (
            record.state if record.state and record.state.strip().lower() not in ("nan", "") else "location unknown"
        ),
        "county": record.county,
    }

    # AIV seasonal context (hedged, weak). The AIV path is preserved on
    # both legacy and causal-cutoff configurations — its phrasing is
    # already real-time-knowable.
    if record.dataset == "aiv" and record.collection_date:
        label = get_aiv_seasonal_label(record.collection_date.month)
        struct["seasonal_context"] = label
        struct["seasonal_context_strength"] = "weak"
    else:
        struct["seasonal_context"] = None
        struct["seasonal_context_strength"] = None

    # Country
    struct["country"] = "Japan" if record.dataset == "japan" else "United States"

    causal = record.causal_cutoff and record.dataset in ("covid", "japan")
    struct["causal_cutoff"] = causal

    if causal:
        # Retrospective phase labels are forbidden on this path.
        struct["pandemic_phase"] = None
        struct["as_of_date"] = (
            record.as_of_date.isoformat() if record.as_of_date else None
        )
        struct["lineage_block_enabled"] = bool(lineage_block_enabled)
        struct["lineage_snapshot"] = snapshot if (snapshot and lineage_block_enabled) else None

        # Time-of-year fields (always real-time-knowable).
        if record.collection_date:
            struct["month"] = record.collection_date.month
            struct["hemisphere"] = country_hemisphere(record.dataset)
            struct["hemisphere_season"] = hemisphere_season(
                record.collection_date.month, hemisphere=struct["hemisphere"],
            )
        else:
            struct["month"] = None
            struct["hemisphere"] = country_hemisphere(record.dataset)
            struct["hemisphere_season"] = None

        struct["record_reporting_lag_days"] = reporting_lag_for_record(
            collection_date=record.collection_date,
            report_date=record.report_date,
            reporting_lag_days=reporting_lag_days,
        )
    else:
        struct["as_of_date"] = None
        struct["lineage_snapshot"] = None
        struct["lineage_block_enabled"] = False
        struct["month"] = None
        struct["hemisphere"] = None
        struct["hemisphere_season"] = None
        struct["record_reporting_lag_days"] = None

        # COVID / Japan retrospective pandemic phase (legacy path only).
        if record.dataset in ("covid", "japan") and record.collection_date:
            struct["pandemic_phase"] = get_covid_pandemic_phase(
                record.collection_date,
            )
        else:
            struct["pandemic_phase"] = None

    return struct


def render_window_card(struct: dict[str, Any]) -> str:
    """Render window card to text.

    Branches on ``struct["causal_cutoff"]``: the causal-cutoff path
    emits the as-of-`as_of_date` schema specified in Appendix B.2.3 of
    the redesigned paper (lineage snapshot + time-of-year + reporting
    lag); the legacy path keeps the retrospective phase-label phrasing.
    """
    if struct.get("causal_cutoff"):
        return _render_window_card_causal(struct)
    return _render_window_card_legacy(struct)


def _render_window_card_legacy(struct: dict[str, Any]) -> str:
    parts: list[str] = []

    # Location + date
    state = struct["state"]
    county = struct["county"]
    coll = struct["collection_date"]
    has_county = bool(county and county.strip().lower() not in ("nan", ""))
    location = f"{county}, {state}" if has_county else state
    country = struct.get("country", "United States")
    parts.append(f"Collected on {coll} in {location}, {country}.")

    # Seasonal context (AIV, hedged)
    seasonal = struct.get("seasonal_context")
    if seasonal and struct.get("seasonal_context_strength") == "weak":
        parts.append(
            f"Collection timing is consistent with the seasonal "
            f"wild-bird {seasonal} context, though local and species-specific "
            f"variation in migration timing is expected."
        )

    # Pandemic phase (COVID)
    phase = struct.get("pandemic_phase")
    if phase:
        parts.append(
            f"Collected during the {phase} of the SARS-CoV-2 pandemic."
        )

    # Report lag
    lag_days = struct.get("report_lag_days")
    lag_quality = struct.get("report_lag_quality")

    if lag_quality == "negative_or_inconsistent":
        parts.append(
            "Temporal ordering in metadata is inconsistent; collection "
            "and report dates may be unreliable."
        )
    elif lag_days is not None and lag_quality:
        if lag_quality == "prompt":
            interpretation = "near-real-time reporting"
        elif lag_quality == "moderate":
            interpretation = "standard reporting delay"
        elif lag_quality == "delayed":
            interpretation = "delayed reporting; may reflect batch uploads"
        else:
            interpretation = (
                "severely delayed reporting; likely retrospective data entry"
            )
        parts.append(
            f"Report lag was {lag_days} days ({lag_quality}), indicating "
            f"{interpretation}."
        )
    elif lag_days is None:
        parts.append("Report lag could not be determined from available metadata.")

    return " ".join(parts)


def _render_window_card_causal(struct: dict[str, Any]) -> str:
    """Render the causal-cutoff Window Card.

    Every emitted string is constructible from information whose
    timestamp is ``<= as_of_date``. Retrospective phase labels and
    dominance-window descriptors must not appear on this path. The
    literal token ``as-of {as_of_date}`` is required so the cut-off is
    machine-checkable.
    """
    state = struct["state"]
    county = struct["county"]
    coll = struct["collection_date"]
    has_county = bool(county and county.strip().lower() not in ("nan", ""))
    location = f"{county}, {state}" if has_county else state
    country = struct.get("country", "United States")
    as_of = struct.get("as_of_date") or "date unknown"

    parts: list[str] = []
    parts.append(f"Sample collected on {coll} in {location}, {country}.")

    snapshot: Optional["LineageSnapshot"] = struct.get("lineage_snapshot")
    block_enabled = bool(struct.get("lineage_block_enabled"))

    if block_enabled and snapshot is not None:
        window_days = snapshot.lineage_window_days
        if snapshot.top3:
            top3_str = ", ".join(
                f"{lin} ({pct}%)" for lin, pct in snapshot.top3
            )
            trend_str = ", ".join(
                f"{lin} {snapshot.trend.get(lin, 'stable')}"
                for lin, _ in snapshot.top3
            )
            parts.append(
                f"Lineage-context snapshot (as-of {as_of}, source: cumulative "
                f"public submissions for {country}): past {window_days} days, "
                f"top-3 most-frequent Pango lineages (assignments available "
                f"by {as_of}): {top3_str}. Trend vs previous {window_days}-day "
                f"window: {trend_str}."
            )
        else:
            parts.append(
                f"Lineage-context snapshot (as-of {as_of}, source: cumulative "
                f"public submissions for {country}): no Pango lineage "
                f"assignments available in the past {window_days} days."
            )

        # This sample's own lineage, if knowable by as-of.
        # The snapshot's first_obs/cum_count maps cover all knowable strains
        # in the country; we look up by the record's strain if present in
        # the rendered struct (set by build_all_cards via the orchestrator).
        record_strain = struct.get("record_strain") or ""
        if record_strain and record_strain in snapshot.first_obs:
            first_obs = snapshot.first_obs[record_strain].isoformat()
            cum = snapshot.cum_count.get(record_strain, 0)
            parts.append(
                f"This sample's strain {record_strain} was first observed in "
                f"public submissions on {first_obs}; cumulative count to "
                f"as-of {as_of}: {cum}."
            )
        elif record_strain:
            parts.append(
                f"This sample's strain {record_strain} has no public "
                f"submissions on record by as-of {as_of}."
            )
        else:
            parts.append(
                f"This sample's strain identifier is unavailable; no "
                f"per-strain submission record can be reported as-of {as_of}."
            )
    elif not block_enabled:
        parts.append(
            f"Lineage-context block disabled for this configuration "
            f"(date-only ablation, as-of {as_of})."
        )
    else:  # snapshot is None despite block_enabled
        parts.append(
            f"Lineage-context snapshot for {country} unavailable as-of "
            f"{as_of}."
        )

    # Time-of-year and data-quality context (always real-time-knowable).
    month = struct.get("month")
    hemisphere = struct.get("hemisphere") or "northern"
    season = struct.get("hemisphere_season") or "season unknown"
    record_lag = struct.get("record_reporting_lag_days")
    hemi_label = "Northern" if hemisphere == "northern" else "Southern"
    month_label = f"calendar month {month}" if month else "calendar month unknown"
    lag_clause = (
        f"reporting lag for this record: {record_lag} days"
        if record_lag is not None else "reporting lag unknown"
    )
    parts.append(
        f"Time-of-year and data-quality context (always real-time-knowable): "
        f"{month_label}; {hemi_label} hemisphere {season}; {lag_clause}."
    )
    return " ".join(parts)


# ======================================================================
# Source Card (surveillance provenance)
# ======================================================================


def build_source_card_struct(record: SampleRecord) -> dict[str, Any]:
    """Build structured source-card fields."""
    host = record.host_system or "unknown"
    common = record.common_name

    if record.dataset == "aiv":
        if host == "wild_birds":
            return {
                "host_system": host,
                "common_name": common,
                "surveillance_type": "opportunistic_wild_bird",
                "bias_factors": [
                    "opportunistic sampling design",
                    "geographic bias toward established monitoring sites",
                    "species-specific detection probability",
                    "variable sampling effort by location and season",
                ],
                "provenance_note": (
                    "Wild bird surveillance metadata sourced from USDA APHIS "
                    "HPAI detections in wild birds dataset."
                ),
            }
        # poultry
        return {
            "host_system": host,
            "common_name": common,
            "surveillance_type": "outbreak_triggered_poultry",
            "bias_factors": [
                "outbreak-triggered sampling",
                "bias toward commercial operations with mandatory reporting",
                "depopulation-event sampling",
                "underrepresentation of small-flock and backyard operations",
            ],
            "provenance_note": (
                "Poultry surveillance metadata sourced from USDA APHIS "
                "confirmed HPAI detections in commercial and backyard flocks."
            ),
        }

    # COVID / Japan — human surveillance.
    # Provenance strings align with Appendix C (Datasets) of the paper:
    #   COVID-US     → NCBI Virus (NCBI / NLM)
    #   COVID-Japan  → MHLW (case counts) + COVID-19 Data Portal JAPAN (DDBJ)
    provenance = (
        "Human case and sequence metadata sourced from the Ministry of "
        "Health, Labour and Welfare and from the COVID-19 Data Portal "
        "JAPAN (DDBJ) public repository."
    ) if record.dataset == "japan" else (
        "Human case and sequence metadata sourced from the NCBI Virus "
        "resource (National Center for Biotechnology Information)."
    )
    return {
        "host_system": "human",
        "common_name": "human",
        "surveillance_type": "human_testing",
        "bias_factors": [
            "variable testing availability and access",
            "home-testing displacement from official counts",
            "sequencing coverage and prioritisation criteria",
            "upload lag to public sequence databases",
            "non-random sample representativeness",
        ],
        "provenance_note": provenance,
    }


def render_source_card(struct: dict[str, Any]) -> str:
    """Render source card to text."""
    parts: list[str] = []

    host = struct["host_system"]
    common = struct["common_name"]
    surv = struct["surveillance_type"]

    if surv == "opportunistic_wild_bird":
        parts.append(
            f"Collected from a wild bird ({common}). "
            f"Wild bird surveillance is opportunistic and subject to "
            f"monitoring-site and species-specific biases."
        )
    elif surv == "outbreak_triggered_poultry":
        parts.append(
            f"Collected from domestic poultry ({common}). "
            f"Poultry detections reflect outbreak-triggered sampling, "
            f"biased toward commercial operations with mandatory reporting."
        )
    else:
        parts.append(
            f"Collected from a human case. "
            f"Surveillance representativeness is subject to testing "
            f"availability, home-testing displacement, sequencing capacity, "
            f"upload lag, and non-random sample selection."
        )

    # Bias factors summary
    biases = struct.get("bias_factors", [])
    if biases:
        bias_str = "; ".join(biases)
        parts.append(f"Key bias factors: {bias_str}.")

    # Provenance
    prov = struct.get("provenance_note", "")
    if prov:
        parts.append(prov)

    return " ".join(parts)


# ======================================================================
# Model Summary
# ======================================================================


def build_model_summary(
    disease_text: str,
    subtype_text: str,
    window_text: str,
    source_text: str,
) -> str:
    """Concatenate all four cards with canonicalization."""
    raw = f"{disease_text} {subtype_text} {window_text} {source_text}"
    return _canonicalize_text(raw)


# ======================================================================
# Orchestrator
# ======================================================================


def build_all_cards(
    record: SampleRecord,
    *,
    include_cards: frozenset[str] | None = None,
    snapshot: Optional["LineageSnapshot"] = None,
    lineage_block_enabled: bool = True,
    reporting_lag_days: int = 14,
) -> CardOutput:
    """Build all four cards for a single sample.

    Returns a :class:`CardOutput` with both structured intermediates
    and rendered text.

    Parameters
    ----------
    include_cards:
        Optional set of card names (subset of :data:`CARD_NAMES`) to
        include in the concatenated ``model_summary``. Cards not in the
        set are still built and rendered (preserving the structured
        outputs and canonical key) but are replaced with empty strings
        before concatenation. ``None`` means include all four cards
        (default behaviour).
    snapshot:
        Optional :class:`LineageSnapshot` for the COVID/Japan
        causal-cutoff path. The orchestrator passes it through to the
        Window Card builder.
    lineage_block_enabled:
        When False the lineage-snapshot block is omitted from the
        Window Card (V1 row (c) date-only ablation).
    reporting_lag_days:
        Δ_lag fallback for the per-record reporting-lag field.
    """
    if include_cards is None:
        include_cards = frozenset(CARD_NAMES)

    # Structs (always built so canonical_key remains stable across ablations)
    disease_struct = build_disease_card_struct(record)
    subtype_struct = build_subtype_card_struct(record)
    window_struct = build_window_card_struct(
        record,
        snapshot=snapshot,
        lineage_block_enabled=lineage_block_enabled,
        reporting_lag_days=reporting_lag_days,
    )
    # Stamp the record's own strain on the window struct so the causal
    # renderer can look up its first-observation date and cumulative
    # count from the snapshot.
    window_struct["record_strain"] = (
        record.strain.strip() if isinstance(record.strain, str) else ""
    )
    source_struct = build_source_card_struct(record)

    # Render
    disease_text = render_disease_card(disease_struct)
    subtype_text = render_subtype_card(subtype_struct)
    window_text = render_window_card(window_struct)
    source_text = render_source_card(source_struct)

    # Summary uses only the cards selected for inclusion. Excluded cards
    # contribute the empty string, which collapses cleanly under
    # _canonicalize_text in build_model_summary.
    summary = build_model_summary(
        disease_text if "disease" in include_cards else "",
        subtype_text if "subtype" in include_cards else "",
        window_text  if "window"  in include_cards else "",
        source_text  if "source"  in include_cards else "",
    )

    # Canonical key
    canonical_key = build_canonical_key(record)

    return CardOutput(
        raw_fields=record.raw_fields_dict(),
        derived_fields=record.derived_fields_dict(),
        disease_card_struct=disease_struct,
        subtype_card_struct=subtype_struct,
        window_card_struct=window_struct,
        source_card_struct=source_struct,
        disease_card=disease_text,
        subtype_card=subtype_text,
        window_card=window_text,
        source_card=source_text,
        model_summary=summary,
        canonical_key=canonical_key,
    )
