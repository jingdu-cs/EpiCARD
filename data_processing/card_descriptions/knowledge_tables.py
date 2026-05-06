"""Static knowledge tables for four-card sample descriptions.

All domain knowledge used by the card builders lives here so that it can be
reviewed, versioned, and overridden via config without touching builder logic.

Design decisions
----------------
* **AIV**: No virulence-increase, host-adaptation-strength, or
  transmissibility-increase claims unless backed by a specific dated
  curated source.  No "spillover" terminology — use "wild-bird ecological
  dissemination", "poultry-system amplification", or "interface between
  wild-bird circulation and poultry outbreak exposure".
* **COVID**: Two-layer lineage table (coarse family + fine-grained dated
  overrides).  No variant-proportion percentages without external
  surveillance data.
* **Seasonal context**: ``seasonal_context_strength = "weak"`` — hedged,
  never stated as fact.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Optional

# ======================================================================
# AIV Knowledge
# ======================================================================

AIV_CLADE_KNOWLEDGE: dict[str, dict[str, str]] = {
    "2.3.4.4b": {
        "lineage": "Goose/Guangdong",
        "pathogenicity": "highly pathogenic (HPAI)",
        "emergence": "clade emerged ~2014; reached North America in late 2021",
        "avian_drift_note": (
            "Antigenic drift from earlier 2.3.4.4 sub-clades"
        ),
        "reassortment_note": (
            "Multiple reassortment events documented with North American "
            "wild bird gene pools"
        ),
        "wild_bird_circulation": (
            "Maintained in wild waterfowl (especially dabbling ducks) as "
            "the primary reservoir for ongoing ecological dissemination"
        ),
        "poultry_amplification": (
            "Domestic poultry outbreaks driven by interface exposure to "
            "wild-bird circulation, with amplification within poultry systems"
        ),
    },
    "Am_nonGsGD": {
        "lineage": "American lineage, non-Goose/Guangdong origin",
        "pathogenicity": "variable; lineage-dependent pathogenicity",
        "emergence": "endemic in North American wild bird populations",
        "avian_drift_note": (
            "Distinct from Goose/Guangdong lineages; evolved independently "
            "within the Americas"
        ),
        "reassortment_note": (
            "Reassortment with co-circulating avian influenza gene segments "
            "in North American wild bird reservoirs"
        ),
        "wild_bird_circulation": (
            "Circulates among North American wild bird populations"
        ),
        "poultry_amplification": (
            "Occasional interface exposure events between wild-bird "
            "circulation and domestic poultry"
        ),
    },
    "EA_nonGsGD": {
        "lineage": "Eurasian lineage, non-Goose/Guangdong origin",
        "pathogenicity": "variable; lineage-dependent pathogenicity",
        "emergence": "endemic in Eurasian wild bird populations",
        "avian_drift_note": (
            "Distinct from Goose/Guangdong lineages; evolved within "
            "Eurasian avian reservoirs"
        ),
        "reassortment_note": (
            "Reassortment with co-circulating Eurasian avian influenza "
            "gene segments"
        ),
        "wild_bird_circulation": (
            "Circulates among Eurasian wild bird populations with periodic "
            "intercontinental movement via migratory flyways"
        ),
        "poultry_amplification": (
            "Interface exposure between wild-bird circulation and domestic "
            "poultry documented in Eurasian contexts"
        ),
    },
}

AIV_SUBTYPE_NOTES: dict[str, str] = {
    "H5N1": "Subtype H5N1: hemagglutinin 5, neuraminidase 1",
    "H5N4": "Subtype H5N4: hemagglutinin 5, neuraminidase 4",
    "H5N6": "Subtype H5N6: hemagglutinin 5, neuraminidase 6",
    "H5": "Subtype H5 without neuraminidase characterisation; partial subtyping only",
}


# ======================================================================
# AIV Seasonal Context (strength = "weak")
# ======================================================================

AIV_SEASONAL_CONTEXT: dict[str, Any] = {
    "seasonal_context_strength": "weak",
    "month_ranges": {
        # (start_month, end_month) -> label
        # Months are inclusive within the tuple key
        (9, 10, 11): "fall migration period",
        (12, 1, 2): "wintering period",
        (3, 4, 5): "spring migration period",
        (6, 7, 8): "breeding season",
    },
}


def get_aiv_seasonal_label(month: int) -> Optional[str]:
    """Return the hedged seasonal label for a given month, or *None*."""
    for months, label in AIV_SEASONAL_CONTEXT["month_ranges"].items():
        if month in months:
            return label
    return None


# ======================================================================
# COVID Knowledge — Layer 1: Coarse Lineage Families
# ======================================================================

COVID_LINEAGE_FAMILIES: list[dict[str, Any]] = [
    # --- Ancestral ---
    {
        "prefixes": ["A"],
        "family": "ancestral_A",
        "who_label": None,
        "dominance_window": (date(2020, 1, 1), date(2020, 9, 1)),
        "fitness_context": "early pandemic lineage with founder-effect dominance",
        "immune_escape": "not applicable; pre-vaccine period",
    },
    {
        "prefixes": ["B"],
        "family": "ancestral_B",
        "who_label": None,
        "dominance_window": (date(2020, 1, 1), date(2020, 12, 1)),
        "fitness_context": "early pandemic lineage co-circulating with A-lineages",
        "immune_escape": "not applicable; pre-vaccine period",
    },
    # --- Alpha ---
    {
        "prefixes": ["B.1.1.7", "Q."],
        "family": "Alpha",
        "who_label": "Alpha",
        "dominance_window": (date(2020, 12, 1), date(2021, 6, 1)),
        "fitness_context": (
            "transmission advantage over ancestral lineages; displaced "
            "earlier variants in many regions"
        ),
        "immune_escape": "minimal immune escape relative to ancestral lineages",
    },
    # --- Beta ---
    {
        "prefixes": ["B.1.351"],
        "family": "Beta",
        "who_label": "Beta",
        "dominance_window": (date(2020, 12, 1), date(2021, 6, 1)),
        "fitness_context": (
            "moderate transmission advantage; geographically limited dominance"
        ),
        "immune_escape": (
            "partial immune escape from natural infection and early vaccines"
        ),
    },
    # --- Gamma ---
    {
        "prefixes": ["P.1"],
        "family": "Gamma",
        "who_label": "Gamma",
        "dominance_window": (date(2020, 12, 1), date(2021, 6, 1)),
        "fitness_context": (
            "transmission advantage; dominant primarily in South America"
        ),
        "immune_escape": (
            "partial immune escape, associated with reinfection events"
        ),
    },
    # --- Delta ---
    {
        "prefixes": ["B.1.617.2", "AY."],
        "family": "Delta",
        "who_label": "Delta",
        "dominance_window": (date(2021, 6, 1), date(2021, 12, 15)),
        "fitness_context": (
            "displaced Alpha and other variants globally via higher "
            "intrinsic transmissibility"
        ),
        "immune_escape": (
            "moderate immune escape; partial reduction in vaccine effectiveness"
        ),
    },
    # --- Omicron BA.1 ---
    {
        "prefixes": ["BA.1"],
        "family": "Omicron_BA1",
        "who_label": "Omicron (BA.1)",
        "dominance_window": (date(2021, 12, 15), date(2022, 3, 1)),
        "fitness_context": (
            "rapid global displacement of Delta via extreme immune evasion "
            "and high intrinsic transmissibility"
        ),
        "immune_escape": (
            "substantial immune escape from prior infection and two-dose "
            "vaccination"
        ),
    },
    # --- Omicron BA.2 ---
    {
        "prefixes": ["BA.2"],
        "family": "Omicron_BA2",
        "who_label": "Omicron (BA.2)",
        "dominance_window": (date(2022, 3, 1), date(2022, 6, 1)),
        "fitness_context": (
            "displaced BA.1 in many regions via incremental "
            "transmissibility advantage"
        ),
        "immune_escape": (
            "substantial immune escape; incremental evasion beyond BA.1"
        ),
    },
    # --- Omicron BA.4/BA.5 ---
    {
        "prefixes": ["BA.4", "BA.5"],
        "family": "Omicron_BA45",
        "who_label": "Omicron (BA.4/BA.5)",
        "dominance_window": (date(2022, 6, 1), date(2022, 11, 1)),
        "fitness_context": (
            "displaced BA.2 via additional immune evasion; BA.5 became "
            "globally dominant"
        ),
        "immune_escape": (
            "significant immune escape from BA.1/BA.2 convalescent and "
            "boosted immunity"
        ),
    },
    # --- Omicron BQ ---
    {
        "prefixes": ["BQ."],
        "family": "Omicron_BQ",
        "who_label": "Omicron (BQ.1 descendant)",
        "dominance_window": (date(2022, 10, 1), date(2023, 1, 15)),
        "fitness_context": (
            "BA.5-descended lineage with convergent immune-evasion mutations"
        ),
        "immune_escape": (
            "pronounced immune escape from bivalent booster-era immunity"
        ),
    },
    # --- Omicron XBB ---
    {
        "prefixes": ["XBB."],
        "family": "Omicron_XBB",
        "who_label": "Omicron (XBB recombinant)",
        "dominance_window": (date(2023, 1, 15), date(2023, 8, 1)),
        "fitness_context": (
            "BA.2-derived recombinant lineage; displaced BQ sub-lineages "
            "globally"
        ),
        "immune_escape": (
            "extreme immune escape; substantial evasion of monovalent and "
            "bivalent booster immunity"
        ),
    },
    # --- Omicron EG (XBB.1.9.2 descendants) ---
    {
        "prefixes": ["EG."],
        "family": "Omicron_EG",
        "who_label": "Omicron (EG.5 / Eris descendants)",
        "dominance_window": (date(2023, 7, 1), date(2023, 12, 1)),
        "fitness_context": (
            "XBB-descended lineage with additional immune-evasion advantage"
        ),
        "immune_escape": (
            "incremental immune escape beyond XBB parent lineage"
        ),
    },
    # --- Omicron JN (BA.2.86 descendants) ---
    {
        "prefixes": ["JN."],
        "family": "Omicron_JN",
        "who_label": "Omicron (JN.1 / BA.2.86 descendants)",
        "dominance_window": (date(2023, 11, 1), None),
        "fitness_context": (
            "BA.2-descended lineage with extensive spike mutations; "
            "displaced XBB/EG-derived lineages"
        ),
        "immune_escape": (
            "substantial immune escape from XBB-era immunity and updated "
            "vaccine formulations"
        ),
    },
    # --- Catch-all Omicron ---
    {
        "prefixes": ["B.1.1.529", "BA."],
        "family": "Omicron_other",
        "who_label": "Omicron",
        "dominance_window": (date(2021, 12, 15), None),
        "fitness_context": (
            "Omicron-lineage variant; specific sub-lineage context unavailable"
        ),
        "immune_escape": (
            "Omicron-era immune escape characteristics"
        ),
    },
]


# ======================================================================
# COVID Knowledge — Layer 2: Fine-Grained Dated Overrides
# ======================================================================

COVID_FINE_GRAINED_OVERRIDES: list[dict[str, Any]] = [
    {
        "exact_lineages": ["BA.2.75"],
        "family_override": "Omicron_BA275",
        "who_label": "Omicron (BA.2.75)",
        "override_fitness": (
            "BA.2-descended lineage with convergent immune-evasion mutations; "
            "competed with BA.5 in some regions"
        ),
        "override_escape": (
            "immune escape profile distinct from BA.5; partial evasion of "
            "BA.1/BA.2 convalescent immunity"
        ),
        "context_window": (date(2022, 6, 1), date(2022, 12, 1)),
    },
    {
        "exact_lineages": ["XBB.1.5"],
        "family_override": "Omicron_XBB15",
        "who_label": "Omicron (XBB.1.5 / Kraken)",
        "override_fitness": (
            "dominant XBB sub-lineage with enhanced ACE2 binding affinity"
        ),
        "override_escape": (
            "extreme immune escape from pre-Omicron and bivalent booster "
            "immunity"
        ),
        "context_window": (date(2023, 1, 1), date(2023, 6, 1)),
    },
    {
        "exact_lineages": ["XBB.1.16"],
        "family_override": "Omicron_XBB116",
        "who_label": "Omicron (XBB.1.16 / Arcturus)",
        "override_fitness": (
            "XBB-descended lineage with additional growth advantage"
        ),
        "override_escape": (
            "comparable immune escape to XBB.1.5 with incremental evasion"
        ),
        "context_window": (date(2023, 3, 1), date(2023, 7, 1)),
    },
]


# ======================================================================
# COVID Pandemic Phases
# ======================================================================

COVID_PANDEMIC_PHASES: list[dict[str, Any]] = [
    {
        "start": date(2020, 1, 1),
        "end": date(2020, 6, 30),
        "phase": "initial pandemic wave",
    },
    {
        "start": date(2020, 7, 1),
        "end": date(2020, 11, 30),
        "phase": "summer-fall resurgence",
    },
    {
        "start": date(2020, 12, 1),
        "end": date(2021, 3, 31),
        "phase": "winter surge and Alpha emergence",
    },
    {
        "start": date(2021, 4, 1),
        "end": date(2021, 6, 30),
        "phase": "vaccine rollout and Alpha-to-Delta transition",
    },
    {
        "start": date(2021, 7, 1),
        "end": date(2021, 11, 30),
        "phase": "Delta wave",
    },
    {
        "start": date(2021, 12, 1),
        "end": date(2022, 3, 31),
        "phase": "Omicron BA.1 wave",
    },
    {
        "start": date(2022, 4, 1),
        "end": date(2022, 6, 30),
        "phase": "BA.2 transition period",
    },
    {
        "start": date(2022, 7, 1),
        "end": date(2022, 11, 30),
        "phase": "BA.4/BA.5 wave",
    },
    {
        "start": date(2022, 12, 1),
        "end": date(2023, 3, 31),
        "phase": "BQ/XBB transition and winter wave",
    },
    {
        "start": date(2023, 4, 1),
        "end": date(2023, 7, 31),
        "phase": "XBB dominance period",
    },
    {
        "start": date(2023, 8, 1),
        "end": None,
        "phase": "EG.5/JN.1 emergence and ongoing evolution",
    },
]


def get_covid_pandemic_phase(d: date) -> Optional[str]:
    """Return the pandemic phase label for a given date, or *None*."""
    for entry in COVID_PANDEMIC_PHASES:
        if entry["end"] is None:
            if d >= entry["start"]:
                return entry["phase"]
        elif entry["start"] <= d <= entry["end"]:
            return entry["phase"]
    return None


# ======================================================================
# COVID Variant Context Lookup
# ======================================================================


def _walk_lineage_prefixes(lineage: str) -> list[str]:
    """Generate progressively shorter prefix candidates.

    ``"BQ.1.1.23"`` -> ``["BQ.1.1.23", "BQ.1.1", "BQ.1", "BQ."]``
    ``"B.1.617.2"`` -> ``["B.1.617.2", "B.1.617", "B.1", "B."]``
    ``"AY.103"``    -> ``["AY.103", "AY."]``
    """
    candidates = [lineage]
    parts = lineage.split(".")
    # Walk up by removing trailing segments
    for i in range(len(parts) - 1, 0, -1):
        prefix = ".".join(parts[:i])
        candidates.append(prefix)
    # Also try the root letter(s) with trailing dot (e.g., "BQ.")
    root = parts[0]
    if f"{root}." not in candidates:
        candidates.append(f"{root}.")
    return candidates


def get_covid_variant_context(
    lineage: str, collection_date: date,
) -> dict[str, Any]:
    """Match a Pango lineage to its variant context at a given date.

    Returns a dict with traceability fields:
    - ``matched_prefix``: the prefix that actually matched
    - ``family``: coarse family name
    - ``who_label``: WHO label or ``None``
    - ``context_window_start`` / ``context_window_end``: dominance window
    - ``confidence``: ``"exact"`` | ``"prefix"`` | ``"family_fallback"``
      | ``"uncharacterised"``
    - ``fallback_level``: 0 = exact, 1 = prefix, 2 = family, 3 = uncharacterised
    - ``fitness_context``: dated fitness statement
    - ``immune_escape``: dated immune-escape characterisation
    """
    clean = lineage.strip()
    if not clean or clean.lower() in ("nan", "not reported"):
        return _uncharacterised_context(lineage)

    # --- Layer 2: check fine-grained overrides first ---
    for entry in COVID_FINE_GRAINED_OVERRIDES:
        if clean in entry["exact_lineages"]:
            start, end = entry["context_window"]
            return {
                "matched_prefix": clean,
                "family": entry["family_override"],
                "who_label": entry.get("who_label"),
                "context_window_start": start,
                "context_window_end": end,
                "confidence": "exact",
                "fallback_level": 0,
                "fitness_context": entry["override_fitness"],
                "immune_escape": entry["override_escape"],
            }

    # --- Layer 1: walk up lineage prefixes ---
    candidates = _walk_lineage_prefixes(clean)

    # Global pass 1: scan ALL walk levels for exact matches first.
    # This ensures specific families (e.g., BA.5) are found before
    # catch-all dot-prefixes (e.g., BA.) that would shadow them.
    for level, candidate in enumerate(candidates):
        for entry in COVID_LINEAGE_FAMILIES:
            for prefix in entry["prefixes"]:
                if candidate == prefix:
                    start, end = entry["dominance_window"]
                    confidence = "exact" if level == 0 else "prefix"
                    return {
                        "matched_prefix": candidate,
                        "family": entry["family"],
                        "who_label": entry.get("who_label"),
                        "context_window_start": start,
                        "context_window_end": end,
                        "confidence": confidence,
                        "fallback_level": min(level, 1),
                        "fitness_context": entry["fitness_context"],
                        "immune_escape": entry["immune_escape"],
                    }

    # Global pass 2: dot-prefix matches — pick the longest (most specific)
    # matching prefix across all candidates to avoid catch-all families
    # like "BA." shadowing more specific families like "BA.4"/"BA.5".
    best_prefix_match: tuple[int, int, dict[str, Any], str] | None = None
    for level, candidate in enumerate(candidates):
        for entry in COVID_LINEAGE_FAMILIES:
            for prefix in entry["prefixes"]:
                if prefix.endswith(".") and candidate.startswith(prefix):
                    # Prefer: (1) longest prefix, (2) lowest walk level
                    if (
                        best_prefix_match is None
                        or len(prefix) > best_prefix_match[0]
                        or (len(prefix) == best_prefix_match[0] and level < best_prefix_match[1])
                    ):
                        best_prefix_match = (len(prefix), level, entry, prefix)
    if best_prefix_match is not None:
        _, level, entry, prefix = best_prefix_match
        start, end = entry["dominance_window"]
        return {
            "matched_prefix": prefix,
            "family": entry["family"],
            "who_label": entry.get("who_label"),
            "context_window_start": start,
            "context_window_end": end,
            "confidence": "prefix" if level <= 1 else "family_fallback",
            "fallback_level": max(level, 1) if level <= 1 else 2,
            "fitness_context": entry["fitness_context"],
            "immune_escape": entry["immune_escape"],
        }

    return _uncharacterised_context(lineage)


def _uncharacterised_context(lineage: str) -> dict[str, Any]:
    """Fallback for lineages that match nothing."""
    return {
        "matched_prefix": lineage,
        "family": "uncharacterised",
        "who_label": None,
        "context_window_start": None,
        "context_window_end": None,
        "confidence": "uncharacterised",
        "fallback_level": 3,
        "fitness_context": (
            "uncharacterised lineage; fitness relative to contemporaneous "
            "variants is not specified"
        ),
        "immune_escape": (
            "immune escape profile not characterised for this lineage"
        ),
    }
