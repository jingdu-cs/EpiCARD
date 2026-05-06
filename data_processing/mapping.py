"""Canonical US location dictionaries and fuzzy state matching.

Provides:
- US_STATES / US_TERRITORIES: abbreviation -> full lowercase name
- CANADIAN_PROVINCES / MEXICAN_STATES: sets for filtering
- resolve_us_state(): fuzzy-tolerant state resolution
- classify_state(): classify raw state string by country
"""

from __future__ import annotations

from typing import Optional

from rapidfuzz import fuzz, process

# ---------------------------------------------------------------------------
# US States: 50 states + DC  (abbreviation -> full lowercase name)
# Replicated from src/data/graph_builder.py to keep this module standalone.
# ---------------------------------------------------------------------------
US_STATES: dict[str, str] = {
    "AL": "alabama", "AK": "alaska", "AZ": "arizona", "AR": "arkansas",
    "CA": "california", "CO": "colorado", "CT": "connecticut", "DE": "delaware",
    "FL": "florida", "GA": "georgia", "HI": "hawaii", "ID": "idaho",
    "IL": "illinois", "IN": "indiana", "IA": "iowa", "KS": "kansas",
    "KY": "kentucky", "LA": "louisiana", "ME": "maine", "MD": "maryland",
    "MA": "massachusetts", "MI": "michigan", "MN": "minnesota", "MS": "mississippi",
    "MO": "missouri", "MT": "montana", "NE": "nebraska", "NV": "nevada",
    "NH": "new hampshire", "NJ": "new jersey", "NM": "new mexico", "NY": "new york",
    "NC": "north carolina", "ND": "north dakota", "OH": "ohio", "OK": "oklahoma",
    "OR": "oregon", "PA": "pennsylvania", "RI": "rhode island", "SC": "south carolina",
    "SD": "south dakota", "TN": "tennessee", "TX": "texas", "UT": "utah",
    "VT": "vermont", "VA": "virginia", "WA": "washington", "WV": "west virginia",
    "WI": "wisconsin", "WY": "wyoming", "DC": "district of columbia",
}

# US territories (kept separate for optional inclusion)
US_TERRITORIES: dict[str, str] = {
    "PR": "puerto rico",
    "VI": "virgin islands",
    "GU": "guam",
    "AS": "american samoa",
    "MP": "northern mariana islands",
}

# Reverse lookup: full lowercase name -> abbreviation
US_STATES_REVERSE: dict[str, str] = {v: k for k, v in US_STATES.items()}
US_TERRITORIES_REVERSE: dict[str, str] = {v: k for k, v in US_TERRITORIES.items()}

# All valid US full names (states + territories) for quick membership checks
_ALL_US_FULL_NAMES: set[str] = set(US_STATES.values())
_ALL_US_TERRITORY_NAMES: set[str] = set(US_TERRITORIES.values())

# Known alternate names for territories (lowercase variant -> canonical name)
_TERRITORY_ALIASES: dict[str, str] = {
    "us virgin islands": "virgin islands",
}

# Combined abbreviation map (for normalization compatibility)
_ALL_US_ABBREVS: dict[str, str] = {**US_STATES, **US_TERRITORIES}

# ---------------------------------------------------------------------------
# Canadian provinces / territories (lowercase)
# ---------------------------------------------------------------------------
CANADIAN_PROVINCES: set[str] = {
    "alberta", "british columbia", "manitoba", "new brunswick",
    "newfoundland and labrador", "northwest territories", "nova scotia",
    "nunavut", "ontario", "prince edward island", "quebec",
    "saskatchewan", "yukon",
}

# ---------------------------------------------------------------------------
# Mexican states (lowercase)
# ---------------------------------------------------------------------------
MEXICAN_STATES: set[str] = {
    "aguascalientes", "baja california", "baja california sur", "campeche",
    "chiapas", "chihuahua", "coahuila", "colima", "durango",
    "guanajuato", "guerrero", "hidalgo", "jalisco", "mexico city",
    "mexico state", "michoacan", "morelos", "nayarit", "nuevo leon",
    "oaxaca", "puebla", "queretaro", "quintana roo", "san luis potosi",
    "sinaloa", "sonora", "tabasco", "tamaulipas", "tlaxcala",
    "veracruz", "yucatan", "zacatecas",
}

# ---------------------------------------------------------------------------
# Known misspellings  (lowercase misspelling -> canonical lowercase name)
# ---------------------------------------------------------------------------
STATE_MISSPELLINGS: dict[str, str] = {
    "calfornia": "california",
    "californa": "california",
    "californi": "california",
    "flordia": "florida",
    "floria": "florida",
    "georiga": "georgia",
    "illnois": "illinois",
    "illinios": "illinois",
    "indana": "indiana",
    "louisianna": "louisiana",
    "massachusets": "massachusetts",
    "massachussets": "massachusetts",
    "michgan": "michigan",
    "minesota": "minnesota",
    "minneosta": "minnesota",
    "missippi": "mississippi",
    "mississipi": "mississippi",
    "missourri": "missouri",
    "montanta": "montana",
    "nebreska": "nebraska",
    "neveda": "nevada",
    "new hamshire": "new hampshire",
    "new jersy": "new jersey",
    "new mexio": "new mexico",
    "northcarolina": "north carolina",
    "north carolin": "north carolina",
    "north dakot": "north dakota",
    "oklahom": "oklahoma",
    "oregn": "oregon",
    "oregeon": "oregon",
    "pensylvania": "pennsylvania",
    "pennsylvana": "pennsylvania",
    "rhode isand": "rhode island",
    "south carolin": "south carolina",
    "south dakot": "south dakota",
    "tennesse": "tennessee",
    "texs": "texas",
    "virgina": "virginia",
    "washingon": "washington",
    "washinton": "washington",
    "west virgina": "west virginia",
    "wisconsn": "wisconsin",
    "wyomng": "wyoming",
}

# Pre-built list of all US state full names for fuzzy matching
_FUZZY_CHOICES: list[str] = sorted(_ALL_US_FULL_NAMES)


def resolve_us_state(raw: str, fuzzy_threshold: int = 85) -> Optional[str]:
    """Resolve a raw state string to a canonical lowercase US state name.

    Resolution order:
    1. Exact match on 2-letter abbreviation (case-insensitive)
    2. Exact match on full name (case-insensitive, stripped)
    3. Known misspelling lookup
    4. Fuzzy match via rapidfuzz (score >= *fuzzy_threshold*)

    Returns the canonical lowercase name, or ``None`` if unresolvable.
    """
    if not raw or not isinstance(raw, str):
        return None

    s = raw.strip()
    if not s or s.lower() == "nan":
        return None

    # 1. Abbreviation
    upper = s.upper()
    if upper in US_STATES:
        return US_STATES[upper]

    # 2. Exact full-name match
    lower = s.lower()
    if lower in _ALL_US_FULL_NAMES:
        return lower

    # 3. Misspelling table
    if lower in STATE_MISSPELLINGS:
        return STATE_MISSPELLINGS[lower]

    # 4. Fuzzy match
    result = process.extractOne(
        lower, _FUZZY_CHOICES, scorer=fuzz.ratio, score_cutoff=fuzzy_threshold
    )
    if result is not None:
        return result[0]

    return None


def resolve_us_territory(raw: str) -> Optional[str]:
    """Resolve a raw string to a US territory name, or None."""
    if not raw or not isinstance(raw, str):
        return None
    s = raw.strip()
    if not s or s.lower() == "nan":
        return None

    upper = s.upper()
    if upper in US_TERRITORIES:
        return US_TERRITORIES[upper]

    lower = s.lower()
    if lower in _ALL_US_TERRITORY_NAMES:
        return lower

    # Check aliases (e.g., "us virgin islands" -> "virgin islands")
    if lower in _TERRITORY_ALIASES:
        return _TERRITORY_ALIASES[lower]

    return None


def classify_state(
    raw: str, fuzzy_threshold: int = 85
) -> tuple[str, Optional[str]]:
    """Classify a raw state string by country/region.

    Returns
    -------
    (classification, resolved_name)
        *classification* is one of:
        ``"us_state"``, ``"us_territory"``, ``"canadian"``,
        ``"mexican"``, ``"unknown"``.
        *resolved_name* is the canonical lowercase name when the
        classification is ``"us_state"`` or ``"us_territory"``,
        otherwise ``None``.
    """
    if not raw or not isinstance(raw, str):
        return ("unknown", None)

    s = raw.strip()
    if not s or s.lower() == "nan":
        return ("unknown", None)

    # Try US state first (most common)
    us = resolve_us_state(s, fuzzy_threshold=fuzzy_threshold)
    if us is not None:
        return ("us_state", us)

    # Try US territory
    terr = resolve_us_territory(s)
    if terr is not None:
        return ("us_territory", terr)

    # Check Canadian
    lower = s.lower()
    if lower in CANADIAN_PROVINCES:
        return ("canadian", None)

    # Check Mexican
    if lower in MEXICAN_STATES:
        return ("mexican", None)

    return ("unknown", None)
