"""Location string normalization functions.

Replicates the normalization logic from ``src/data/graph_builder.py``
(``LocationNormalizer``) to ensure byte-identical output for US inputs.
This module is standalone and does not import from ``src/``.
"""

from __future__ import annotations

from typing import Any, Optional

from data_processing.mapping import _ALL_US_ABBREVS

# Same suffix list as graph_builder.py line 49
COUNTY_SUFFIXES: tuple[str, ...] = (
    "county",
    "parish",
    "borough",
    "census area",
    "municipality",
)


def normalize_state(raw: Any) -> Optional[str]:
    """Normalize a state value to full lowercase name.

    Matches ``LocationNormalizer._normalize_state`` exactly:
    - Strip whitespace
    - ``"Nan"`` / ``""`` / ``None`` -> ``None``
    - 2-letter abbreviation -> full lowercase name
    - Otherwise -> ``lowercase.strip()``
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if s.lower() in ("nan", ""):
        return None
    upper = s.upper()
    if upper in _ALL_US_ABBREVS:
        return _ALL_US_ABBREVS[upper]
    return s.lower()


def normalize_county(raw: Any) -> Optional[str]:
    """Normalize a county value.

    Matches ``LocationNormalizer._normalize_county`` exactly:
    - Strip whitespace
    - ``"Nan"`` / ``""`` / ``None`` -> ``None``
    - Numeric values -> ``"unknown"``
    - Strip known suffixes (county, parish, borough, census area, municipality)
    - Return lowercase
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if s.lower() in ("nan", ""):
        return None
    # Numeric county -> "unknown"
    try:
        float(s)
        return "unknown"
    except ValueError:
        pass
    s = s.lower()
    for suffix in COUNTY_SUFFIXES:
        if s.endswith(suffix):
            s = s[: -len(suffix)].strip()
            break
    # Post-strip check: suffix removal may expose "nan" (e.g., "Nan County")
    if s in ("nan", ""):
        return None
    return s


def normalize_abundance_key(key: str) -> tuple[str, str]:
    """Parse ``"state|county"`` abundance key.

    Matches ``LocationNormalizer.from_abundance_key`` exactly.
    Returns ``(state, county)`` both normalized.
    """
    parts = key.split("|")
    if len(parts) != 2:
        return (key.lower().strip(), "unknown")
    raw_state, raw_county = parts
    return (
        normalize_state(raw_state) or "unknown",
        normalize_county(raw_county) or "unknown",
    )
