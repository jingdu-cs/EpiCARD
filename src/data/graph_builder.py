"""Case- and location-feature builders for HierEpiGNN.

Contains:
- LocationNormalizer: normalizes location strings from different data sources
- HostEncoder: one-hot encoding for host categories
- StrainEncoder: hash-based strain encoding
- sinusoidal_temporal_encoding(): temporal feature encoding
- GraphBuilder: assembles per-window case and location node-feature tensors
"""

from __future__ import annotations

import hashlib
import logging
import math
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# US State abbreviation mapping (2-letter -> full lowercase name)
# ---------------------------------------------------------------------------
US_STATE_ABBREV: dict[str, str] = {
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
    "PR": "puerto rico", "VI": "virgin islands", "GU": "guam",
    "AS": "american samoa", "MP": "northern mariana islands",
}

# Suffixes to strip from county names during normalization
_COUNTY_SUFFIXES = ("county", "parish", "borough", "census area", "municipality")


# ============================================================================
# LocationNormalizer
# ============================================================================

class LocationNormalizer:
    """Normalize location strings from heterogeneous data sources.

    Handles:
    - Trailing whitespace / case normalization
    - "Nan" sentinel -> None
    - Numeric county values -> "unknown"
    - 2-letter state abbreviations -> full lowercase names
    - Removal of county suffixes (County, Parish, Borough, Census Area)
    - Abundance key parsing ("state|county")
    - FIPS-based records (covid_confirmed.csv)
    """

    def normalize(
        self, state: str, county: str
    ) -> tuple[Optional[str], Optional[str]]:
        """Normalize a (state, county) pair.

        Returns (normalized_state, normalized_county).
        Either value may be ``None`` when input is missing.
        """
        norm_state = self._normalize_state(state)
        norm_county = self._normalize_county(county)
        return norm_state, norm_county

    # ------------------------------------------------------------------
    # Abundance key parsing
    # ------------------------------------------------------------------

    def from_abundance_key(self, key: str) -> tuple[str, str]:
        """Parse ``"state|county"`` abundance key.

        Returns (state, county) both lowercase, suffixes already stripped.
        """
        parts = key.split("|")
        if len(parts) != 2:
            logger.warning("Unexpected abundance key format: %s", key)
            return (key.lower().strip(), "unknown")
        raw_state, raw_county = parts
        return (
            self._normalize_state(raw_state) or "unknown",
            self._normalize_county(raw_county) or "unknown",
        )

    # ------------------------------------------------------------------
    # FIPS / covid_confirmed.csv
    # ------------------------------------------------------------------

    def from_fips(
        self, fips: int, county_name: str, state_abbrev: str
    ) -> tuple[str, str]:
        """Normalize a record from covid_confirmed.csv.

        Parameters
        ----------
        fips : int
            County FIPS code (unused for normalization, kept for API).
        county_name : str
            County name, possibly with "County" suffix.
        state_abbrev : str
            Two-letter state abbreviation **or** full state name.
        """
        norm_state = self._normalize_state(state_abbrev)
        norm_county = self._normalize_county(county_name)
        return (norm_state or "unknown", norm_county or "unknown")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_state(raw: Any) -> Optional[str]:
        if raw is None:
            return None
        s = str(raw).strip()
        if s.lower() in ("nan", ""):
            return None
        # Check abbreviation mapping
        upper = s.upper()
        if upper in US_STATE_ABBREV:
            return US_STATE_ABBREV[upper]
        return s.lower()

    @staticmethod
    def _normalize_county(raw: Any) -> Optional[str]:
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
        # Strip known suffixes
        for suffix in _COUNTY_SUFFIXES:
            if s.endswith(suffix):
                s = s[: -len(suffix)].strip()
                break
        return s


# ============================================================================
# HostEncoder
# ============================================================================

class HostEncoder:
    """One-hot encoding for (category, subcategory) host labels.

    Call :meth:`fit` once with all observed labels, then :meth:`encode` per case.
    Output dimension = ``host_encoding_dim`` from config.
    """

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._cat_vocab: dict[str, int] = {}
        self._subcat_vocab: dict[str, int] = {}
        self._fitted = False

    def fit(
        self,
        categories: list[str],
        subcategories: list[str],
    ) -> None:
        """Build vocabulary from observed category/subcategory values."""
        unique_cats = sorted(set(c.lower().strip() for c in categories if c))
        unique_subcats = sorted(set(s.lower().strip() for s in subcategories if s))

        self._cat_vocab = {c: i for i, c in enumerate(unique_cats)}
        self._subcat_vocab = {s: i for i, s in enumerate(unique_subcats)}
        self._fitted = True

        logger.info(
            "HostEncoder fitted: %d categories, %d subcategories -> dim %d",
            len(self._cat_vocab), len(self._subcat_vocab), self.dim,
        )

    def encode(self, category: str, subcategory: str) -> Tensor:
        """Encode a single (category, subcategory) pair.

        Returns
        -------
        Tensor
            Shape ``[dim]``. First half encodes category, second half encodes
            subcategory (both one-hot, truncated/padded to fit ``dim``).
        """
        if not self._fitted:
            raise RuntimeError("HostEncoder.fit() must be called before encode()")

        vec = torch.zeros(self.dim, dtype=torch.float32)  # [dim]

        cat_slots = self.dim // 2
        subcat_slots = self.dim - cat_slots

        cat_key = category.lower().strip() if category else ""
        subcat_key = subcategory.lower().strip() if subcategory else ""

        cat_idx = self._cat_vocab.get(cat_key)
        if cat_idx is not None and cat_idx < cat_slots:
            vec[cat_idx] = 1.0

        subcat_idx = self._subcat_vocab.get(subcat_key)
        if subcat_idx is not None and subcat_idx < subcat_slots:
            vec[cat_slots + subcat_idx] = 1.0

        return vec  # [dim]


# ============================================================================
# StrainEncoder
# ============================================================================

class StrainEncoder:
    """Hash-based strain encoding (no learned parameters).

    Uses MD5 hash of the strain string to produce a deterministic
    real-valued vector of dimension ``dim``.
    """

    def __init__(self, dim: int) -> None:
        self.dim = dim

    def encode(self, strain: str) -> Tensor:
        """Encode a single strain string.

        Returns
        -------
        Tensor
            Shape ``[dim]``, values in [0, 1).
        """
        if not strain or str(strain).lower().strip() in ("nan", ""):
            return torch.zeros(self.dim, dtype=torch.float32)  # [dim]

        # MD5 gives 16 bytes = 128 bits; cycle if dim > 128
        digest = hashlib.md5(strain.encode("utf-8")).digest()
        byte_arr = np.frombuffer(digest, dtype=np.uint8)  # [16]

        # Tile to at least dim bytes, then truncate
        repeats = math.ceil(self.dim / len(byte_arr))
        extended = np.tile(byte_arr, repeats)[:self.dim]  # [dim]

        # Normalize to [0, 1)
        vec = extended.astype(np.float32) / 255.0  # [dim]
        return torch.from_numpy(vec)  # [dim]


# ============================================================================
# sinusoidal_temporal_encoding
# ============================================================================

def sinusoidal_temporal_encoding(
    dates: list[pd.Timestamp],
    ref_date: pd.Timestamp,
    dim: int,
) -> Tensor:
    """Sinusoidal positional encoding for temporal features.

    Encodes each date as ``days_since_ref_date`` using sine/cosine
    frequencies at multiple scales (analogous to Transformer PE).

    Parameters
    ----------
    dates : list[pd.Timestamp]
        Collection dates for each case.
    ref_date : pd.Timestamp
        Window start date (reference point, day 0).
    dim : int
        Output dimensionality (must be even).

    Returns
    -------
    Tensor
        Shape ``[N, dim]`` where N = len(dates).
    """
    n = len(dates)
    if n == 0:
        return torch.zeros(0, dim, dtype=torch.float32)  # [0, dim]

    # Compute days since reference date  # [N]
    days = torch.tensor(
        [(d - ref_date).days for d in dates],
        dtype=torch.float32,
    )  # [N]

    # Frequency bands  # [dim//2]
    half_dim = dim // 2
    freq = torch.exp(
        torch.arange(0, half_dim, dtype=torch.float32)
        * -(math.log(10000.0) / half_dim)
    )  # [dim//2]

    # Outer product: [N, 1] * [1, dim//2] -> [N, dim//2]
    angles = days.unsqueeze(1) * freq.unsqueeze(0)  # [N, dim//2]

    # Interleave sin and cos  # [N, dim]
    encoding = torch.zeros(n, dim, dtype=torch.float32)  # [N, dim]
    encoding[:, 0::2] = torch.sin(angles)  # [N, dim//2]
    encoding[:, 1::2] = torch.cos(angles)  # [N, dim//2]

    return encoding  # [N, dim]


# ============================================================================
# GraphBuilder
# ============================================================================

class GraphBuilder:
    """Builds per-window case- and location-node feature tensors.

    All feature dimensions are read from *cfg* (the ``data`` sub-dict of
    ``configs/default.yaml``).
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        """
        Parameters
        ----------
        cfg : dict
            The ``data`` section of the configuration.
        """
        self.cfg = cfg
        self.temporal_dim: int = cfg["temporal_encoding_dim"]
        self.host_dim: int = cfg["host_encoding_dim"]
        self.strain_dim: int = cfg["strain_encoding_dim"]
        self.genetic_placeholder_dim: int = cfg["genetic_feat_placeholder_dim"]

        self.host_encoder = HostEncoder(dim=self.host_dim)
        self.strain_encoder = StrainEncoder(dim=self.strain_dim)

        # Strain LLM embeddings (optional, precomputed). Two on-disk formats
        # are supported:
        #
        # * ``v1`` — ``dict[unique_id, Tensor]`` (legacy, retrospective cards).
        # * ``v2_as_of`` — ``{"format": "v2_as_of", "embeddings": dict[
        #   "{unique_id}|{as_of_iso}", Tensor], "dim": int, ...}`` (causal-cutoff).
        embedding_path = cfg.get("strain_embedding_file")
        if embedding_path is not None:
            logger.info("Loading strain LLM embeddings from %s", embedding_path)
            loaded = torch.load(
                embedding_path, map_location="cpu", weights_only=True,
            )
            if isinstance(loaded, dict) and loaded.get("format") == "v2_as_of":
                self._embedding_format: str = "v2_as_of"
                self.strain_embeddings: dict[str, Tensor] | None = loaded["embeddings"]
                self.strain_embedding_dim: int = int(
                    loaded.get("dim") or cfg.get("strain_embedding_dim", 4096)
                )
                logger.info(
                    "Embedding cache format=v2_as_of (%d entries)",
                    len(self.strain_embeddings),
                )
            else:
                self._embedding_format = "v1"
                self.strain_embeddings = loaded
                self.strain_embedding_dim = cfg.get("strain_embedding_dim", 4096)
        else:
            self.strain_embeddings = None
            self.strain_embedding_dim = 0
            self._embedding_format = "v1"

        # Embedding lookup key: "Unique_Identifier" (sample-level four-card embeddings).
        self._embedding_key_col: str = cfg.get("strain_embedding_key", "Unique_Identifier")

        # Ablation: replace every per-sample LLM embedding with a zero
        # vector at graph-build time. The strain_proj Linear layer in the
        # model still runs, isolating "no LLM information" from "no
        # architecture component". Requires strain_embedding_file to be
        # set so the dimension is known.
        self.zero_strain_embeddings: bool = bool(
            cfg.get("zero_strain_embeddings", False)
        )
        if self.zero_strain_embeddings and self.strain_embeddings is None:
            raise ValueError(
                "data.zero_strain_embeddings=True requires "
                "data.strain_embedding_file to be set so the embedding "
                "dimension is known."
            )
        if self.zero_strain_embeddings:
            logger.info(
                "Ablation: zero_strain_embeddings=True; per-sample "
                "embeddings will be overwritten with zeros at graph "
                "build time."
            )

    # ------------------------------------------------------------------
    # Case graph
    # ------------------------------------------------------------------

    def build_case_graph(
        self,
        cases: pd.DataFrame,
        window_start: pd.Timestamp,
        location_index: dict[tuple[str, str], int],
    ) -> dict[str, Tensor]:
        """Build the case-level node tensors for one time window.

        Parameters
        ----------
        cases : pd.DataFrame
            Rows of case_data filtered to the current window. Expected columns:
            ``Collection_Date``, ``Category``, ``Subcategory``, ``Strain``,
            plus normalized ``state`` and ``county``.
        window_start : pd.Timestamp
            Start of the time window (for temporal encoding reference).
        location_index : dict
            ``(state, county) -> location_node_index`` mapping for batch assignment.

        Returns
        -------
        dict
            Keys: ``x``, ``strain_emb``, ``batch``. ``strain_emb`` is ``None``
            when no strain embedding file is configured, otherwise
            ``Tensor[N, strain_embedding_dim]``.
        """
        n_cases = len(cases)
        logger.debug("Building case features: %d cases", n_cases)

        if n_cases == 0:
            feat_dim = (
                self.temporal_dim + self.host_dim
                + self.strain_dim + self.genetic_placeholder_dim
            )
            strain_emb_empty: Tensor | None = None
            if self.strain_embeddings is not None:
                strain_emb_empty = torch.zeros(
                    0, self.strain_embedding_dim, dtype=torch.float32,
                )
            return {
                "x": torch.zeros(0, feat_dim, dtype=torch.float32),  # [0, D]
                "strain_emb": strain_emb_empty,                       # [0, E] or None
                "batch": torch.zeros(0, dtype=torch.long),            # [0]
            }

        # --- Temporal encoding ---  # [N, temporal_dim]
        dates = pd.to_datetime(cases["Collection_Date"], format="mixed", dayfirst=True)
        temporal_feats = sinusoidal_temporal_encoding(
            dates.tolist(), window_start, self.temporal_dim,
        )  # [N, temporal_dim]

        # --- Host encoding ---  # [N, host_dim]
        host_feats = torch.stack([
            self.host_encoder.encode(
                str(row.get("Category", "")),
                str(row.get("Subcategory", "")),
            )
            for _, row in cases.iterrows()
        ], dim=0)  # [N, host_dim]

        # --- Strain encoding ---  # [N, strain_dim]
        strain_feats = torch.stack([
            self.strain_encoder.encode(str(row.get("Strain", "")))
            for _, row in cases.iterrows()
        ], dim=0)  # [N, strain_dim]

        # --- Genetic placeholder ---  # [N, genetic_placeholder_dim]
        genetic_feats = torch.zeros(
            n_cases, self.genetic_placeholder_dim, dtype=torch.float32
        )  # [N, genetic_placeholder_dim]

        # --- Strain LLM embeddings (separate tensor) ---
        strain_emb: Tensor | None = None
        if self.strain_embeddings is not None:
            unknown_emb = self.strain_embeddings.get(
                "__UNKNOWN__",
                torch.zeros(self.strain_embedding_dim, dtype=torch.float32),
            )
            emb_list: list[Tensor] = []
            v2 = self._embedding_format == "v2_as_of"
            n_missing = 0
            for _, row in cases.iterrows():
                base_key = str(row.get(self._embedding_key_col, ""))
                if v2:
                    as_of = row.get("as_of_date")
                    if isinstance(as_of, pd.Timestamp):
                        as_of_iso = as_of.date().isoformat()
                    elif hasattr(as_of, "isoformat"):
                        as_of_iso = as_of.isoformat()
                    else:
                        as_of_iso = str(as_of) if as_of else ""
                    key = f"{base_key}|{as_of_iso}" if as_of_iso else base_key
                else:
                    key = base_key
                emb = self.strain_embeddings.get(key)
                if emb is None:
                    emb = unknown_emb
                    n_missing += 1
                emb_list.append(emb)
            if v2 and n_missing:
                logger.warning(
                    "v2_as_of embedding lookup: %d/%d cases fell back to "
                    "__UNKNOWN__ (missing composite key).",
                    n_missing, len(emb_list),
                )
            strain_emb = torch.stack(emb_list, dim=0)  # [N, strain_embedding_dim]
            if self.zero_strain_embeddings:
                strain_emb = torch.zeros_like(strain_emb)

        # Concatenate all features  # [N, D_case]
        x = torch.cat(
            [temporal_feats, host_feats, strain_feats, genetic_feats], dim=1
        )  # [N, D_case]

        # --- Batch (case -> location mapping) ---
        batch = self.build_case_to_location_mapping(
            cases, location_index
        )  # [N]

        logger.debug("Case features built: x=%s, batch=%s", list(x.shape), list(batch.shape))

        return {
            "x": x,                   # [N, D_case]
            "strain_emb": strain_emb,  # [N, strain_embedding_dim] or None
            "batch": batch,            # [N]
        }

    # ------------------------------------------------------------------
    # Location features
    # ------------------------------------------------------------------

    def build_location_graph(
        self,
        location_features: Tensor,
    ) -> dict[str, Tensor]:
        """Wrap location-level features into the model's input dict shape."""
        logger.debug("Wrapping location features: shape=%s", list(location_features.shape))
        return {"x": location_features}  # [N_loc, D_loc]

    # ------------------------------------------------------------------
    # Case-to-location batch mapping
    # ------------------------------------------------------------------

    @staticmethod
    def build_case_to_location_mapping(
        cases: pd.DataFrame,
        location_index: dict[tuple[str, str], int],
    ) -> Tensor:
        """Map each case to its location node index.

        Parameters
        ----------
        cases : pd.DataFrame
            Must have ``state`` and ``county`` columns (already normalized).
        location_index : dict
            ``(state, county) -> int`` location index.

        Returns
        -------
        Tensor
            Shape ``[N_cases]``, values in ``[0, N_loc)``.
            Cases with unknown location map to index 0 (with a warning).
        """
        batch_list: list[int] = []
        unknown_count = 0
        for _, row in cases.iterrows():
            raw_state = row.get("state", "")
            raw_county = row.get("county", "")
            key = (
                "unknown" if pd.isna(raw_state) or raw_state == "" else str(raw_state),
                "unknown" if pd.isna(raw_county) or raw_county == "" else str(raw_county),
            )
            loc_idx = location_index.get(key)
            if loc_idx is None:
                loc_idx = 0
                unknown_count += 1
            batch_list.append(loc_idx)

        if unknown_count > 0:
            logger.warning(
                "%d cases mapped to fallback location index 0", unknown_count
            )

        return torch.tensor(batch_list, dtype=torch.long)  # [N_cases]
