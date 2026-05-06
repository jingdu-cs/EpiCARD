"""Generate sample-level LLM embeddings using four-card structured descriptions.

Offline preprocessing script that:
1. Reads ``case_data.csv`` for a given dataset (aiv / covid / japan)
2. Builds a structured four-card description for each sample
3. Deduplicates by canonical key (structural, not text-based)
4. Encodes unique descriptions with Meta-Llama-3-8B
5. Saves per-sample embeddings as a ``.pt`` file

Usage::

    python -m data_processing.generate_sample_embeddings \\
        --dataset japan \\
        --input-csv data/processed/japan/case_data.csv \\
        --output data/processed/japan/sample_embeddings_llama3.pt \\
        --model-name meta-llama/Meta-Llama-3-8B \\
        --batch-size 4 \\
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import torch

from data_processing.card_descriptions import (
    CARD_NAMES,
    build_sample_descriptions,
    build_sample_descriptions_with_as_of,
)
from data_processing.utils import setup_logging, write_processing_report

logger = logging.getLogger("data_processing")


def _parse_as_of_dates(arg: str) -> List[date]:
    """Parse the --as-of-dates argument.

    Accepts either a comma-separated list of ISO dates or a path to a JSON
    file containing a list of ISO date strings. Returns a sorted, deduped
    list of ``date`` objects.
    """
    arg = arg.strip()
    if not arg:
        raise ValueError("--as-of-dates is empty.")
    text: str
    if Path(arg).expanduser().exists():
        text = Path(arg).expanduser().read_text()
        try:
            iso_list = json.loads(text)
        except json.JSONDecodeError:
            iso_list = [line.strip() for line in text.splitlines() if line.strip()]
    else:
        iso_list = [s.strip() for s in arg.split(",") if s.strip()]
    out: list[date] = []
    for iso in iso_list:
        out.append(date.fromisoformat(iso))
    return sorted(set(out))


def encode_descriptions(
    descriptions: Dict[str, str],
    model_name: str,
    batch_size: int,
    device: str,
    max_length: int = 512,
) -> Dict[str, torch.Tensor]:
    """Encode descriptions into L2-normalised embeddings via mean pooling.

    Returns ``{key_string: Tensor[hidden_dim]}`` with unit-norm vectors.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading tokenizer from %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading model from %s (float16)", model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16
    )
    model.to(device)
    model.eval()

    keys = list(descriptions.keys())
    texts = [descriptions[k] for k in keys]
    embeddings: Dict[str, torch.Tensor] = {}

    logger.info(
        "Encoding %d descriptions in batches of %d on %s",
        len(texts), batch_size, device,
    )

    for start in range(0, len(texts), batch_size):
        batch_keys = keys[start : start + batch_size]
        batch_texts = texts[start : start + batch_size]

        encoded = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        with torch.no_grad():
            outputs = model(**encoded, output_hidden_states=True)

        last_hidden = outputs.hidden_states[-1]  # [batch, seq_len, D]
        attention_mask = encoded["attention_mask"]  # [batch, seq_len]

        mask_expanded = attention_mask.unsqueeze(-1).float()
        summed = (last_hidden * mask_expanded).sum(dim=1)
        counts = mask_expanded.sum(dim=1).clamp(min=1.0)
        pooled = summed / counts
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)

        for i, key in enumerate(batch_keys):
            embeddings[key] = pooled[i].cpu()

        logger.info(
            "  Encoded batch %d-%d / %d",
            start + 1, min(start + batch_size, len(texts)), len(texts),
        )

    return embeddings


def main() -> None:
    """CLI entry point for sample-level embedding generation."""
    parser = argparse.ArgumentParser(
        description="Generate sample-level LLM embeddings with four-card descriptions.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["aiv", "covid", "japan"],
        required=True,
        help="Dataset type: 'aiv', 'covid', or 'japan'.",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=None,
        help="Path to case_data.csv. Default: data/processed/{dataset}/case_data.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .pt path. Default: data/processed/{dataset}/sample_embeddings_llama3.pt",
    )
    parser.add_argument(
        "--descriptions-output",
        type=Path,
        default=None,
        help="Output .json path for descriptions. Default: data/processed/{dataset}/sample_descriptions.json",
    )
    parser.add_argument(
        "--no-descriptions-output",
        action="store_true",
        help=(
            "Skip writing the human-readable descriptions JSON entirely "
            "(saves disk for large causal-cutoff runs). Mutually exclusive "
            "with --descriptions-output."
        ),
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="meta-llama/Meta-Llama-3-8B",
        help="HuggingFace model identifier (default: meta-llama/Meta-Llama-3-8B).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for LLM encoding (default: 4).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=1024,
        help="Max token length for LLM encoding (default: 1024).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for inference (default: cuda if available, else cpu).",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("data/processed/logs"),
        help="Directory for log files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate descriptions only; skip LLM encoding.",
    )
    parser.add_argument(
        "--exclude-card",
        action="append",
        default=[],
        choices=list(CARD_NAMES),
        help=(
            "Repeatable. Exclude the named card from the concatenated "
            "model_summary before LLM encoding (used for per-card ablations). "
            "Default output filename is suffixed with '_no_<card>' for each "
            "excluded card unless --output is given explicitly."
        ),
    )
    parser.add_argument(
        "--output-tag",
        type=str,
        default="llama3_8B",
        help=(
            "Base tag for the default output filename: "
            "data/processed/{dataset}/sample_embeddings_{tag}[<excluded suffix>].pt. "
            "Ignored when --output is given explicitly."
        ),
    )
    parser.add_argument(
        "--causal-cutoff",
        action="store_true",
        help=(
            "Enable the causal-cutoff Window Card path. Requires --as-of-dates. "
            "Output is written in v2 format keyed by (Unique_Identifier|as_of_iso) "
            "and the AIV path is unaffected (no snapshot is computed for AIV)."
        ),
    )
    parser.add_argument(
        "--as-of-dates",
        type=str,
        default=None,
        help=(
            "Comma-separated list of ISO as-of dates (YYYY-MM-DD), or path to "
            "a JSON file containing a list of ISO date strings. Required with "
            "--causal-cutoff."
        ),
    )
    parser.add_argument(
        "--reporting-lag-days",
        type=int,
        default=14,
        help=(
            "Δ_lag fallback when Report Date is missing (default: 14). "
            "Sweep this knob for the V2 as-of-date sensitivity ablation."
        ),
    )
    parser.add_argument(
        "--lineage-window-days",
        type=int,
        default=28,
        help="Past-window length for top-3 lineage frequencies (default: 28).",
    )
    parser.add_argument(
        "--no-lineage-block",
        dest="lineage_block_enabled",
        action="store_false",
        default=True,
        help=(
            "Disable the lineage-snapshot block (V1 row (c) date-only ablation). "
            "Causal-cutoff path only."
        ),
    )
    parser.add_argument(
        "--trend-threshold",
        type=float,
        default=0.10,
        help="±threshold for the rising/declining trend classifier (default: 0.10).",
    )
    parser.add_argument(
        "--window-size-days",
        type=int,
        default=None,
        help=(
            "Case-window length in days. When given, only emit cards for "
            "(record, as_of) pairs where the record falls into the case "
            "window ending at as_of + reporting_lag_days; this matches the "
            "training-time graph builder. Strongly recommended for the "
            "causal-cutoff path — without it the (record × as_of) Cartesian "
            "product produces a much larger cache. Examples: 56 for the "
            "default 8-week window (COVID/Japan), 8*7 if running with the "
            "weekly time_unit. Match the value used by the trainer."
        ),
    )
    args = parser.parse_args()

    excluded_cards = sorted(set(args.exclude_card))
    include_cards = frozenset(c for c in CARD_NAMES if c not in excluded_cards)
    suffix = "".join(f"_no_{c}" for c in excluded_cards)

    if args.causal_cutoff:
        if not args.as_of_dates:
            parser.error("--causal-cutoff requires --as-of-dates.")
        suffix = f"{suffix}_causal_v2"
        if not args.lineage_block_enabled:
            suffix = f"{suffix}_no_lineage"

    # Resolve defaults
    if args.input_csv is None:
        args.input_csv = Path(f"data/processed/{args.dataset}/case_data.csv")
    if args.output is None:
        args.output = Path(
            f"data/processed/{args.dataset}/"
            f"sample_embeddings_{args.output_tag}{suffix}.pt",
        )
    if args.no_descriptions_output:
        if args.descriptions_output is not None:
            parser.error(
                "--no-descriptions-output is mutually exclusive with --descriptions-output."
            )
        args.descriptions_output = None
    elif args.descriptions_output is None:
        args.descriptions_output = Path(
            f"data/processed/{args.dataset}/"
            f"sample_descriptions{suffix}.json",
        )

    log = setup_logging(args.log_dir)
    log.info("=" * 60)
    log.info("Generating sample-level embeddings for dataset=%s", args.dataset)
    log.info("=" * 60)

    start_time = time.time()

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    log.info("Reading %s", args.input_csv)
    df = pd.read_csv(args.input_csv, dtype=str)
    log.info("Loaded %d rows", len(df))

    # ------------------------------------------------------------------
    # 2. Build four-card descriptions
    # ------------------------------------------------------------------
    log.info(
        "Building four-card descriptions (causal_cutoff=%s, excluded cards: %s)...",
        args.causal_cutoff, excluded_cards or "<none>",
    )

    # Map: cache_key (str) -> (sample_id, as_of_iso_or_empty), card_output
    # On the legacy path the cache_key is the sample's Unique_Identifier.
    # On the v2 path it is f"{Unique_Identifier}|{as_of_iso}".
    descriptions: dict[str, "object"] = {}
    cache_format = "v1"

    if args.causal_cutoff:
        cache_format = "v2_as_of"
        as_of_dates = _parse_as_of_dates(args.as_of_dates)
        log.info("Causal-cutoff mode: %d as-of dates", len(as_of_dates))
        composite = build_sample_descriptions_with_as_of(
            df, args.dataset,
            as_of_dates=as_of_dates,
            include_cards=include_cards,
            lineage_block_enabled=args.lineage_block_enabled,
            reporting_lag_days=args.reporting_lag_days,
            lineage_window_days=args.lineage_window_days,
            trend_threshold=args.trend_threshold,
            window_size_days=args.window_size_days,
        )
        # Re-key into the v2 cache contract.
        for (sid, as_of_iso), card in composite.items():
            descriptions[f"{sid}|{as_of_iso}"] = card
    else:
        legacy = build_sample_descriptions(
            df, args.dataset, include_cards=include_cards,
        )
        for sid, card in legacy.items():
            descriptions[sid] = card

    log.info(
        "Generated %d entries in cache_format=%s", len(descriptions), cache_format,
    )

    if descriptions:
        first_key = next(iter(descriptions))
        log.info(
            "Sample summary [%s]: %s",
            first_key, descriptions[first_key].model_summary[:400],
        )

    # ------------------------------------------------------------------
    # 3. Save descriptions JSON (for inspection/debugging)
    # ------------------------------------------------------------------
    if args.descriptions_output is None:
        log.info("Skipping descriptions JSON (--no-descriptions-output).")
    else:
        log.info("Saving descriptions to %s", args.descriptions_output)
        args.descriptions_output.parent.mkdir(parents=True, exist_ok=True)
        desc_json = {
            cache_key: card.to_dict() for cache_key, card in descriptions.items()
        }
        with open(args.descriptions_output, "w") as f:
            json.dump(desc_json, f, indent=2, default=str)

    # ------------------------------------------------------------------
    # 4. Deduplicate by canonical key
    # ------------------------------------------------------------------
    key_to_samples: dict[str, list[str]] = defaultdict(list)
    key_to_summary: dict[str, str] = {}
    for cache_key, card in descriptions.items():
        key_to_samples[card.canonical_key].append(cache_key)
        if card.canonical_key not in key_to_summary:
            key_to_summary[card.canonical_key] = card.model_summary

    n_unique = len(key_to_summary)
    log.info(
        "Deduplication: %d entries -> %d unique canonical keys (%.1f%% reduction)",
        len(descriptions),
        n_unique,
        (1 - n_unique / max(len(descriptions), 1)) * 100,
    )

    if args.dry_run:
        log.info("Dry run — skipping LLM encoding.")
        _write_report(
            args, len(df), len(descriptions), n_unique,
            0, time.time() - start_time, dry_run=True,
        )
        return

    # ------------------------------------------------------------------
    # 5. Encode unique descriptions with LLM
    # ------------------------------------------------------------------
    key_to_summary["__UNKNOWN__"] = "Unknown or unspecified viral strain sample"

    log.info(
        "Encoding %d unique descriptions with %s (batch_size=%d, max_length=%d)",
        len(key_to_summary), args.model_name, args.batch_size, args.max_length,
    )
    key_embeddings: Dict[str, torch.Tensor] = encode_descriptions(
        key_to_summary,
        model_name=args.model_name,
        batch_size=args.batch_size,
        device=args.device,
        max_length=args.max_length,
    )

    # ------------------------------------------------------------------
    # 6. Map back to per-cache-key embeddings
    # ------------------------------------------------------------------
    sample_embeddings: Dict[str, torch.Tensor] = {}
    for canonical, cache_keys in key_to_samples.items():
        emb = key_embeddings[canonical]
        for ck in cache_keys:
            sample_embeddings[ck] = emb
    sample_embeddings["__UNKNOWN__"] = key_embeddings["__UNKNOWN__"]

    # ------------------------------------------------------------------
    # 7. Save
    # ------------------------------------------------------------------
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if cache_format == "v2_as_of":
        torch.save(
            {
                "format": "v2_as_of",
                "embeddings": sample_embeddings,
                "dim": int(next(iter(key_embeddings.values())).shape[-1]),
                "reporting_lag_days": args.reporting_lag_days,
                "lineage_window_days": args.lineage_window_days,
                "lineage_block_enabled": args.lineage_block_enabled,
            },
            args.output,
        )
    else:
        torch.save(sample_embeddings, args.output)
    log.info(
        "Saved %d entries (format=%s) to %s",
        len(sample_embeddings), cache_format, args.output,
    )

    elapsed = time.time() - start_time
    _write_report(
        args, len(df), len(descriptions), n_unique,
        len(sample_embeddings), elapsed,
    )
    log.info("Done in %.1f seconds", elapsed)


def _write_report(
    args: argparse.Namespace,
    n_rows: int,
    n_descriptions: int,
    n_unique_keys: int,
    n_embeddings: int,
    elapsed: float,
    dry_run: bool = False,
) -> None:
    """Write processing report."""
    report = {
        "dataset": args.dataset,
        "input_csv": str(args.input_csv),
        "output_path": str(args.output),
        "descriptions_output": (
            str(args.descriptions_output) if args.descriptions_output else None
        ),
        "model_name": args.model_name,
        "device": args.device,
        "total_rows": n_rows,
        "descriptions_generated": n_descriptions,
        "filtered_out": n_rows - n_descriptions,
        "unique_canonical_keys": n_unique_keys,
        "dedup_reduction_pct": round(
            (1 - n_unique_keys / max(n_descriptions, 1)) * 100, 1,
        ),
        "total_embeddings": n_embeddings,
        "dry_run": dry_run,
        "elapsed_seconds": round(elapsed, 1),
    }
    report_path = args.log_dir / f"sample_embeddings_{args.dataset}_report.json"
    write_processing_report(report, report_path)
    logger.info("Report written to %s", report_path)


if __name__ == "__main__":
    main()
