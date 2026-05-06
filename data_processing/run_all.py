"""Entry point for the data preprocessing pipeline.

Usage::

    python -m data_processing.run_all [--aiv] [--covid] [--fuzzy-threshold 85]
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np

from data_processing.process_aiv import process_aiv
from data_processing.process_covid import process_covid
from data_processing.process_japan import process_japan
from data_processing.utils import ensure_output_dirs, setup_logging, write_processing_report

# Deterministic seed
SEED = 42


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess raw epidemic data for HierEpiGNN."
    )
    parser.add_argument("--aiv", action="store_true", help="Process AIV data only")
    parser.add_argument("--covid", action="store_true", help="Process COVID data only")
    parser.add_argument("--japan", action="store_true", help="Process Japan data only")
    parser.add_argument(
        "--fuzzy-threshold",
        type=int,
        default=85,
        help="Minimum rapidfuzz score for state matching (default: 85)",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data"),
        help="Base directory containing raw data (default: data/)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Base output directory (default: data/processed/)",
    )
    args = parser.parse_args()

    # Default: process all
    if not args.aiv and not args.covid and not args.japan:
        args.aiv = True
        args.covid = True
        args.japan = True

    # Reproducibility
    random.seed(SEED)
    np.random.seed(SEED)

    ensure_output_dirs(args.output_dir)
    log_dir = args.output_dir / "logs"
    logger = setup_logging(log_dir)

    logger.info("Starting data preprocessing (seed=%d)", SEED)
    start = time.time()

    reports: dict = {}

    if args.aiv:
        reports["aiv"] = process_aiv(
            raw_dir=args.raw_dir / "aiv",
            output_dir=args.output_dir / "aiv",
            log_dir=log_dir,
            fuzzy_threshold=args.fuzzy_threshold,
        )

    if args.covid:
        reports["covid"] = process_covid(
            raw_dir=args.raw_dir / "covid",
            output_dir=args.output_dir / "covid",
            log_dir=log_dir,
            fuzzy_threshold=args.fuzzy_threshold,
        )

    if args.japan:
        reports["japan"] = process_japan(
            raw_dir=args.raw_dir / "japan",
            output_dir=args.output_dir / "japan",
            log_dir=log_dir,
            fuzzy_threshold=args.fuzzy_threshold,
        )

    elapsed = time.time() - start
    logger.info("Preprocessing complete in %.1f seconds", elapsed)

    # Write combined report
    report_path = log_dir / "processing_report.json"
    write_processing_report(reports, report_path)
    logger.info("Report written to %s", report_path)

    # Print summary
    _print_summary(reports)


def _print_summary(reports: dict) -> None:
    """Print a human-readable summary table."""
    print("\n" + "=" * 60)
    print("PREPROCESSING SUMMARY")
    print("=" * 60)

    for dataset, files in reports.items():
        print(f"\n  [{dataset.upper()}]")
        if isinstance(files, str):
            print(f"    {files}")
            continue
        for filename, stats in files.items():
            if isinstance(stats, str):
                print(f"    {filename}: {stats}")
            elif isinstance(stats, dict):
                total = stats.get("total_rows", stats.get("total_keys_original", "?"))
                kept = stats.get("rows_kept", stats.get("total_keys_kept", "?"))
                print(f"    {filename}: {kept}/{total} rows kept")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
