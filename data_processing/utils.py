"""Shared utilities for data preprocessing."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path


def setup_logging(log_dir: Path) -> logging.Logger:
    """Configure a logger that writes to both console and a file in *log_dir*."""
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("data_processing")
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File
    fh = logging.FileHandler(log_dir / "processing.log", mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def ensure_output_dirs(base: Path) -> None:
    """Create output directory tree under *base*."""
    (base / "aiv").mkdir(parents=True, exist_ok=True)
    (base / "aiv" / "abundance").mkdir(parents=True, exist_ok=True)
    (base / "covid").mkdir(parents=True, exist_ok=True)
    (base / "japan").mkdir(parents=True, exist_ok=True)
    (base / "logs").mkdir(parents=True, exist_ok=True)


def copy_file_unchanged(src: Path, dst: Path, logger: logging.Logger) -> None:
    """Copy *src* to *dst* without modification."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Copying %s -> %s (unchanged)", src.name, dst)
    shutil.copy2(src, dst)


def log_processing_stats(
    logger: logging.Logger, filename: str, stats: dict
) -> None:
    """Log a standardized processing summary for one file."""
    logger.info("--- %s ---", filename)
    for key, value in stats.items():
        logger.info("  %s: %s", key, value)


def write_processing_report(report: dict, path: Path) -> None:
    """Write *report* as formatted JSON to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)
