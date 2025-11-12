#!/usr/bin/env python3
"""
Repair legacy intermediate parquet files so downstream merge steps can rely on
standardised identifier columns.

Actions performed for each parquet file:
  * Rename `dataset_target_id` -> `building_id`
  * Rename `dataset_patch_id` -> `streetview_image_id`
  * Drop `dataset_target_patch_id`
  * Back-fill `dataset_id` from `building_id` when missing

The script rewrites files in place by default; use `--dry-run` to preview the
changes without writing.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable

import pandas as pd
from rich.console import Console
from rich.logging import RichHandler


def setup_logging(verbose: bool) -> logging.Logger:
    """Configure Rich logging."""
    logger = logging.getLogger("fix_intermediate_schema")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()
    handler = RichHandler(
        console=Console(stderr=True),
        rich_tracebacks=True,
        show_path=False,
        enable_link_path=False,
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    return logger


def discover_parquet_files(intermediates_dir: Path) -> list[Path]:
    """Return sorted list of parquet files under intermediates_dir."""
    files = sorted(intermediates_dir.glob("*.parquet"))
    return files


def infer_dataset_id_from_building(series: pd.Series) -> pd.Series:
    """
    Extract dataset_id from `building_id` values of the form '0003_1338'.

    Raises ValueError if parsing fails for any row.
    """
    dataset_part = series.astype(str).str.split("_", n=1).str[0]
    dataset_numeric = pd.to_numeric(dataset_part, errors="raise")
    return dataset_numeric.astype("int64")


def fix_dataframe(
    df: pd.DataFrame, logger: logging.Logger
) -> tuple[pd.DataFrame, list[str]]:
    """
    Apply column fixes to the dataframe.

    Returns the possibly-modified dataframe and a list of human-readable change notes.
    """
    notes: list[str] = []
    working_df = df.copy()

    if (
        "dataset_target_id" in working_df.columns
        and "building_id" not in working_df.columns
    ):
        working_df = working_df.rename(columns={"dataset_target_id": "building_id"})
        notes.append("renamed dataset_target_id -> building_id")

    if (
        "dataset_patch_id" in working_df.columns
        and "streetview_image_id" not in working_df.columns
    ):
        working_df = working_df.rename(
            columns={"dataset_patch_id": "streetview_image_id"}
        )
        notes.append("renamed dataset_patch_id -> streetview_image_id")

    if "dataset_target_patch_id" in working_df.columns:
        working_df = working_df.drop(columns=["dataset_target_patch_id"])
        notes.append("dropped dataset_target_patch_id")

    if "dataset_id" not in working_df.columns:
        if "building_id" not in working_df.columns:
            raise ValueError(
                "Cannot infer dataset_id: building_id column missing after renames."
            )
        working_df["dataset_id"] = infer_dataset_id_from_building(
            working_df["building_id"]
        )
        notes.append("inferred dataset_id from building_id")
    else:
        # Normalise dtype to int64 for consistency
        working_df["dataset_id"] = pd.to_numeric(
            working_df["dataset_id"], errors="raise"
        ).astype("int64")

    # Ensure identifier columns are strings
    for col in ["building_id", "streetview_image_id"]:
        if col in working_df.columns:
            working_df[col] = working_df[col].astype(str)

    return working_df, notes


def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Keep identifier columns grouped toward the front for readability."""
    preferred_order = [
        "dataset_id",
        "building_id",
        "streetview_image_id",
        "target_id",
        "patch_id",
        "street_view_id",
        "target_lat",
        "target_lon",
        "image_path",
        "tar_file",
    ]
    existing_preferred = [col for col in preferred_order if col in df.columns]
    remaining = [col for col in df.columns if col not in existing_preferred]
    return df[existing_preferred + remaining]


def process_file(path: Path, dry_run: bool, logger: logging.Logger) -> bool:
    """Fix a single parquet file. Returns True when modifications were written."""
    logger.debug("Reading %s", path)
    df = pd.read_parquet(path)

    try:
        fixed_df, notes = fix_dataframe(df, logger)
    except Exception as exc:  # pragma: no cover - surface to caller
        logger.error("Failed to fix %s: %s", path.name, exc, exc_info=True)
        return False

    if not notes:
        logger.info("No changes needed for %s", path.name)
        return False

    fixed_df = reorder_columns(fixed_df)

    logger.info("%s: %s", path.name, "; ".join(notes))
    if dry_run:
        logger.debug("Dry run enabled; skipping write for %s", path.name)
        return False

    fixed_df.to_parquet(path, engine="pyarrow", compression="snappy", index=False)
    logger.debug("Wrote updated parquet to %s", path)
    return True


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repair legacy intermediate parquet files so merge step can run.",
    )
    parser.add_argument(
        "--intermediates-dir",
        type=Path,
        default=Path("data/intermediates"),
        help="Directory containing intermediate parquet files (default: data/intermediates)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan files and report planned changes without writing updates.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    logger = setup_logging(args.verbose)

    intermediates_dir: Path = args.intermediates_dir
    if not intermediates_dir.exists() or not intermediates_dir.is_dir():
        logger.error("Intermediates directory not found: %s", intermediates_dir)
        return 1

    files = discover_parquet_files(intermediates_dir)
    if not files:
        logger.warning("No parquet files found in %s", intermediates_dir)
        return 0

    logger.info(
        "Processing %d parquet files in %s (dry_run=%s)",
        len(files),
        intermediates_dir,
        args.dry_run,
    )

    modified_count = 0
    for path in files:
        if process_file(path, dry_run=args.dry_run, logger=logger):
            modified_count += 1

    logger.info("Completed; %d/%d files modified.", modified_count, len(files))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
