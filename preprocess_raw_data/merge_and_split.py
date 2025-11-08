#!/usr/bin/env python3
"""
Merge intermediate Parquet files, filter singleton TargetIDs, create train/val/test splits,
and write final Parquet files.

This script reads all intermediate Parquet files produced during tar preprocessing, removes singleton TargetIDs
(which cannot form triplets), creates deterministic splits by TargetID, and writes final
Parquet files for training.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from preprocess_raw_data.config import MergeAndSplitConfig, load_config_from_file


def setup_logging(log_file: Path | None = None) -> logging.Logger:
    """Set up logging with Rich handler."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Remove existing handlers
    logger.handlers.clear()

    # Create Rich handler
    if log_file:
        # When writing to file, use file-aware console
        handler = RichHandler(
            console=Console(file=open(log_file, "w", encoding="utf-8")),
            rich_tracebacks=True,
            show_path=False,
        )
    else:
        handler = RichHandler(rich_tracebacks=True, show_path=False)

    handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
    logger.addHandler(handler)

    return logger


def discover_intermediate_files(intermediates_dir: Path) -> list[Path]:
    """Find all .parquet files in intermediates directory."""
    parquet_files = sorted(intermediates_dir.glob("*.parquet"))
    return parquet_files


def read_intermediate_files(
    parquet_files: list[Path],
    logger: logging.Logger,
    progress: Progress,
) -> pd.DataFrame:
    """Read all intermediate Parquet files and concatenate."""
    dfs = []
    task = progress.add_task("Reading intermediate files...", total=len(parquet_files))

    for parquet_file in parquet_files:
        try:
            df = pd.read_parquet(parquet_file, engine="pyarrow")
            dfs.append(df)
            logger.debug(f"Read {len(df)} rows from {parquet_file.name}")
        except Exception as e:
            logger.error(f"Failed to read {parquet_file}: {e}")
            continue
        progress.update(task, advance=1)

    if not dfs:
        raise ValueError("No valid intermediate files found")

    logger.info(f"Read {len(dfs)} intermediate files, concatenating...")
    # Concatenate all DataFrames
    combined_df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Total rows after concatenation: {len(combined_df):,}")
    return combined_df


def ensure_composite_identifiers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure dataset-aware identifier columns exist (dataset_target_id, dataset_patch_id, dataset_target_patch_id).

    Returns a copy of the DataFrame with the necessary columns.
    """
    required_columns = {"dataset_id", "target_id", "patch_id"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            "Missing required columns for composite identifiers. "
            f"Ensure tar preprocessing outputs include {', '.join(sorted(required_columns))}. "
            f"Missing: {', '.join(sorted(missing))}"
        )

    enriched_df = df.copy()

    dataset_str = enriched_df["dataset_id"].astype(int).astype(str).str.zfill(4)
    target_str = enriched_df["target_id"].astype(int).astype(str)
    patch_str = enriched_df["patch_id"].astype(int).astype(str)

    if "dataset_target_id" not in enriched_df.columns:
        enriched_df["dataset_target_id"] = dataset_str + "_" + target_str
    if "dataset_patch_id" not in enriched_df.columns:
        enriched_df["dataset_patch_id"] = dataset_str + "_" + patch_str
    if "dataset_target_patch_id" not in enriched_df.columns:
        enriched_df["dataset_target_patch_id"] = dataset_str + "_" + target_str + "_" + patch_str

    return enriched_df


def filter_singleton_targets(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """Remove entries where a dataset_target_id appears only once."""
    total_entries_before = len(df)
    unique_targets_before = df["dataset_target_id"].nunique()

    logger.info(
        "Before filtering: %s entries from %s unique dataset_target_ids",
        f"{total_entries_before:,}",
        f"{unique_targets_before:,}",
    )

    # Count occurrences per dataset_target_id
    target_counts = df.groupby("dataset_target_id").size()
    singleton_targets = target_counts[target_counts == 1].index

    logger.info(f"Found {len(singleton_targets):,} singleton dataset_target_ids")
    logger.info(f"Removing {len(singleton_targets):,} entries")

    # Filter out singletons
    filtered_df = df[~df["dataset_target_id"].isin(singleton_targets)].copy()

    total_entries_after = len(filtered_df)
    unique_targets_after = filtered_df["dataset_target_id"].nunique()

    logger.info(
        "After filtering: %s entries from %s unique dataset_target_ids",
        f"{total_entries_after:,}",
        f"{unique_targets_after:,}",
    )
    logger.info(
        "Removed %s entries (%.1f%%)",
        f"{total_entries_before - total_entries_after:,}",
        ((total_entries_before - total_entries_after) / total_entries_before * 100) if total_entries_before else 0.0,
    )

    return filtered_df


def create_splits(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Assign split labels to entries based on dataset_target_id."""
    unique_target_keys = df["dataset_target_id"].unique()
    n_targets = len(unique_target_keys)

    logger.info(f"Splitting {n_targets:,} unique dataset_target_ids")
    logger.info(f"Ratios: train={train_ratio:.1%}, val={val_ratio:.1%}, test={test_ratio:.1%}")

    # Shuffle composite identifiers deterministically
    rng = np.random.default_rng(seed)
    shuffled_target_ids = rng.permutation(unique_target_keys)

    # Calculate split boundaries
    n_train = int(n_targets * train_ratio)
    n_val = int(n_targets * val_ratio)
    # n_test = n_targets - n_train - n_val (to handle rounding)

    train_ids = set(shuffled_target_ids[:n_train])
    val_ids = set(shuffled_target_ids[n_train : n_train + n_val])
    test_ids = set(shuffled_target_ids[n_train + n_val :])

    # Assign split labels
    df["split"] = df["dataset_target_id"].map(
        lambda tid: "train" if tid in train_ids else ("val" if tid in val_ids else "test")
    )

    # Log split statistics
    split_counts = df["split"].value_counts()
    logger.info("Split distribution:")
    for split, count in split_counts.items():
        logger.info(f"  {split}: {count:,} entries ({count/len(df):.1%})")

    # Convert split to categorical for memory efficiency
    df["split"] = df["split"].astype("category")

    return df


def write_final_files(
    df: pd.DataFrame,
    output_dir: Path,
    logger: logging.Logger,
    progress: Progress,
) -> None:
    """Write final Parquet files, one per split."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        split_df = df[df["split"] == split].copy()

        if len(split_df) == 0:
            logger.warning(f"No entries for {split} split, skipping")
            continue

        logger.info(f"Writing {split} split: {len(split_df):,} entries")

        # Drop split and tar_file columns (not needed in final files)
        split_df = split_df.drop(columns=["split", "tar_file"])

        # Write final file
        final_file = output_dir / f"{split}.parquet"
        task = progress.add_task(f"Writing {split}...", total=1)

        split_df.to_parquet(
            final_file,
            engine="pyarrow",
            compression="snappy",
            index=False,
        )

        progress.update(task, advance=1)

        file_size_mb = final_file.stat().st_size / (1024 * 1024)
        logger.info(f"Wrote {final_file.name}: {len(split_df):,} entries, {file_size_mb:.1f} MB")


def merge_and_split(
    intermediates_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    log_file: Path | None = None,
) -> dict[str, int]:
    """
    Merge intermediate Parquet files, filter singletons, create splits, and write final files.

    Returns statistics dictionary.
    """
    logger = setup_logging(log_file)
    logger.info("=" * 60)
    logger.info("Dataset assembly: merge and split intermediates")
    logger.info("=" * 60)

    # Discover intermediate files
    logger.info(f"Discovering intermediate files in: {intermediates_dir}")
    parquet_files = discover_intermediate_files(intermediates_dir)
    logger.info(f"Found {len(parquet_files)} intermediate Parquet files")

    if not parquet_files:
        raise ValueError(f"No Parquet files found in {intermediates_dir}")

    # Read intermediate files
    logger.info("Reading intermediate files...")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        refresh_per_second=1,
    ) as progress:
        df = read_intermediate_files(parquet_files, logger, progress)

    # Ensure composite identifier columns are present
    df = ensure_composite_identifiers(df)

    # Filter singleton targets
    logger.info("Filtering singleton dataset_target_ids...")
    df = filter_singleton_targets(df, logger)

    # Create splits
    logger.info("Creating train/val/test splits...")
    df = create_splits(df, train_ratio, val_ratio, test_ratio, seed, logger)

    # Write final files
    logger.info("Writing final Parquet files...")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        refresh_per_second=1,
    ) as progress:
        write_final_files(df, output_dir, logger, progress)

    # Calculate statistics
    stats = {
        "total_intermediate_files": len(parquet_files),
        "total_rows_read": len(df),
        "unique_dataset_target_ids": df["dataset_target_id"].nunique(),
        "train_entries": len(df[df["split"] == "train"]),
        "val_entries": len(df[df["split"] == "val"]),
        "test_entries": len(df[df["split"] == "test"]),
    }

    logger.info("=" * 60)
    logger.info("Dataset assembly complete")
    logger.info("=" * 60)
    logger.info(f"Total intermediate files processed: {stats['total_intermediate_files']}")
    logger.info(f"Total entries in final dataset: {stats['total_rows_read']:,}")
    logger.info(f"Unique dataset_target_ids: {stats['unique_dataset_target_ids']:,}")
    logger.info(f"Train entries: {stats['train_entries']:,}")
    logger.info(f"Val entries: {stats['val_entries']:,}")
    logger.info(f"Test entries: {stats['test_entries']:,}")

    return stats


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Merge intermediate Parquet files, filter singletons, create splits, and write final files"
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to TOML config file. If not provided, config is loaded from environment variables.",
    )

    args = parser.parse_args()

    # Set up logger early (before config loading to catch config errors)
    logger = setup_logging(None)

    # Load configuration
    try:
        if args.config:
            config = load_config_from_file(args.config, "merge_and_split")
        else:
            # Load from environment variables
            config = MergeAndSplitConfig()
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return 1

    # Re-setup logger with log file if specified
    if config.log_file:
        logger = setup_logging(config.log_file)

    # Validate inputs
    if not config.intermediates_dir.exists():
        logger.error(f"Intermediates directory not found: {config.intermediates_dir}")
        return 1

    if not config.intermediates_dir.is_dir():
        logger.error(f"Not a directory: {config.intermediates_dir}")
        return 1

    try:
        # Run merge and split
        stats = merge_and_split(
            intermediates_dir=config.intermediates_dir,
            output_dir=config.output_dir,
            train_ratio=config.train_ratio,
            val_ratio=config.val_ratio,
            test_ratio=config.test_ratio,
            seed=config.seed,
            log_file=config.log_file,
        )

        # Print summary
        logger.info("")
        logger.info("Processing Summary:")
        logger.info(f"  Total intermediate files: {stats['total_intermediate_files']}")
        logger.info(f"  Total entries: {stats['total_rows_read']:,}")
        logger.info(f"  Unique dataset_target_ids: {stats['unique_dataset_target_ids']:,}")
        logger.info(f"  Train entries: {stats['train_entries']:,}")
        logger.info(f"  Val entries: {stats['val_entries']:,}")
        logger.info(f"  Test entries: {stats['test_entries']:,}")

        return 0
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        if config.log_file:
            logger.info(f"See log file for details: {config.log_file}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

