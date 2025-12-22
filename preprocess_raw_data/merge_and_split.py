#!/usr/bin/env python3
"""
Merge intermediate Parquet files, filter singleton building IDs, create train/val/test splits,
and write final Parquet files.

This script reads embedding Parquet files produced after tar preprocessing, removes singleton buildings
(which cannot form triplets), creates deterministic splits by building, and writes final
Parquet files for training. Resulting splits include embedding vectors and exclude raw image paths.

Uses a two-pass streaming approach to avoid OOM errors with large datasets:
- Pass 1: Streams only identifier/coord columns to compute building counts and split mapping
- Pass 2: Streams full data in configurable batches, filters singletons, assigns splits, writes incrementally

To test with a small sample, create a temporary directory with a subset of embedding files and
run with a small batch_size (e.g., 1000) to verify split ratios and output structure.
"""

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
)

from config import MergeAndSplitConfig, load_config_from_file
from pandas.util import hash_pandas_object


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


def _enrich_batch_identifiers(
    batch_df: pd.DataFrame, logger: logging.Logger
) -> pd.DataFrame:
    """
    Enrich a batch DataFrame with building_id, streetview_image_id, and target_coord_hash.
    
    This is a streaming-friendly version that works on batches.
    """
    # Ensure dataset_id exists (simplified version for batches)
    if "dataset_id" not in batch_df.columns:
        # Try to infer from other columns
        if "building_id" in batch_df.columns:
            prefixes = batch_df["building_id"].astype(str).str.split("_", n=1).str[0]
            inferred = pd.to_numeric(prefixes, errors="coerce")
        elif "tar_file" in batch_df.columns:
            stems = batch_df["tar_file"].astype(str).str.removesuffix(".tar")
            inferred = pd.to_numeric(stems, errors="coerce")
        else:
            raise ValueError("Cannot infer dataset_id from batch")
        if inferred.isna().any():
            sample = inferred[inferred.isna()].head(3).to_list()
            raise ValueError(
                f"Failed to infer dataset_id for batch rows (examples: {sample}). "
                "Ensure dataset_id is present or inferable."
            )
        batch_df["dataset_id"] = inferred.astype("Int64")
    else:
        # Validate existing dataset_id has no NA after coercion
        coerced = pd.to_numeric(batch_df["dataset_id"], errors="coerce")
        if coerced.isna().any():
            sample = coerced[coerced.isna()].head(3).to_list()
            raise ValueError(
                f"dataset_id contains non-numeric/NA values (examples: {sample})."
            )
        batch_df["dataset_id"] = coerced.astype("Int64")
    
    # Ensure building_id and streetview_image_id
    required_columns = {"dataset_id", "target_id", "patch_id"}
    missing = required_columns - set(batch_df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns for composite identifiers: {', '.join(sorted(missing))}"
        )
    
    enriched_df = batch_df.copy()
    
    dataset_str = enriched_df["dataset_id"].astype("Int64").astype(str).str.zfill(4)
    target_str = enriched_df["target_id"].astype(int).astype(str)
    patch_str = enriched_df["patch_id"].astype(int).astype(str)
    
    if "building_id" not in enriched_df.columns:
        enriched_df["building_id"] = dataset_str + "_" + target_str
    if "streetview_image_id" not in enriched_df.columns:
        enriched_df["streetview_image_id"] = dataset_str + "_" + patch_str
    
    # Add coordinate hash
    if "target_lat" in enriched_df.columns and "target_lon" in enriched_df.columns:
        hash_source = enriched_df[["target_lat", "target_lon"]]
        hash_values = hash_pandas_object(hash_source, index=False, categorize=False)
        hash_strings = hash_values.map(lambda value: format(int(value), "016x"))
        enriched_df["target_coord_hash"] = hash_strings.astype("string")
    
    return enriched_df


def ensure_building_identifiers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure building-aware identifier columns exist (`building_id`, `streetview_image_id`).

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

    if "building_id" not in enriched_df.columns:
        enriched_df["building_id"] = dataset_str + "_" + target_str
    if "streetview_image_id" not in enriched_df.columns:
        enriched_df["streetview_image_id"] = dataset_str + "_" + patch_str

    return enriched_df


def add_coordinate_hash(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Compute a short hash for each (target_lat, target_lon) pair.

    This enables downstream deduplication by exact coordinate rather than building_id.
    """
    required_columns = {"target_lat", "target_lon"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required coordinate columns for hashing: {', '.join(sorted(missing))}"
        )

    hash_source = df[["target_lat", "target_lon"]]
    hash_values = hash_pandas_object(hash_source, index=False, categorize=False)
    hash_strings = hash_values.map(lambda value: format(int(value), "016x"))
    df = df.copy()
    df["target_coord_hash"] = hash_strings.astype("string")

    unique_coords = df["target_coord_hash"].nunique(dropna=False)
    logger.info(
        "Computed coordinate hashes for %s rows (%s unique lat/lon pairs)",
        f"{len(df):,}",
        f"{unique_coords:,}",
    )
    return df


def ensure_dataset_id(
    df: pd.DataFrame, logger: logging.Logger, *, source: Path | None = None
) -> pd.DataFrame:
    """
    Guarantee that a `dataset_id` column exists with integer dtype.

    Attempts to back-fill missing values using other identifier columns typically
    present in intermediate or embedding parquet outputs.
    """
    working_df = df.copy()

    def _coerce_numeric(series: pd.Series, label: str) -> pd.Series:
        numeric = pd.to_numeric(series, errors="coerce")
        return pd.Series(
            pd.array(numeric, dtype="Int64"), index=working_df.index, name=label
        )

    # Start with existing dataset_id values if present
    if "dataset_id" in working_df.columns:
        dataset_series = _coerce_numeric(working_df["dataset_id"], "dataset_id")
    else:
        dataset_series = pd.Series(
            pd.array([pd.NA] * len(working_df), dtype="Int64"), index=working_df.index
        )

    # Helper to merge inferred values into dataset_series
    def _fill_from(label: str, values: pd.Series) -> None:
        nonlocal dataset_series
        try:
            inferred = _coerce_numeric(values, label)
        except Exception:
            return
        missing = dataset_series.isna()
        if not missing.any():
            return
        fill_mask = missing & inferred.notna()
        if fill_mask.any():
            dataset_series = dataset_series.where(~fill_mask, inferred)
            logger.debug(
                "Filled dataset_id for %d rows using %s",
                int(fill_mask.sum()),
                label,
            )

    if "building_id" in working_df.columns:
        prefixes = working_df["building_id"].astype(str).str.split("_", n=1).str[0]
        _fill_from("building_id", prefixes)

    if dataset_series.isna().any() and "streetview_image_id" in working_df.columns:
        prefixes = (
            working_df["streetview_image_id"].astype(str).str.split("_", n=1).str[0]
        )
        _fill_from("streetview_image_id", prefixes)

    if dataset_series.isna().any() and "image_path" in working_df.columns:
        prefixes = working_df["image_path"].astype(str).str.split("/", n=1).str[0]
        _fill_from("image_path", prefixes)

    if dataset_series.isna().any() and "tar_file" in working_df.columns:
        stems = working_df["tar_file"].astype(str).str.removesuffix(".tar")
        _fill_from("tar_file", stems)

    if dataset_series.isna().any():
        sample_missing = min(5, int(dataset_series.isna().sum()))
        context_columns = [
            col
            for col in ["building_id", "streetview_image_id", "image_path", "tar_file"]
            if col in working_df.columns
        ]
        context = (
            working_df.loc[dataset_series.isna(), context_columns].head(sample_missing)
            if context_columns
            else pd.DataFrame(index=working_df.index)
        )
        raise ValueError(
            "Unable to infer dataset_id for all rows. "
            "Expected at least one of building_id, streetview_image_id, image_path, or tar_file "
            "to contain dataset prefixes. "
            f"Problematic sample:\n{context.to_string(index=False)}"
        )

    working_df["dataset_id"] = dataset_series.astype("int64")
    if source:
        logger.debug("Ensured dataset_id for %s", source)
    return working_df


def filter_singleton_buildings(
    df: pd.DataFrame, logger: logging.Logger
) -> pd.DataFrame:
    """Remove entries where a building_id appears only once."""
    if "building_id" not in df.columns:
        raise ValueError("Expected 'building_id' column to filter singleton buildings.")

    total_entries_before = len(df)
    unique_targets_before = df["building_id"].nunique()

    logger.info(
        "Before filtering: %s entries from %s unique building_ids",
        f"{total_entries_before:,}",
        f"{unique_targets_before:,}",
    )

    # Count occurrences per building_id
    target_counts = df.groupby("building_id").size()
    singleton_targets = target_counts[target_counts == 1].index

    logger.info(f"Found {len(singleton_targets):,} singleton building_ids")
    logger.info(f"Removing {len(singleton_targets):,} entries")

    # Filter out singletons
    filtered_df = df[~df["building_id"].isin(singleton_targets)].copy()

    total_entries_after = len(filtered_df)
    unique_targets_after = filtered_df["building_id"].nunique()

    logger.info(
        "After filtering: %s entries from %s unique building_ids",
        f"{total_entries_after:,}",
        f"{unique_targets_after:,}",
    )
    logger.info(
        "Removed %s entries (%.1f%%)",
        f"{total_entries_before - total_entries_after:,}",
        (
            ((total_entries_before - total_entries_after) / total_entries_before * 100)
            if total_entries_before
            else 0.0
        ),
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
    """
    Assign split labels to entries based on target_coord_hash (spatial split).

    Using target_coord_hash ensures that all buildings at the same location (e.g. same building,
    different angles/ids) end up in the same split, preventing data leakage.
    """
    split_key = "target_coord_hash"
    if split_key not in df.columns:
        logger.warning(
            f"{split_key} not found, falling back to building_id. "
            "WARNING: This may cause data leakage if multiple building_ids share coordinates."
        )
        split_key = "building_id"

    if split_key not in df.columns:
        raise ValueError(f"Expected '{split_key}' column to create splits.")

    unique_keys = df[split_key].unique()
    n_keys = len(unique_keys)

    logger.info(f"Splitting {n_keys:,} unique keys ({split_key})")
    logger.info(
        f"Ratios: train={train_ratio:.1%}, val={val_ratio:.1%}, test={test_ratio:.1%}"
    )

    # Shuffle keys deterministically
    rng = np.random.default_rng(seed)
    shuffled_keys = rng.permutation(unique_keys)

    # Calculate split boundaries
    n_train = int(n_keys * train_ratio)
    n_val = int(n_keys * val_ratio)
    # n_test = n_keys - n_train - n_val (to handle rounding)

    train_keys = set(shuffled_keys[:n_train])
    val_keys = set(shuffled_keys[n_train : n_train + n_val])
    # test_keys = set(shuffled_keys[n_train + n_val :])

    # Assign split labels
    df["split"] = df[split_key].map(
        lambda key: (
            "train" if key in train_keys else ("val" if key in val_keys else "test")
        )
    )

    # Log split statistics
    split_counts = df["split"].value_counts()
    logger.info("Split distribution (entries):")
    for split, count in split_counts.items():
        logger.info(f"  {split}: {count:,} entries ({count/len(df):.1%})")

    # Convert split to categorical for memory efficiency
    df["split"] = df["split"].astype("category")

    return df


def pass1_compute_metadata(
    dataset: ds.Dataset,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    logger: logging.Logger,
    progress: Progress,
) -> tuple[set[str], dict[str, str], int]:
    """
    Pass 1: Stream identifier/coord columns to compute building counts and split mapping.
    
    Returns:
        - singleton_building_ids: Set of building_ids to filter out
        - coord_hash_to_split: Mapping from target_coord_hash to split name
        - total_building_ids: Total number of unique building_ids (before filtering)
    """
    logger.info("Pass 1: Computing building counts and split assignments...")
    
    # Columns needed for pass 1 (excluding embedding to save memory)
    identifier_columns = [
        "dataset_id",
        "target_id",
        "patch_id",
        "target_lat",
        "target_lon",
    ]
    
    # Check which columns exist in the dataset
    schema = dataset.schema
    available_columns = set(schema.names)
    needed_columns = [col for col in identifier_columns if col in available_columns]
    required_coords = {"target_lat", "target_lon"}
    missing_coords = required_coords - available_columns
    if missing_coords:
        raise ValueError(
            "Missing required coordinate columns for hashing: "
            f"{', '.join(sorted(missing_coords))}. "
            "Ensure embedding parquet outputs include target_lat and target_lon."
        )
    
    if not needed_columns:
        raise ValueError(
            f"None of the required identifier columns found. "
            f"Available: {sorted(available_columns)}"
        )
    
    # Stream batches and collect building_id counts and coordinate hashes
    building_id_counts: Counter[str] = Counter()
    coord_hashes: set[str] = set()
    total_rows = 0
    
    task = progress.add_task("Pass 1: Streaming identifiers...", total=None)
    
    scanner = dataset.scanner(columns=needed_columns)
    for batch in scanner.to_batches():
        batch_df = batch.to_pandas()
        total_rows += len(batch_df)
        
        # Enrich with identifiers
        batch_df = _enrich_batch_identifiers(batch_df, logger)
        
        # Count building_ids
        if "building_id" in batch_df.columns:
            building_id_counts.update(batch_df["building_id"].tolist())
        
        # Collect unique coordinate hashes
        if "target_coord_hash" in batch_df.columns:
            coord_hashes.update(batch_df["target_coord_hash"].unique())
        
        progress.update(task, advance=len(batch_df))
    
    # Identify singleton building_ids
    singleton_building_ids = {
        bid for bid, count in building_id_counts.items() if count == 1
    }
    total_building_ids = len(building_id_counts)
    
    logger.info(
        f"Pass 1 complete: {total_rows:,} rows, {total_building_ids:,} unique building_ids, "
        f"{len(singleton_building_ids):,} singletons, "
        f"{len(coord_hashes):,} unique coordinate hashes"
    )
    
    # Create deterministic split assignment for coordinate hashes
    unique_coords_list = sorted(coord_hashes)
    n_keys = len(unique_coords_list)
    
    logger.info(f"Splitting {n_keys:,} unique coordinate hashes")
    logger.info(
        f"Ratios: train={train_ratio:.1%}, val={val_ratio:.1%}, test={test_ratio:.1%}"
    )
    
    # Shuffle keys deterministically
    rng = np.random.default_rng(seed)
    shuffled_coords = rng.permutation(unique_coords_list)
    
    # Calculate split boundaries
    n_train = int(n_keys * train_ratio)
    n_val = int(n_keys * val_ratio)
    
    train_keys = set(shuffled_coords[:n_train])
    val_keys = set(shuffled_coords[n_train : n_train + n_val])
    # test_keys are the remainder
    
    # Create mapping
    coord_hash_to_split: dict[str, str] = {}
    for coord_hash in train_keys:
        coord_hash_to_split[coord_hash] = "train"
    for coord_hash in val_keys:
        coord_hash_to_split[coord_hash] = "val"
    for coord_hash in shuffled_coords[n_train + n_val :]:
        coord_hash_to_split[coord_hash] = "test"
    
    return singleton_building_ids, coord_hash_to_split, total_building_ids


def pass2_stream_and_write(
    dataset: ds.Dataset,
    singleton_building_ids: set[str],
    coord_hash_to_split: dict[str, str],
    output_dir: Path,
    batch_size: int,
    logger: logging.Logger,
    progress: Progress,
) -> dict[str, int]:
    """
    Pass 2: Stream full data in batches, filter singletons, assign splits, write incrementally.
    
    Returns statistics dictionary.
    """
    logger.info("Pass 2: Streaming full data and writing splits...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize Parquet writers for each split
    split_writers: dict[str, pq.ParquetWriter | None] = {
        "train": None,
        "val": None,
        "test": None,
    }
    
    # Track statistics
    stats = {
        "total_rows_read": 0,
        "rows_after_filtering": 0,
        "train_entries": 0,
        "val_entries": 0,
        "test_entries": 0,
    }
    
    # Infer schema once before starting batch processing to ensure consistency
    split_schemas: dict[str, pa.Schema | None] = {
        "train": None,
        "val": None,
        "test": None,
    }
    
    task = progress.add_task("Pass 2: Processing batches...", total=None)
    
    try:
        # Stream batches
        scanner = dataset.scanner(batch_size=batch_size)
        for batch in scanner.to_batches():
            batch_df = batch.to_pandas()
            rows_in_batch = len(batch_df)
            stats["total_rows_read"] += rows_in_batch
            
            # Enrich with identifiers
            batch_df = _enrich_batch_identifiers(batch_df, logger)
            
            # Filter singleton buildings
            if "building_id" in batch_df.columns:
                before_filter = len(batch_df)
                batch_df = batch_df[~batch_df["building_id"].isin(singleton_building_ids)]
                after_filter = len(batch_df)
                filtered_out = before_filter - after_filter
                stats["rows_after_filtering"] += after_filter
                if filtered_out > 0:
                    logger.debug(
                        "Filtered out %d singleton rows in current batch (before: %d, after: %d)",
                        filtered_out,
                        before_filter,
                        after_filter,
                    )
            else:
                stats["rows_after_filtering"] += len(batch_df)
            
            # Assign splits based on coordinate hash
            if "target_coord_hash" in batch_df.columns:
                batch_df["split"] = batch_df["target_coord_hash"].map(
                    lambda h: coord_hash_to_split.get(h, "test")
                )
            else:
                logger.warning("target_coord_hash not found, assigning all to test")
                batch_df["split"] = "test"
            
            # Drop columns not needed in final files (but keep split for now)
            columns_to_drop = ["tar_file", "image_path"]
            batch_df_clean = batch_df.drop(columns=[c for c in columns_to_drop if c in batch_df.columns])
            
            # Write to appropriate split files
            for split_name in ["train", "val", "test"]:
                split_batch = batch_df_clean[batch_df_clean["split"] == split_name]
                if len(split_batch) == 0:
                    continue
                
                # Drop split column before writing
                split_batch_final = split_batch.drop(columns=["split"], errors="ignore")
                
                # Initialize schema and writer if needed
                if split_schemas[split_name] is None:
                    # Infer schema from first batch for this split
                    split_schemas[split_name] = pa.Schema.from_pandas(split_batch_final)
                
                if split_writers[split_name] is None:
                    output_file = output_dir / f"{split_name}.parquet"
                    split_writers[split_name] = pq.ParquetWriter(
                        output_file,
                        split_schemas[split_name],
                        compression="snappy",
                    )
                
                # Convert to Arrow table and write using the consistent schema
                split_table = pa.Table.from_pandas(split_batch_final, schema=split_schemas[split_name], preserve_index=False)
                split_writers[split_name].write_table(split_table)
                
                stats[f"{split_name}_entries"] += len(split_batch)
            
            # Update progress based on original input rows read
            progress.update(task, advance=rows_in_batch)
        
        # Close all writers
        for split_name, writer in split_writers.items():
            if writer is not None:
                writer.close()
                output_file = output_dir / f"{split_name}.parquet"
                file_size_mb = output_file.stat().st_size / (1024 * 1024)
                logger.info(
                    f"Wrote {split_name}.parquet: {stats[f'{split_name}_entries']:,} entries, "
                    f"{file_size_mb:.1f} MB"
                )
    
    except Exception:
        # Close writers on error
        for writer in split_writers.values():
            if writer is not None:
                writer.close()
        raise
    
    return stats


def merge_and_split(
    intermediates_dir: Path,
    embeddings_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    batch_size: int = 100_000,
    log_file: Path | None = None,
) -> dict[str, int]:
    """
    Merge intermediate Parquet files, filter singletons, create splits, and write final files.
    
    Uses a two-pass streaming approach to avoid loading all data into memory:
    - Pass 1: Stream identifier/coord columns to compute building counts and split mapping
    - Pass 2: Stream full data in batches, filter singletons, assign splits, write incrementally

    Returns statistics dictionary.
    """
    logger = setup_logging(log_file)
    logger.info("=" * 60)
    logger.info("Dataset assembly: merge and split intermediates (streaming mode)")
    logger.info("=" * 60)
    logger.info(f"Batch size: {batch_size:,} rows")

    # Discover intermediate files
    logger.info(f"Discovering intermediate files in: {intermediates_dir}")
    intermediate_files = discover_intermediate_files(intermediates_dir)
    logger.info(f"Found {len(intermediate_files)} intermediate Parquet files")

    if not intermediate_files:
        raise ValueError(f"No Parquet files found in {intermediates_dir}")

    logger.info(f"Discovering embedding files in: {embeddings_dir}")
    embedding_files = sorted(embeddings_dir.glob("*_embeddings.parquet"))
    logger.info(f"Found {len(embedding_files)} embedding Parquet files")

    if not embedding_files:
        raise ValueError(f"No embedding Parquet files found in {embeddings_dir}")

    # Warn if counts differ
    intermediate_stems = {path.stem for path in intermediate_files}
    embedding_stems = {
        path.stem.removesuffix("_embeddings") for path in embedding_files
    }
    missing_embeddings = sorted(intermediate_stems - embedding_stems)
    if missing_embeddings:
        logger.warning(
            "Missing embeddings for %d intermediate files (e.g., %s)",
            len(missing_embeddings),
            ", ".join(missing_embeddings[:3]),
        )

    # Create PyArrow dataset from embedding files
    logger.info("Creating PyArrow dataset from embedding files...")
    dataset = ds.dataset(
        embedding_files,
        format="parquet",
        schema=None,  # Auto-detect schema
    )
    
    # Verify embedding column exists
    schema = dataset.schema
    if "embedding" not in schema.names:
        raise ValueError(
            "Embedding column not found in dataset. "
            "Ensure generate_embeddings.py outputs include an 'embedding' column."
        )

    # Pass 1: Compute building counts and split mapping
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        refresh_per_second=1,
    ) as progress:
        (
            singleton_building_ids,
            coord_hash_to_split,
            total_building_ids,
        ) = pass1_compute_metadata(
            dataset,
            train_ratio,
            val_ratio,
            test_ratio,
            seed,
            logger,
            progress,
        )

    # Pass 2: Stream full data and write splits
    logger.info("Pass 2: Streaming full data and writing splits...")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        refresh_per_second=1,
    ) as progress:
        pass2_stats = pass2_stream_and_write(
            dataset,
            singleton_building_ids,
            coord_hash_to_split,
            output_dir,
            batch_size,
            logger,
            progress,
        )

    # Calculate final statistics
    unique_building_ids_after_filter = total_building_ids - len(singleton_building_ids)
    stats = {
        "total_intermediate_files": len(intermediate_files),
        "total_embedding_files": len(embedding_files),
        "total_rows_read": pass2_stats["total_rows_read"],
        "rows_after_filtering": pass2_stats["rows_after_filtering"],
        "unique_building_ids": unique_building_ids_after_filter,
        "train_entries": pass2_stats["train_entries"],
        "val_entries": pass2_stats["val_entries"],
        "test_entries": pass2_stats["test_entries"],
    }

    logger.info("=" * 60)
    logger.info("Dataset assembly complete")
    logger.info("=" * 60)
    logger.info(
        f"Total intermediate files processed: {stats['total_intermediate_files']}"
    )
    logger.info(f"Total embedding files processed: {stats['total_embedding_files']}")
    logger.info(f"Total rows read: {stats['total_rows_read']:,}")
    logger.info(f"Rows after filtering singletons: {stats['rows_after_filtering']:,}")
    logger.info(
        f"Unique building IDs (after filtering singletons): {stats['unique_building_ids']:,}"
    )
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
            embeddings_dir=config.embeddings_dir,
            output_dir=config.merged_dir,
            train_ratio=config.train_ratio,
            val_ratio=config.val_ratio,
            test_ratio=config.test_ratio,
            seed=config.seed if config.seed is not None else 42,
            batch_size=config.batch_size,
            log_file=config.log_file,
        )

        # Print summary
        logger.info("")
        logger.info("Processing Summary:")
        logger.info(f"  Total intermediate files: {stats['total_intermediate_files']}")
        logger.info(f"  Total entries: {stats['total_rows_read']:,}")
        logger.info(f"  Unique building_ids: {stats['unique_building_ids']:,}")
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
    # Example: To test with a small sample, create a test directory with a few embedding files
    # and run: python merge_and_split.py --config config.toml
    # Verify output split ratios match config ratios and check output parquet file sizes.
    sys.exit(main())
