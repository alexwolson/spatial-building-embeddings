#!/usr/bin/env python3
"""
Compute per-target difficulty metadata using sklearn's exact nearest neighbours.

The script reads a Phase 2-style parquet, computes top-K cross-building negatives
per target, assigns global difficulty bands, and writes results to parquet
artifacts in `output_dir`.
"""
import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rich.console import Console
from rich.logging import RichHandler
from sklearn.neighbors import NearestNeighbors

from difficulty_metadata.config import DifficultyConfig, load_config_from_file


def setup_logging(log_file: Path | None) -> logging.Logger:
    """Configure logging using Rich for consistency with other scripts."""
    logger = logging.getLogger("difficulty_metadata")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handler = RichHandler(
            console=Console(file=log_file.open("w", encoding="utf-8")),
            show_path=False,
            rich_tracebacks=True,
        )
    else:
        handler = RichHandler(show_path=False, rich_tracebacks=True)

    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def ensure_composite_identifiers(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure dataset-scoped identifier columns are present."""
    required_columns = {"dataset_id", "target_id"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            "Missing required columns for composite identifiers. "
            f"Ensure upstream preprocessing emits dataset_id/target_id. Missing: {', '.join(sorted(missing))}"
        )

    enriched_df = df.copy()
    dataset_str = enriched_df["dataset_id"].astype(int).astype(str).str.zfill(4)
    target_str = enriched_df["target_id"].astype(int).astype(str)

    if "dataset_target_id" not in enriched_df.columns:
        enriched_df["dataset_target_id"] = dataset_str + "_" + target_str

    if "patch_id" in enriched_df.columns:
        patch_str = enriched_df["patch_id"].astype(int).astype(str)
        if "dataset_patch_id" not in enriched_df.columns:
            enriched_df["dataset_patch_id"] = dataset_str + "_" + patch_str
        if "dataset_target_patch_id" not in enriched_df.columns:
            enriched_df["dataset_target_patch_id"] = dataset_str + "_" + target_str + "_" + patch_str

    return enriched_df


def compute_neighbors(
    source_df: pd.DataFrame,
    config: DifficultyConfig,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    """Compute neighbor cache and return the DataFrame plus metadata."""
    processed_df = source_df.dropna(subset=["target_lat", "target_lon"]).copy()
    processed_df = ensure_composite_identifiers(processed_df)
    if processed_df.empty:
        raise ValueError("No rows remain after dropping missing coordinates.")

    before_dedupe = len(processed_df)
    processed_df = (
        processed_df.drop_duplicates(subset=["dataset_target_id"], keep="first")
        .reset_index(drop=True)
    )
    if len(processed_df) < before_dedupe:
        logger.info(
            "Dropped %s duplicate dataset_target_id rows",
            before_dedupe - len(processed_df),
        )

    processed_df["target_id"] = processed_df["target_id"].astype(str)
    processed_df["dataset_target_id"] = processed_df["dataset_target_id"].astype(str)
    processed_df["building_id"] = processed_df["building_id"].astype(str)
    processed_df["anchor_idx"] = np.arange(len(processed_df), dtype=np.int32)

    num_targets = len(processed_df)
    if num_targets < 2:
        raise ValueError("At least two targets are required to compute negatives.")

    coords = processed_df[["target_lat", "target_lon"]].to_numpy(dtype=np.float32)
    building_codes, unique_buildings = pd.factorize(processed_df["building_id"], sort=False)
    building_counts = np.bincount(building_codes, minlength=len(unique_buildings))
    max_same_building = int(building_counts.max()) if building_counts.size else 0

    logger.info("Loaded %s targets across %s buildings", num_targets, len(unique_buildings))

    overshoot = max_same_building + 10
    n_neighbors_search = min(num_targets, config.k_neighbors + overshoot)
    if n_neighbors_search < config.k_neighbors + 1:
        n_neighbors_search = min(num_targets, config.k_neighbors + 1)

    logger.info(
        "Computing neighbors with n_neighbors=%s (k=%s, overshoot=%s)",
        n_neighbors_search,
        config.k_neighbors,
        overshoot,
    )

    nn = NearestNeighbors(
        metric="euclidean",
        algorithm="auto",
        n_jobs=config.n_jobs,
    )
    nn.fit(coords)
    distances, indices = nn.kneighbors(coords, n_neighbors=n_neighbors_search, return_distance=True)

    target_keys = processed_df["dataset_target_id"].tolist()
    target_ids = processed_df["target_id"].tolist()
    building_ids = processed_df["building_id"].tolist()
    anchor_neighbor_indices: list[np.ndarray] = []
    anchor_distances: list[np.ndarray] = []
    negative_counts: list[int] = []

    for anchor_idx in range(num_targets):
        neighbor_idx_row = indices[anchor_idx]
        neighbor_dist_row = distances[anchor_idx]
        anchor_building = building_codes[anchor_idx]

        mask_valid = (neighbor_idx_row != anchor_idx) & (
            building_codes[neighbor_idx_row] != anchor_building
        )
        filtered_idx = neighbor_idx_row[mask_valid][: config.k_neighbors]
        filtered_dist = neighbor_dist_row[mask_valid][: config.k_neighbors]

        anchor_neighbor_indices.append(filtered_idx.astype(np.int32, copy=False))
        anchor_distances.append(filtered_dist.astype(np.float32, copy=False))
        negative_counts.append(int(filtered_idx.size))

    all_distance_chunks = [arr for arr in anchor_distances if arr.size > 0]
    if all_distance_chunks:
        concatenated = np.concatenate(all_distance_chunks).astype(np.float32, copy=False)
        quantiles = np.linspace(0.0, 1.0, config.bands + 1, dtype=np.float64)
        band_edges = np.quantile(concatenated, quantiles).astype(np.float32)
        band_edges[0] = 0.0
        band_edges = np.maximum.accumulate(band_edges)
    else:
        band_edges = np.zeros(config.bands + 1, dtype=np.float32)
        quantiles = np.linspace(0.0, 1.0, config.bands + 1, dtype=np.float64)

    anchor_band_lists: list[list[int]] = []
    for dist_array in anchor_distances:
        if dist_array.size == 0:
            anchor_band_lists.append([])
            continue
        bands = np.searchsorted(band_edges, dist_array, side="right") - 1
        bands = np.clip(bands, 0, config.bands - 1)
        anchor_band_lists.append(bands.astype(int).tolist())

    records: list[dict[str, Any]] = []
    shortfall_count = 0
    total_negatives = 0

    for anchor_idx in range(num_targets):
        neighbor_idx = anchor_neighbor_indices[anchor_idx]
        dist_array = anchor_distances[anchor_idx]
        bands_list = anchor_band_lists[anchor_idx]
        negative_count = negative_counts[anchor_idx]
        if negative_count < config.k_neighbors:
            shortfall_count += 1
        total_negatives += negative_count

        neighbor_target_ids = [target_keys[i] for i in neighbor_idx] if negative_count else []
        neighbor_distances = dist_array.tolist() if negative_count else []

        records.append(
            {
                "target_id": target_ids[anchor_idx],
                "dataset_target_id": target_keys[anchor_idx],
                "anchor_idx": int(anchor_idx),
                "building_id": building_ids[anchor_idx],
                "negative_count": negative_count,
                "neighbor_target_ids": neighbor_target_ids,
                "neighbor_distances": neighbor_distances,
                "neighbor_bands": bands_list,
                "band_edges": None,
                "config_json": None,
                "run_metadata_json": None,
                "is_metadata": False,
            }
        )

    run_metadata = {
        "total_targets": num_targets,
        "total_buildings": int(len(unique_buildings)),
        "k_neighbors": config.k_neighbors,
        "bands": config.bands,
        "total_negatives": int(total_negatives),
        "anchors_with_shortfall": int(shortfall_count),
        "n_neighbors_requested": int(n_neighbors_search),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    metadata_record = {
        "target_id": "__metadata__",
        "dataset_target_id": "__metadata__",
        "anchor_idx": -1,
        "building_id": "__metadata__",
        "negative_count": None,
        "neighbor_target_ids": [],
        "neighbor_distances": [],
        "neighbor_bands": [],
        "band_edges": band_edges.tolist(),
        "config_json": config.to_json(),
        "run_metadata_json": json.dumps(run_metadata, indent=2, sort_keys=True),
        "is_metadata": True,
    }
    records.append(metadata_record)

    neighbors_df = pd.DataFrame(records)

    metadata = {
        "band_edges": band_edges.tolist(),
        "quantiles": quantiles.tolist(),
        "run_metadata": run_metadata,
    }
    return neighbors_df, metadata, processed_df


def write_outputs(
    neighbors_df: pd.DataFrame,
    index_map_df: pd.DataFrame,
    metadata: dict[str, Any],
    config: DifficultyConfig,
    logger: logging.Logger,
) -> None:
    """Persist the neighbors and index map parquet files."""
    config.output_dir.mkdir(parents=True, exist_ok=True)

    neighbors_path = config.output_dir / config.neighbors_filename
    index_map_path = config.output_dir / config.index_map_filename

    if neighbors_path.exists() and not config.overwrite:
        raise FileExistsError(f"Neighbors parquet already exists: {neighbors_path}")
    if index_map_path.exists() and not config.overwrite:
        raise FileExistsError(f"Index map parquet already exists: {index_map_path}")

    neighbors_df.to_parquet(
        neighbors_path,
        engine="pyarrow",
        compression="snappy",
        index=False,
    )
    logger.info("Wrote neighbors parquet: %s", neighbors_path)

    index_map_df.to_parquet(
        index_map_path,
        engine="pyarrow",
        compression="snappy",
        index=False,
    )
    logger.info("Wrote index map parquet: %s", index_map_path)

    logger.info(
        "Difficulty metadata summary: %s",
        json.dumps(metadata["run_metadata"], indent=2, sort_keys=True),
    )


def compute_difficulty(config: DifficultyConfig, logger: logging.Logger) -> int:
    """Main computation pipeline."""
    if not config.source_parquet.exists():
        logger.error("Source parquet not found: %s", config.source_parquet)
        return 1

    logger.info("Reading source parquet: %s", config.source_parquet)
    df = pd.read_parquet(config.source_parquet, engine="pyarrow")

    start_time = datetime.now(timezone.utc)

    try:
        neighbors_df, metadata, processed_df = compute_neighbors(df, config, logger)
    except Exception as exc:
        logger.error("Failed to compute neighbors: %s", exc, exc_info=True)
        return 1

    index_map_df = processed_df[
        ["anchor_idx", "target_id", "dataset_target_id", "building_id", "target_lat", "target_lon"]
    ].copy()
    index_map_df["target_id"] = index_map_df["target_id"].astype(str)
    index_map_df["dataset_target_id"] = index_map_df["dataset_target_id"].astype(str)
    index_map_df["building_id"] = index_map_df["building_id"].astype(str)

    try:
        write_outputs(neighbors_df, index_map_df, metadata, config, logger)
    except FileExistsError as exc:
        logger.error(str(exc))
        return 1

    elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
    logger.info("Completed difficulty metadata computation in %.2f seconds", elapsed)
    return 0


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Precompute difficulty metadata for target buildings.",
    )
    parser.add_argument("--config", type=Path, help="Path to TOML config file.")
    return parser.parse_args()


def main() -> int:
    """CLI entrypoint."""
    args = parse_args()
    logger = setup_logging(None)

    try:
        if args.config:
            config = load_config_from_file(args.config, "difficulty_metadata")
        else:
            config = DifficultyConfig()
    except Exception as exc:
        logger.error("Error loading configuration: %s", exc)
        return 1

    if config.log_file:
        logger = setup_logging(config.log_file)

    return compute_difficulty(config, logger)


if __name__ == "__main__":
    sys.exit(main())

