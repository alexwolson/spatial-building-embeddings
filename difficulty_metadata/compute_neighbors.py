#!/usr/bin/env python3
"""
Compute global difficulty metadata for buildings.

The script reads merged split parquet files, builds a global BallTree over building
coordinates, calibrates difficulty band edges, and writes a single parquet file with
per-building neighbour information and global metadata.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from rich.console import Console
from rich.logging import RichHandler
from sklearn.neighbors import BallTree

from config import DifficultyMetadataConfig, load_config_from_file

EARTH_RADIUS_METERS = 6_371_008.8
MIN_CALIBRATION_SAMPLE = 10_000
FULL_DATASET_CALIBRATION_THRESHOLD = 250_000
CALIBRATION_PERCENTILES = np.array([5, 10, 20, 40, 60, 85], dtype=np.float64)
RNG_SEED = 42


def _extract_positive_local_scales(
    distances: np.ndarray,
    k0: int,
    *,
    logger: logging.Logger | None = None,
) -> np.ndarray:
    """
    Return the k0-th neighbour distance for each anchor, falling back to the
    nearest strictly-positive distance when duplicates collapse the radius to zero.
    """
    if k0 < 1 or k0 > distances.shape[1]:
        raise ValueError(f"k0 ({k0}) must be within the neighbour range.")

    positive_mask = distances > 0.0
    if not positive_mask.any():
        raise ValueError(
            "All neighbour distances are zero; unable to derive local scales."
        )

    global_min_positive = distances[positive_mask].min()
    local_scales = distances[:, k0 - 1].astype(np.float64, copy=False)
    non_positive_mask = local_scales <= 0.0

    if non_positive_mask.any():
        fallback_scales = np.where(positive_mask, distances, np.inf).min(axis=1)
        local_scales = np.where(non_positive_mask, fallback_scales, local_scales)

        if logger:
            logger.warning(
                "Adjusted %d anchors where k0-th neighbour distance was non-positive.",
                int(non_positive_mask.sum()),
            )

    still_bad_mask = (~np.isfinite(local_scales)) | (local_scales <= 0.0)
    if still_bad_mask.any():
        local_scales = np.where(still_bad_mask, global_min_positive, local_scales)
        if logger:
            logger.warning(
                "Replaced %d anchors with global minimum positive distance due to missing fallback.",
                int(still_bad_mask.sum()),
            )

    still_bad_mask = (~np.isfinite(local_scales)) | (local_scales <= 0.0)
    if still_bad_mask.any():
        problematic = np.where(still_bad_mask)[0][:5]
        raise ValueError(
            "Unable to determine a positive local scale for all anchors; "
            f"sample indices with issues: {problematic.tolist()}"
        )

    return local_scales


class BuildingTable(NamedTuple):
    """Container for building-level arrays."""

    target_coord_hashes: np.ndarray
    building_ids: np.ndarray
    lat_radians: np.ndarray
    lon_radians: np.ndarray
    lat_degrees: np.ndarray
    lon_degrees: np.ndarray


def setup_logging(log_file: Path | None = None) -> logging.Logger:
    """Set up logging with Rich handler."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    if log_file:
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


def validate_coords(
    lat_deg: np.ndarray, lon_deg: np.ndarray, logger: logging.Logger
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate latitude and longitude bounds, returning masks for valid rows."""
    valid_lat = np.isfinite(lat_deg) & (lat_deg >= -90.0) & (lat_deg <= 90.0)
    valid_lon = np.isfinite(lon_deg) & (lon_deg >= -180.0) & (lon_deg <= 180.0)
    valid_mask = valid_lat & valid_lon
    invalid_count = np.size(valid_mask) - int(valid_mask.sum())
    if invalid_count:
        logger.warning("Dropping %d rows with invalid coordinates", invalid_count)
    return lat_deg[valid_mask], lon_deg[valid_mask], valid_mask


def load_buildings(dataset_path: Path, logger: logging.Logger) -> BuildingTable:
    """Load and deduplicate building-level coordinates from merged parquet splits."""
    logger.info("Loading building tables from %s", dataset_path)
    dataset = ds.dataset(dataset_path)

    required_columns = [
        "building_id",
        "target_lat",
        "target_lon",
        "target_coord_hash",
    ]
    missing = [
        column for column in required_columns if column not in dataset.schema.names
    ]
    if missing:
        raise ValueError(f"Missing required columns in dataset: {', '.join(missing)}")

    table = dataset.to_table(columns=required_columns)

    building_ids = table.column("building_id").to_numpy(zero_copy_only=False)
    lat_deg = table.column("target_lat").to_numpy(zero_copy_only=False)
    lon_deg = table.column("target_lon").to_numpy(zero_copy_only=False)
    coord_hashes = table.column("target_coord_hash").to_numpy(zero_copy_only=False)

    lat_deg, lon_deg, valid_mask = validate_coords(lat_deg, lon_deg, logger)
    building_ids = building_ids[valid_mask]
    coord_hashes = coord_hashes[valid_mask]

    building_ids = building_ids.astype("U")
    coord_hashes = coord_hashes.astype("U")

    # Deduplicate by coordinate hash, selecting the lexicographically smallest building_id per coordinate.
    sort_order = np.lexsort((building_ids, coord_hashes))
    sorted_hashes = coord_hashes[sort_order]
    dedup_mask = np.ones(sorted_hashes.shape[0], dtype=bool)
    dedup_mask[1:] = sorted_hashes[1:] != sorted_hashes[:-1]
    dedup_indices = sort_order[dedup_mask]

    logger.info(
        "Retained %d unique coordinate hashes out of %d valid rows",
        len(dedup_indices),
        len(coord_hashes),
    )

    if len(dedup_indices) < len(coord_hashes):
        logger.info(
            "Removed %d duplicate coordinate entries",
            len(coord_hashes) - len(dedup_indices),
        )

    building_ids = building_ids[dedup_indices]
    coord_hashes = coord_hashes[dedup_indices]
    lat_deg = lat_deg[dedup_indices]
    lon_deg = lon_deg[dedup_indices]

    order = np.argsort(coord_hashes, kind="mergesort")

    coord_hashes = coord_hashes[order].astype("U")
    building_ids = building_ids[order].astype("U")
    lat_deg = lat_deg[order]
    lon_deg = lon_deg[order]

    lat_rad = np.deg2rad(lat_deg)
    lon_rad = np.deg2rad(lon_deg)

    return BuildingTable(
        target_coord_hashes=coord_hashes,
        building_ids=building_ids,
        lat_radians=lat_rad,
        lon_radians=lon_rad,
        lat_degrees=lat_deg,
        lon_degrees=lon_deg,
    )


def build_ball_tree(coords_rad: np.ndarray, logger: logging.Logger) -> BallTree:
    """Construct a BallTree over building coordinates (haversine metric)."""
    logger.info("Building BallTree for %d buildings", coords_rad.shape[0])
    tree = BallTree(coords_rad, metric="haversine")
    logger.info("BallTree construction complete")
    return tree


def compute_neighbour_distances(
    tree: BallTree,
    coords_rad: np.ndarray,
    neighbors: int,
    batch_size: int,
    distance_dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute neighbour indices and distances for all anchors."""
    total = coords_rad.shape[0]
    query_k = neighbors + 1  # include self

    indices = np.empty((total, neighbors), dtype=np.int32)
    distances = np.empty((total, neighbors), dtype=distance_dtype)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_coords = coords_rad[start:end]
        dist_rad, ind = tree.query(
            batch_coords,
            k=query_k,
            sort_results=True,
            return_distance=True,
        )

        # drop self (first column)
        indices_batch = ind[:, 1:]
        dists_batch = dist_rad[:, 1:] * EARTH_RADIUS_METERS

        indices[start:end] = indices_batch.astype(np.int32, copy=False)
        distances[start:end] = dists_batch.astype(distance_dtype, copy=False)

    return indices, distances


def calibrate_band_edges(
    distances: np.ndarray,
    k0: int,
    sample_fraction: float,
    rng: np.random.Generator,
    logger: logging.Logger,
) -> np.ndarray:
    """
    Calibrate difficulty band edges using a sample of anchors.

    Returns the quantile edges (excluding 0 and 1).
    """
    total = distances.shape[0]
    if total == 0:
        raise ValueError("No anchors available for calibration.")

    if total <= FULL_DATASET_CALIBRATION_THRESHOLD:
        sample_indices = np.arange(total, dtype=np.int64)
        logger.info(
            "Using all %d anchors for band calibration (<= %d threshold)",
            total,
            FULL_DATASET_CALIBRATION_THRESHOLD,
        )
    else:
        sample_size = max(
            int(math.ceil(total * sample_fraction)), MIN_CALIBRATION_SAMPLE
        )
        sample_size = min(sample_size, total)
        logger.info(
            "Sampling %d anchors (%.2f%%) for band calibration",
            sample_size,
            (sample_size / total) * 100,
        )
        sample_indices = rng.choice(total, size=sample_size, replace=False)
    sampled_distances = distances[sample_indices].astype(np.float64, copy=False)

    local_scales = _extract_positive_local_scales(sampled_distances, k0, logger=logger)
    standardized = sampled_distances / local_scales[:, None]
    edges = np.percentile(standardized, CALIBRATION_PERCENTILES, axis=None)
    logger.info(
        "Calibrated band edges at percentiles %s: %s",
        CALIBRATION_PERCENTILES.tolist(),
        edges.tolist(),
    )
    return edges.astype(np.float32, copy=False)


def assign_bands(
    distances: np.ndarray, local_scales: np.ndarray, edges: np.ndarray
) -> np.ndarray:
    """Assign difficulty bands using global quantile edges."""
    if np.any(local_scales <= 0):
        raise ValueError("Encountered non-positive local scale while assigning bands.")
    standardized = (
        distances.astype(np.float64, copy=False)
        / local_scales.astype(np.float64, copy=False)[:, None]
    )
    bands = np.searchsorted(edges, standardized, side="right")
    return bands.astype(np.int16, copy=False)


def write_parquet(
    buildings: BuildingTable,
    neighbour_indices: np.ndarray,
    neighbour_distances: np.ndarray,
    bands: np.ndarray,
    output_path: Path,
    row_group_size: int,
    distance_dtype: str,
    metadata: dict[str, str],
) -> None:
    """Write the difficulty metadata parquet file using a DataFrame workflow."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dtype_np = np.float32 if distance_dtype == "float32" else np.float64
    neighbor_ids = buildings.building_ids[neighbour_indices]
    neighbor_distances = neighbour_distances.astype(dtype_np, copy=False)

    df = pd.DataFrame(
        {
            "target_coord_hash": buildings.target_coord_hashes,
            "target_lat": buildings.lat_degrees,
            "target_lon": buildings.lon_degrees,
            "neighbor_building_ids": neighbor_ids.tolist(),
            "neighbor_distances_meters": neighbor_distances.tolist(),
            "neighbor_bands": bands.tolist(),
        }
    )

    table = pa.Table.from_pandas(df, preserve_index=False)
    if metadata:
        encoded_metadata = {
            key.encode("utf-8"): value.encode("utf-8")
            for key, value in metadata.items()
        }
        existing_metadata = table.schema.metadata or {}
        table = table.replace_schema_metadata({**existing_metadata, **encoded_metadata})

    pq.write_table(
        table, output_path, compression="snappy", row_group_size=row_group_size
    )


def compute_difficulty_metadata(config: DifficultyMetadataConfig) -> None:
    """Main orchestration routine."""
    logger = setup_logging(None)
    logger.info("=" * 60)
    logger.info("Difficulty metadata computation: starting")
    logger.info("=" * 60)

    buildings = load_buildings(config.input_parquet_path, logger)
    total_buildings = buildings.building_ids.size
    if total_buildings <= config.neighbors:
        raise ValueError(
            f"Not enough buildings ({total_buildings}) to query {config.neighbors} neighbours per anchor."
        )

    coords_rad = np.column_stack((buildings.lat_radians, buildings.lon_radians))
    tree = build_ball_tree(coords_rad, logger)
    distance_dtype_np = np.float32 if config.distance_dtype == "float32" else np.float64
    indices, distances = compute_neighbour_distances(
        tree=tree,
        coords_rad=coords_rad,
        neighbors=config.neighbors,
        batch_size=config.batch_size,
        distance_dtype=distance_dtype_np,
    )

    local_scales = _extract_positive_local_scales(
        distances,
        config.k0_for_local_scale,
        logger=logger,
    )
    rng = np.random.default_rng(RNG_SEED)
    edges = calibrate_band_edges(
        distances=distances,
        k0=config.k0_for_local_scale,
        sample_fraction=config.sample_fraction_for_bands,
        rng=rng,
        logger=logger,
    )

    bands = assign_bands(distances, local_scales, edges)

    metadata = {
        "neighbors": str(config.neighbors),
        "k0_for_local_scale": str(config.k0_for_local_scale),
        "sample_fraction_for_bands": str(config.sample_fraction_for_bands),
        "band_edges": json.dumps(edges.tolist()),
        "distance_dtype": config.distance_dtype,
        "batch_size": str(config.batch_size),
        "row_group_size": str(config.row_group_size),
        "total_buildings": str(total_buildings),
    }

    write_parquet(
        buildings=buildings,
        neighbour_indices=indices,
        neighbour_distances=distances,
        bands=bands,
        output_path=config.output_parquet_path,
        row_group_size=config.row_group_size,
        distance_dtype=config.distance_dtype,
        metadata=metadata,
    )

    logger.info("Wrote difficulty metadata to %s", config.output_parquet_path)
    logger.info("Difficulty metadata computation complete")


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Compute difficulty metadata for buildings."
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to TOML config file. If not provided, config is loaded from environment variables.",
    )
    args = parser.parse_args()

    logger = setup_logging(None)

    try:
        if args.config:
            config = load_config_from_file(args.config, "difficulty_metadata")
        else:
            config = DifficultyMetadataConfig()
    except Exception as exc:
        logger.error("Error loading configuration: %s", exc)
        return 1

    try:
        compute_difficulty_metadata(config)
        return 0
    except Exception as exc:
        logger.error("Execution failed: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
