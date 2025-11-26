#!/usr/bin/env python3
"""
Compute visual difficulty metadata for images.

The script reads fingerprint parquet files, computes PCA to reduce dimensionality,
builds a BallTree over the reduced visual features, finds nearest neighbors (excluding
same-building matches), calibrates difficulty band edges, and writes a single parquet
file with per-image neighbour information and metadata.
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
import pyarrow.parquet as pq
from rich.console import Console
from rich.logging import RichHandler
from sklearn.decomposition import PCA
from sklearn.neighbors import BallTree

from config import DifficultyMetadataConfig, load_config_from_file

MIN_CALIBRATION_SAMPLE = 10_000
FULL_DATASET_CALIBRATION_THRESHOLD = 250_000
CALIBRATION_PERCENTILES = np.array([5, 10, 20, 40, 60, 85], dtype=np.float64)


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


class FingerprintTable(NamedTuple):
    """Container for image fingerprints and metadata."""

    streetview_image_ids: np.ndarray
    building_ids: np.ndarray
    fingerprints: np.ndarray  # Shape: (N, D)


def load_fingerprints(fingerprints_dir: Path, logger: logging.Logger) -> FingerprintTable:
    """Load and concatenate all fingerprint parquet files."""
    logger.info("Loading fingerprints from %s", fingerprints_dir)
    parquet_files = sorted(fingerprints_dir.glob("*.parquet"))
    
    if not parquet_files:
        raise ValueError(f"No fingerprint files found in {fingerprints_dir}")

    dfs = []
    for p_file in parquet_files:
        try:
            df = pd.read_parquet(p_file, columns=["streetview_image_id", "building_id", "fingerprint"])
            dfs.append(df)
        except Exception as e:
            logger.warning(f"Failed to read {p_file}: {e}")

    if not dfs:
        raise ValueError("Failed to load any fingerprint data")

    full_df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Loaded {len(full_df)} total fingerprints")

    # Extract arrays
    streetview_image_ids = full_df["streetview_image_id"].astype(str).to_numpy()
    building_ids = full_df["building_id"].astype(str).to_numpy()
    
    # Fingerprints are stored as lists/arrays in parquet, need to stack them
    logger.info("Stacking fingerprint arrays...")
    fingerprints_list = full_df["fingerprint"].tolist()
    fingerprints = np.stack(fingerprints_list).astype(np.float32)
    
    # Normalize pixel values to 0-1 range
    fingerprints /= 255.0

    return FingerprintTable(
        streetview_image_ids=streetview_image_ids,
        building_ids=building_ids,
        fingerprints=fingerprints,
    )


def reduce_dimensionality(
    fingerprints: np.ndarray, n_components: int, seed: int, logger: logging.Logger
) -> np.ndarray:
    """Reduce dimensionality using PCA."""
    logger.info(f"Fitting PCA to reduce {fingerprints.shape[1]} dims to {n_components}...")
    pca = PCA(n_components=n_components, random_state=seed)
    reduced = pca.fit_transform(fingerprints)
    explained_variance = np.sum(pca.explained_variance_ratio_)
    logger.info(f"PCA explained variance: {explained_variance:.4f}")
    return reduced


def build_ball_tree(features: np.ndarray, logger: logging.Logger) -> BallTree:
    """Construct a BallTree over visual features."""
    logger.info("Building BallTree for %d images", features.shape[0])
    tree = BallTree(features, metric="euclidean")
    logger.info("BallTree construction complete")
    return tree


def compute_visual_neighbors(
    tree: BallTree,
    features: np.ndarray,
    building_ids: np.ndarray,
    target_neighbors: int,
    batch_size: int,
    logger: logging.Logger,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Compute neighbour indices and distances, filtering out same-building matches.
    
    We query more neighbors than needed (k * factor) to allow for filtering.
    """
    total = features.shape[0]
    # Query more neighbors to ensure we have enough after filtering same-building
    query_k = min(total, target_neighbors + 50) 
    
    final_indices = []
    final_distances = []

    logger.info(f"Querying {query_k} neighbors to find {target_neighbors} valid ones...")

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_features = features[start:end]
        
        dists, inds = tree.query(batch_features, k=query_k)
        
        # Filter logic
        batch_indices = []
        batch_distances = []
        
        for i in range(end - start):
            anchor_idx = start + i
            anchor_building = building_ids[anchor_idx]
            
            # Get candidates for this anchor
            candidates_ind = inds[i]
            candidates_dist = dists[i]
            
            # Filter: exclude same building
            # (Note: this also excludes self, as self has same building_id)
            candidate_buildings = building_ids[candidates_ind]
            mask = candidate_buildings != anchor_building
            
            valid_ind = candidates_ind[mask]
            valid_dist = candidates_dist[mask]
            
            if len(valid_ind) < target_neighbors:
                # This assumes we found enough. If not, we take what we have.
                # In a huge dataset, this is rare unless query_k is too small.
                pass
            else:
                valid_ind = valid_ind[:target_neighbors]
                valid_dist = valid_dist[:target_neighbors]
            
            batch_indices.append(valid_ind)
            batch_distances.append(valid_dist)
            
        final_indices.extend(batch_indices)
        final_distances.extend(batch_distances)
        
        if (start // batch_size) % 10 == 0:
            logger.info(f"Processed {end}/{total} queries")

    return final_indices, final_distances


def _extract_positive_local_scales(
    distances_list: list[np.ndarray],
    k0: int,
    logger: logging.Logger | None = None,
) -> np.ndarray:
    """Return the k0-th neighbour distance for each anchor."""
    # Convert list of arrays to array (padding if necessary, but here we expect mostly uniform)
    # Actually, lengths might vary if we didn't find enough neighbors.
    # Safe approach: iterate.
    
    local_scales = np.zeros(len(distances_list), dtype=np.float32)
    
    count_short = 0
    for i, dists in enumerate(distances_list):
        if len(dists) >= k0:
            scale = dists[k0 - 1]
        elif len(dists) > 0:
            scale = dists[-1] # Fallback to furthest available
            count_short += 1
        else:
            scale = 1.0 # Fallback default
            count_short += 1
        
        if scale <= 0:
             # Fallback for zero distance (duplicate images)
             # Find first positive
             pos_mask = dists > 0
             if np.any(pos_mask):
                 scale = dists[pos_mask][0]
             else:
                 scale = 1.0 # Ultimate fallback
                 
        local_scales[i] = scale
        
    if count_short > 0 and logger:
        logger.warning(f"Found {count_short} anchors with fewer than {k0} neighbors")
        
    return local_scales


def calibrate_band_edges(
    distances_list: list[np.ndarray],
    local_scales: np.ndarray,
    sample_fraction: float,
    rng: np.random.Generator,
    logger: logging.Logger,
) -> np.ndarray:
    """Calibrate difficulty band edges."""
    total = len(distances_list)
    if total == 0:
        raise ValueError("No anchors available for calibration.")

    sample_size = max(int(math.ceil(total * sample_fraction)), MIN_CALIBRATION_SAMPLE)
    sample_size = min(sample_size, total)
    
    sample_indices = rng.choice(total, size=sample_size, replace=False)
    
    sampled_ratios = []
    for idx in sample_indices:
        dists = distances_list[idx]
        scale = local_scales[idx]
        if len(dists) > 0 and scale > 0:
            sampled_ratios.extend(dists / scale)
            
    sampled_ratios = np.array(sampled_ratios, dtype=np.float32)
    edges = np.percentile(sampled_ratios, CALIBRATION_PERCENTILES)
    
    logger.info(
        "Calibrated band edges at percentiles %s: %s",
        CALIBRATION_PERCENTILES.tolist(),
        edges.tolist(),
    )
    return edges.astype(np.float32)


def assign_bands(
    distances_list: list[np.ndarray],
    local_scales: np.ndarray,
    edges: np.ndarray,
) -> list[np.ndarray]:
    """Assign difficulty bands."""
    bands_list = []
    for i, dists in enumerate(distances_list):
        scale = local_scales[i]
        if scale <= 0:
            bands = np.zeros_like(dists, dtype=np.int16) # Fallback
        else:
            ratios = dists / scale
            bands = np.searchsorted(edges, ratios, side="right").astype(np.int16)
        bands_list.append(bands)
    return bands_list


def write_parquet(
    fingerprints: FingerprintTable,
    neighbour_indices: list[np.ndarray],
    neighbour_distances: list[np.ndarray],
    bands: list[np.ndarray],
    output_path: Path,
    row_group_size: int,
    metadata: dict[str, str],
) -> None:
    """Write the difficulty metadata parquet file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # We need to map neighbor INDICES back to neighbor BUILDING IDs
    # neighbour_indices contains indices into fingerprints.building_ids
    
    neighbor_building_ids_list = []
    for indices in neighbour_indices:
        neighbor_building_ids_list.append(fingerprints.building_ids[indices].tolist())
        
    neighbour_distances_list = [d.tolist() for d in neighbour_distances]
    bands_list = [b.tolist() for b in bands]

    # Create target_coord_hash equivalent: streetview_image_id
    # The training code expects `target_coord_hash` as the key to lookup metadata.
    # Since we are now doing image-based lookup, we should map `streetview_image_id` 
    # to the column expected by the trainer, OR update the trainer to look up by `streetview_image_id`.
    # The plan says "Update _build_difficulty_index to index by streetview_image_id".
    # So we can output `streetview_image_id` here.
    
    # However, to avoid breaking existing schemas too much, let's keep the file consistent.
    # The current `difficulty_metadata.parquet` has `target_coord_hash`. 
    # We will output `target_coord_hash` as `streetview_image_id` effectively using the image ID as the hash.
    
    df = pd.DataFrame(
        {
            "target_coord_hash": fingerprints.streetview_image_ids, # Using ID as hash/key
            "neighbor_building_ids": neighbor_building_ids_list,
            "neighbor_distances_meters": neighbour_distances_list, # Actually visual distances
            "neighbor_bands": bands_list,
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
    logger.info("Visual difficulty metadata computation: starting")
    logger.info("=" * 60)
    
    # Use fingerprints_dir from config, now properly defined
    fingerprints_dir = config.fingerprints_dir
    
    data = load_fingerprints(fingerprints_dir, logger)
    
    # PCA
    reduced_features = reduce_dimensionality(
        data.fingerprints, 
        config.pca_components, 
        config.seed if config.seed is not None else 42, 
        logger
    )
    
    # BallTree
    tree = build_ball_tree(reduced_features, logger)
    
    # Neighbors
    indices, distances = compute_visual_neighbors(
        tree,
        reduced_features,
        data.building_ids,
        config.neighbors,
        config.batch_size,
        logger
    )
    
    # Local scales
    local_scales = _extract_positive_local_scales(
        distances,
        config.k0_for_local_scale,
        logger=logger,
    )
    
    # Calibration
    rng = np.random.default_rng(config.seed if config.seed is not None else 42)
    edges = calibrate_band_edges(
        distances,
        local_scales,
        config.sample_fraction_for_bands,
        rng,
        logger,
    )
    
    # Bands
    bands = assign_bands(distances, local_scales, edges)
    
    metadata = {
        "neighbors": str(config.neighbors),
        "k0_for_local_scale": str(config.k0_for_local_scale),
        "sample_fraction_for_bands": str(config.sample_fraction_for_bands),
        "band_edges": json.dumps(edges.tolist()),
        "distance_metric": "euclidean_visual",
        "pca_components": str(config.pca_components),
    }

    write_parquet(
        data,
        indices,
        distances,
        bands,
        config.difficulty_metadata_path,
        config.row_group_size,
        metadata,
    )

    logger.info("Wrote difficulty metadata to %s", config.difficulty_metadata_path)
    logger.info("Visual difficulty metadata computation complete")


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Compute visual difficulty metadata for buildings."
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to TOML config file.",
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

