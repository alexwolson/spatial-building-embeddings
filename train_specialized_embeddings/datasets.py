"""
Dataset and sampler implementations for triplet loss training.

Provides:
- TripletDataset: Groups embeddings by building_id and yields triplets
- UCBDifficultySampler: UCB-based sampling of difficulty bands for negative selection
"""

import logging
from collections import defaultdict
from typing import NamedTuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

logger = logging.getLogger(__name__)


class TripletSample(NamedTuple):
    """A single triplet sample."""

    anchor_idx: int
    positive_idx: int
    negative_idx: int
    anchor_embedding: torch.Tensor
    positive_embedding: torch.Tensor
    negative_embedding: torch.Tensor
    difficulty_band: int


class TripletDataset(Dataset):
    """
    Dataset that groups embeddings by building_id and provides triplet sampling.

    For each building, all images of that building are positives for each other.
    Negatives are sampled from other buildings using UCB-guided difficulty band selection.
    """

    def __init__(
        self,
        embeddings_df: pd.DataFrame,
        difficulty_metadata_df: pd.DataFrame,
        config,
    ):
        """
        Initialize triplet dataset.

        Args:
            embeddings_df: DataFrame with columns: building_id, embedding (list of floats)
            difficulty_metadata_df: DataFrame with columns: target_coord_hash, neighbor_building_ids,
                                   neighbor_bands, neighbor_distances_meters
            config: Training configuration object
        """
        self.config = config

        # Validate required columns
        required_cols = {"building_id", "embedding"}
        missing = required_cols - set(embeddings_df.columns)
        if missing:
            raise ValueError(f"Missing required columns in embeddings_df: {missing}")
            
        # Ensure streetview_image_id is present
        if "streetview_image_id" not in embeddings_df.columns:
            # Fallback for legacy data or if merge_and_split didn't add it
            # This is risky if IDs don't match metadata, but better than crashing if reproducible
            if "dataset_id" in embeddings_df.columns and "patch_id" in embeddings_df.columns:
                 dataset_str = embeddings_df["dataset_id"].astype(int).astype(str).str.zfill(4)
                 patch_str = embeddings_df["patch_id"].astype(int).astype(str)
                 self.streetview_image_ids = (dataset_str + "_" + patch_str).to_numpy(copy=False)
            elif "target_coord_hash" in embeddings_df.columns:
                 # Fallback to coordinate hash if that was used as key (legacy spatial mode)
                 self.streetview_image_ids = embeddings_df["target_coord_hash"].astype(str).to_numpy(copy=False)
            else:
                 raise ValueError("streetview_image_id column missing and cannot be inferred.")
        else:
            self.streetview_image_ids = embeddings_df["streetview_image_id"].astype(str).to_numpy(copy=False)

        # Materialise embeddings once and place them in shared memory for DataLoader workers
        logger.info("Materialising embedding tensor for triplet sampling...")
        embedding_matrix = np.stack(
            embeddings_df["embedding"].to_numpy(),  # array of lists
            axis=0,
        ).astype(np.float32, copy=False)
        self.embeddings = torch.as_tensor(embedding_matrix)
        self.embeddings.share_memory_()
        del embedding_matrix

        self.building_ids = (
            embeddings_df["building_id"].astype(str).to_numpy(copy=False)
        )

        # Group indices by building_id
        self.building_to_indices: dict[str, list[int]] = defaultdict(list)
        for idx, building_id in enumerate(self.building_ids):
            self.building_to_indices[building_id].append(idx)

        # Filter out buildings with only one image (can't form triplets) and cache index arrays
        self.valid_buildings: dict[str, np.ndarray] = {}
        for bid, indices in self.building_to_indices.items():
            if len(indices) > 1:
                self.valid_buildings[bid] = np.asarray(indices, dtype=np.int64)

        logger.info(
            f"Loaded {len(self.embeddings)} embeddings from {len(self.building_to_indices)} buildings "
            f"({len(self.valid_buildings)} with >1 image)"
        )

        # Pre-compute valid building ids list for sampling
        self.valid_building_ids = list(self.valid_buildings.keys())
        self.valid_building_ids_array = np.asarray(
            self.valid_building_ids, dtype=object
        )
        self.num_valid_buildings = len(self.valid_building_ids)
        self.total_anchor_candidates = sum(
            indices.size for indices in self.valid_buildings.values()
        )
        if self.total_anchor_candidates == 0:
            raise ValueError(
                "No buildings with at least two images available for triplet sampling."
            )

        # Determine epoch length: use full set of anchor candidates
        self.samples_per_epoch = self.total_anchor_candidates

        # Build difficulty metadata index
        # Map streetview_image_id to neighbors and difficulty bands
        self._build_difficulty_index(difficulty_metadata_df)

        # Initialize UCB sampler
        self.ucb_sampler = UCBDifficultySampler(
            exploration_constant=config.ucb_exploration_constant,
            warmup_samples=config.ucb_warmup_samples,
        )

        # Track statistics
        self.total_samples_generated = 0

    def _build_difficulty_index(self, difficulty_metadata_df: pd.DataFrame):
        """Build index mapping streetview_image_id to neighbors and difficulty bands."""
        required_cols = {"target_coord_hash", "neighbor_building_ids", "neighbor_bands"}
        missing = required_cols - set(difficulty_metadata_df.columns)
        if missing:
            raise ValueError(
                f"Missing required columns in difficulty_metadata_df: {missing}"
            )

        # Create mapping from image_id to neighbors by band
        # Note: In the new visual difficulty metadata, 'target_coord_hash' column actually contains 'streetview_image_id'
        self.image_to_neighbors: dict[str, list[str]] = {}
        self.image_to_bands: dict[str, list[int]] = {}

        # Index difficulty metadata by image ID (stored in target_coord_hash column)
        for _, row in difficulty_metadata_df.iterrows():
            image_id = str(row["target_coord_hash"]) # Effectively streetview_image_id
            
            neighbors = self._ensure_sequence(row["neighbor_building_ids"])
            bands = self._ensure_sequence(row["neighbor_bands"])
            
            self.image_to_neighbors[image_id] = neighbors
            self.image_to_bands[image_id] = bands

        logger.info(
            f"Indexed difficulty metadata for {len(self.image_to_neighbors)} images"
        )

    def __len__(self) -> int:
        """Return number of valid triplets (approximate)."""
        return self.samples_per_epoch

    def __getitem__(self, idx: int) -> TripletSample:
        """
        Get a triplet sample.

        Returns:
            TripletSample with anchor, positive, negative embeddings and difficulty band
        """
        # Sample a random building
        building_idx = np.random.randint(self.num_valid_buildings)
        building_id = str(self.valid_building_ids_array[building_idx])
        building_indices = self.valid_buildings[building_id]

        # Sample anchor and positive from same building
        anchor_idx, positive_idx = np.random.choice(
            building_indices, size=2, replace=False
        )
        anchor_idx = int(anchor_idx)
        positive_idx = int(positive_idx)

        # Sample negative using UCB-guided difficulty band selection
        # Pass the anchor's specific image ID to find its specific visual neighbors
        anchor_image_id = self.streetview_image_ids[anchor_idx]
        negative_idx, difficulty_band = self._sample_negative(anchor_image_id, building_id)

        # Get embeddings
        anchor_emb = self.embeddings[anchor_idx]
        positive_emb = self.embeddings[positive_idx]
        negative_emb = self.embeddings[negative_idx]

        return TripletSample(
            anchor_idx=anchor_idx,
            positive_idx=positive_idx,
            negative_idx=negative_idx,
            anchor_embedding=anchor_emb,
            positive_embedding=positive_emb,
            negative_embedding=negative_emb,
            difficulty_band=difficulty_band,
        )

    def _sample_negative(self, anchor_image_id: str, anchor_building_id: str) -> tuple[int, int]:
        """
        Sample a negative building using UCB-guided difficulty band selection.

        Returns:
            Tuple of (negative_idx, difficulty_band)
        """
        # Get neighbors and bands for this anchor image
        neighbors = self._ensure_sequence(
            self.image_to_neighbors.get(anchor_image_id)
        )
        bands = self._ensure_sequence(self.image_to_bands.get(anchor_image_id))

        if len(neighbors) == 0 or len(bands) == 0:
            # Fallback: sample random building that's not the anchor
            negative_building_id = self._random_building_id(
                exclude_id=anchor_building_id
            )
            negative_indices = self.valid_buildings[negative_building_id]
            negative_idx = int(np.random.choice(negative_indices))
            return negative_idx, -1  # Unknown band (fallback)

        # Group neighbors by difficulty band
        band_to_neighbors: dict[int, list[str]] = defaultdict(list)
        for neighbor, band in zip(neighbors, bands):
            band_to_neighbors[band].append(neighbor)

        # Use UCB to select difficulty band
        selected_band = self.ucb_sampler.select_band(list(band_to_neighbors.keys()))
        used_selected_band = True

        # Sample a neighbor from the selected band
        candidates = band_to_neighbors[selected_band]
        if not candidates:
            # Fallback to any neighbor
            candidates = neighbors
            used_selected_band = False

        # Filter to buildings that exist in our dataset
        valid_candidates = [c for c in candidates if c in self.valid_buildings]
        if not valid_candidates:
            # Fallback: sample any other building
            negative_building_id = self._random_building_id(
                exclude_id=anchor_building_id
            )
            used_selected_band = False
        else:
            negative_building_id = np.random.choice(valid_candidates)

        negative_indices = self.valid_buildings[negative_building_id]
        negative_idx = int(np.random.choice(negative_indices))

        return negative_idx, selected_band if used_selected_band else -1

    def _random_building_id(self, exclude_id: str | None = None) -> str:
        """Sample a random building id, optionally excluding one."""
        if self.num_valid_buildings == 0:
            raise ValueError("No valid buildings available for sampling.")

        if exclude_id is None or self.num_valid_buildings == 1:
            idx = np.random.randint(self.num_valid_buildings)
            return str(self.valid_building_ids_array[idx])

        # Retry until we pull a different building id. With many buildings this is cheap.
        while True:
            idx = np.random.randint(self.num_valid_buildings)
            candidate = str(self.valid_building_ids_array[idx])
            if candidate != exclude_id:
                return candidate

    def update_ucb_reward(self, difficulty_band: int, reward: float):
        """Update UCB sampler with reward for a difficulty band."""
        self.ucb_sampler.update_reward(difficulty_band, reward)

    @staticmethod
    def _ensure_sequence(value):
        """Normalize metadata fields that may arrive as numpy arrays or scalars."""
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, pd.Series):
            return value.tolist()
        return [value]


class UCBDifficultySampler:
    """
    Upper Confidence Bound (UCB) sampler for difficulty band selection.

    Maintains statistics per difficulty band and uses UCB to balance exploration
    and exploitation when selecting bands for negative sampling.
    """

    def __init__(self, exploration_constant: float = 2.0, warmup_samples: int = 1000):
        """
        Initialize UCB sampler.

        Args:
            exploration_constant: UCB exploration constant (c)
            warmup_samples: Number of samples before UCB kicks in (uses uniform sampling)
        """
        self.exploration_constant = exploration_constant
        self.warmup_samples = warmup_samples
        self.total_samples = 0

        # Per-band statistics
        self.band_counts: dict[int, int] = defaultdict(int)
        self.band_rewards: dict[int, float] = defaultdict(float)
        self.band_means: dict[int, float] = defaultdict(float)

    def select_band(self, available_bands: list[int]) -> int:
        """
        Select a difficulty band using UCB.

        Args:
            available_bands: List of available band IDs

        Returns:
            Selected band ID
        """
        if not available_bands:
            raise ValueError("No available bands provided")

        # Warmup phase: uniform sampling
        if self.total_samples < self.warmup_samples:
            return np.random.choice(available_bands)

        # UCB phase
        ucb_values = {}
        for band in available_bands:
            count = self.band_counts[band]
            if count == 0:
                # Never sampled: high priority
                ucb_values[band] = float("inf")
            else:
                mean_reward = self.band_means[band]
                confidence = self.exploration_constant * np.sqrt(
                    np.log(self.total_samples) / count
                )
                ucb_values[band] = mean_reward + confidence

        # Select band with highest UCB value
        selected_band = max(ucb_values.items(), key=lambda x: x[1])[0]
        return selected_band

    def update_reward(self, difficulty_band: int, reward: float):
        """
        Update statistics for a difficulty band.

        Args:
            difficulty_band: Band ID
            reward: Reward value (e.g., loss or distance)
        """
        self.total_samples += 1
        self.band_counts[difficulty_band] += 1
        self.band_rewards[difficulty_band] += reward

        # Update mean reward
        count = self.band_counts[difficulty_band]
        self.band_means[difficulty_band] = self.band_rewards[difficulty_band] / count

    def get_statistics(self) -> dict:
        """Get current UCB statistics."""
        return {
            "total_samples": self.total_samples,
            "band_counts": dict(self.band_counts),
            "band_means": dict(self.band_means),
        }
