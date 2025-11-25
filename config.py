"""
Unified configuration management for the spatial building embeddings pipeline.

Supports loading configuration from:
- Environment variables (with prefixes)
- TOML config files with type-based sections
- Command-line argument to specify config file path

The unified config file structure:
- [global] section: Shared settings (seed, log_dir)
- [paths] section: All pipeline file and directory paths
- [embedding_model] section: Model settings for embedding generation
- [training_model] section: Model architecture settings for training
- [training] section: Training hyperparameters and settings
- [data] section: Data processing settings (splits, neighbors, etc.)
- [infrastructure] section: Device, workers, and performance settings
- [logging] section: Logging and monitoring settings

Workflows automatically read from the relevant sections based on their needs.
"""

from pathlib import Path
from typing import Literal, Union

import tomllib
from pydantic import Field, PositiveFloat, PositiveInt, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class GlobalConfig(BaseSettings):
    """Global configuration shared across all workflows."""

    model_config = SettingsConfigDict(
        env_prefix="GLOBAL_",
        case_sensitive=False,
        extra="ignore",
    )

    seed: int = Field(42, description="Random seed for reproducibility")
    log_dir: Path = Field(
        ...,
        description="Base directory for all log files (mandatory)",
    )


class ProcessTarConfig(BaseSettings):
    """Configuration for tar preprocessing (per-tar parquet extraction)."""

    model_config = SettingsConfigDict(
        env_prefix="PROCESS_TAR_",
        case_sensitive=False,
        extra="ignore",
    )

    tar_file: Path = Field(..., description="Path to tar file to process")
    intermediates_dir: Path = Field(
        ..., description="Output directory for intermediate Parquet files (shared pipeline location)"
    )
    log_file: Path | None = Field(None, description="Optional log file path")
    temp_dir: Path | None = Field(
        None, description="Optional temporary directory for extraction"
    )
    seed: int | None = Field(
        None, description="Random seed (inherits from global if not set)"
    )


class MergeAndSplitConfig(BaseSettings):
    """Configuration for dataset assembly (merging and splitting intermediate files)."""

    model_config = SettingsConfigDict(
        env_prefix="MERGE_",
        case_sensitive=False,
        extra="ignore",
    )

    intermediates_dir: Path = Field(
        ..., description="Directory containing intermediate Parquet files"
    )
    merged_dir: Path = Field(
        ..., description="Directory for final output Parquet files (shared pipeline location)"
    )
    embeddings_dir: Path = Field(
        ..., description="Directory containing per-tar embedding Parquet files"
    )
    train_ratio: float = Field(0.7, ge=0.0, le=1.0, description="Training set ratio")
    val_ratio: float = Field(0.15, ge=0.0, le=1.0, description="Validation set ratio")
    test_ratio: float = Field(0.15, ge=0.0, le=1.0, description="Test set ratio")
    seed: int | None = Field(
        None, description="Random seed (inherits from global if not set)"
    )
    log_file: Path | None = Field(None, description="Optional log file path")

    def model_post_init(self, __context):
        """Validate that ratios sum to 1.0."""
        ratio_sum = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(ratio_sum - 1.0) > 1e-6:
            raise ValueError(f"Ratios must sum to 1.0, got {ratio_sum}")


class GenerateEmbeddingsConfig(BaseSettings):
    """Configuration for embedding generation."""

    model_config = SettingsConfigDict(
        env_prefix="GENERATE_EMBEDDINGS_",
        case_sensitive=False,
        extra="ignore",
    )

    parquet_file: Path | None = Field(
        None, description="Path to intermediate parquet file to process"
    )
    tar_file: Path | None = Field(
        None, description="Optional: tar file path (auto-detect if None)"
    )
    embeddings_dir: Path = Field(
        ..., description="Output directory for embedding parquet files (shared pipeline location)"
    )
    model_name: str = Field(
        "vit_base_patch14_dinov2.lvd142m", description="Timm model name"
    )
    batch_size: int = Field(
        128,
        ge=1,
        description="Batch size for inference (default: 128 for H100 80GB GPUs on Nibi)",
    )
    # Note: GPU is always required - no device parameter, script will fail if GPU unavailable
    temp_dir: Path | None = Field(
        None, description="Optional temporary directory for extraction"
    )
    log_file: Path | None = Field(None, description="Optional log file path")

    def model_post_init(self, __context):
        """Auto-detect tar file if not provided."""
        if self.parquet_file is not None and self.tar_file is None:
            self.tar_file = self._auto_detect_tar_file()

    def _auto_detect_tar_file(self) -> Path:
        """
        Auto-detect tar file from parquet filename.

        Extracts dataset_id from parquet filename (e.g., '0061.parquet' -> 61)
        and looks for corresponding tar file in same directory or data/raw/.
        """
        # Extract dataset_id from parquet filename (e.g., '0061.parquet' -> 61)
        parquet_stem = self.parquet_file.stem
        try:
            dataset_id = int(parquet_stem)
        except ValueError:
            raise ValueError(
                f"Could not extract dataset ID from parquet filename: {self.parquet_file}. "
                "Expected format: <dataset_id>.parquet (e.g., 0061.parquet)"
            )

        # Try same directory as parquet file first
        tar_file_same_dir = self.parquet_file.parent / f"{dataset_id:04d}.tar"
        if tar_file_same_dir.exists():
            return tar_file_same_dir

        # Try data/raw/ directory (relative to parquet file's parent's parent)
        # Assuming structure: data/intermediates/0061.parquet -> data/raw/0061.tar
        raw_dir = self.parquet_file.parent.parent / "raw"
        tar_file_raw = raw_dir / f"{dataset_id:04d}.tar"
        if tar_file_raw.exists():
            return tar_file_raw
            
        # Try raw/dataset_unaligned/ directory (another common structure)
        # Assuming structure: data/intermediates/0061.parquet -> data/raw/dataset_unaligned/0061.tar
        raw_unaligned_dir = raw_dir / "dataset_unaligned"
        tar_file_unaligned = raw_unaligned_dir / f"{dataset_id:04d}.tar"
        if tar_file_unaligned.exists():
            return tar_file_unaligned

        # Try absolute path data/raw/ from project root
        # Walk up from parquet file to find project root (has pyproject.toml)
        current = self.parquet_file.parent
        for _ in range(5):  # Limit search depth
            if (current / "pyproject.toml").exists():
                tar_file_project = current / "data" / "raw" / f"{dataset_id:04d}.tar"
                if tar_file_project.exists():
                    return tar_file_project
                
                # Check data/raw/dataset_unaligned/ from project root
                tar_file_project_unaligned = current / "data" / "raw" / "dataset_unaligned" / f"{dataset_id:04d}.tar"
                if tar_file_project_unaligned.exists():
                    return tar_file_project_unaligned

            if current.parent == current:  # Reached root
                break
            current = current.parent

        raise FileNotFoundError(
            f"Could not find tar file for dataset ID {dataset_id:04d}. "
            f"Tried: {tar_file_same_dir}, {tar_file_raw}, {tar_file_unaligned}, and {tar_file_project} (and dataset_unaligned variants)"
        )


class TripletTrainingConfig(BaseSettings):
    """Configuration for triplet loss training."""

    model_config = SettingsConfigDict(
        env_prefix="TRIPLET_TRAINING_",
        case_sensitive=False,
        extra="ignore",
    )

    # Data paths
    train_parquet_path: Path = Field(
        ..., description="Path to training split parquet file"
    )
    val_parquet_path: Path = Field(
        ..., description="Path to validation split parquet file"
    )
    difficulty_metadata_path: Path = Field(
        ..., description="Path to difficulty_metadata.parquet"
    )
    checkpoint_dir: Path = Field(..., description="Directory for saving checkpoints")

    # Model architecture
    input_dim: int = Field(
        768, description="Input embedding dimension (nomic-embed-vision-v1.5 = 768)"
    )
    hidden_dim: int = Field(512, description="Base hidden layer dimension")
    num_hidden_layers: PositiveInt = Field(
        1, description="Number of hidden layers in the projection head"
    )
    hidden_dim_multiplier: PositiveFloat = Field(
        1.0,
        description=(
            "Multiplier applied to hidden dimensions for deeper layers "
            "(values <1 create bottlenecks, >1 widen layers)"
        ),
    )
    activation: Literal["gelu", "relu", "silu"] = Field(
        "gelu", description="Activation function used between hidden layers"
    )
    output_dim: int = Field(256, description="Output embedding dimension")
    dropout: float = Field(0.1, ge=0.0, le=1.0, description="Dropout probability")
    use_residual: bool = Field(True, description="Use residual shortcut connection")
    use_layer_norm: bool = Field(
        True, description="Use LayerNorm after each hidden layer"
    )

    # Training hyperparameters
    batch_size: PositiveInt = Field(256, description="Batch size for training")
    num_epochs: PositiveInt = Field(50, description="Number of training epochs")
    learning_rate: PositiveFloat = Field(1e-4, description="Learning rate")
    weight_decay: float = Field(1e-5, ge=0.0, description="Weight decay for optimizer")
    margin: PositiveFloat = Field(0.5, description="Triplet loss margin")
    loss_distance: Literal["euclidean", "cosine"] = Field(
        "euclidean", description="Distance metric for triplet loss"
    )

    # UCB sampler configuration
    ucb_exploration_constant: float = Field(
        2.0, ge=0.0, description="UCB exploration constant (c)"
    )
    ucb_warmup_samples: int = Field(
        1000, ge=0, description="Number of warmup samples before UCB kicks in"
    )

    # Training configuration
    # device, num_workers, and pin_memory are determined dynamically in the code
    seed: int | None = Field(
        None, description="Random seed (inherits from global if not set)"
    )

    # Checkpointing and validation
    save_every_n_epochs: int = Field(
        5, ge=1, description="Save checkpoint every N epochs"
    )
    validate_every_n_epochs: int = Field(
        1, ge=1, description="Run validation every N epochs"
    )
    resume_from_checkpoint: Path | None = Field(
        None, description="Path to checkpoint to resume from"
    )
    early_stopping_patience: int = Field(
        0,
        ge=0,
        description="Number of validations with no improvement before stopping (0 disables early stopping)",
    )
    retrieval_metric_top_k: tuple[int, ...] = Field(
        (1, 5, 10, 100, 1000),
        description="Top-k values to report for the retrieval metric",
    )
    retrieval_metric_max_queries: int = Field(
        4096,
        ge=1,
        description="Maximum number of validation exemplars used when computing retrieval metrics",
    )
    retrieval_metric_per_building_limit: int = Field(
        4,
        ge=0,
        description="Cap on how many samples to draw per building when evaluating retrieval (0 disables the cap)",
    )

    # Logging
    log_file: Path | None = Field(None, description="Optional log file path")
    log_every_n_batches: int = Field(
        100, ge=1, description="Log metrics every N batches"
    )
    wandb_enabled: bool = Field(
        True,
        description="Enable logging to Weights & Biases (wandb)",
    )
    wandb_project: str | None = Field(
        None,
        description="wandb project name",
    )
    wandb_run_name: str | None = Field(
        None,
        description="Optional explicit wandb run name",
    )
    wandb_mode: Literal["online", "offline"] = Field(
        "online",
        description="wandb mode: online syncs immediately, offline defers sync",
    )

    # Best training configuration (for fetch_best_and_train.py)
    best_training_trial_name_pattern: str = Field(
        ".*-trial-.*",
        description="Regex pattern to match Optuna trial run names in WandB (for best training)",
    )
    best_training_metric_name: str = Field(
        "val/retrieval_recall@100",
        description="Metric name to use for selecting best run from WandB (for best training)",
    )



class DifficultyMetadataConfig(BaseSettings):
    """Configuration for building difficulty metadata generation."""

    model_config = SettingsConfigDict(
        env_prefix="DIFFICULTY_METADATA_",
        case_sensitive=False,
        extra="ignore",
    )

    merged_dir: Path = Field(
        ...,
        description="Directory containing train/val/test parquet outputs (shared pipeline location).",
    )
    difficulty_metadata_path: Path = Field(
        Path("difficulty_metadata.parquet"),
        description="Destination parquet file for aggregated difficulty metadata (shared pipeline location).",
    )
    neighbors: PositiveInt = Field(
        150,
        description="Number of neighbours to retain per anchor.",
    )
    k0_for_local_scale: PositiveInt = Field(
        50,
        description="Neighbour rank used to define the local scale L(a).",
    )
    sample_fraction_for_bands: float = Field(
        0.03,
        ge=0.0,
        le=1.0,
        description="Fraction of anchors sampled to calibrate band edges.",
    )
    distance_dtype: Literal["float32", "float64"] = Field(
        "float32",
        description="Floating point precision for stored neighbour distances (meters).",
    )
    batch_size: PositiveInt = Field(
        100_000,
        description="Number of anchors to process per BallTree query batch.",
    )
    row_group_size: PositiveInt = Field(
        50_000,
        description="Row group size for the output parquet writer.",
    )
    seed: int | None = Field(
        None, description="Random seed (inherits from global if not set)"
    )

    @model_validator(mode="after")
    def validate_parameters(self) -> "DifficultyMetadataConfig":
        if self.k0_for_local_scale >= self.neighbors:
            raise ValueError(
                "k0_for_local_scale must be less than the total neighbours requested."
            )
        return self


def load_config_from_file(
    config_path: Path,
    config_type: Literal[
        "process_tar",
        "merge_and_split",
        "generate_embeddings",
        "triplet_training",
        "difficulty_metadata",
    ],
) -> Union[
    ProcessTarConfig,
    MergeAndSplitConfig,
    GenerateEmbeddingsConfig,
    TripletTrainingConfig,
    DifficultyMetadataConfig,
]:
    """
    Load configuration from a unified TOML file organized by type.

    The config file should have type-based sections:
    - [global]: Shared settings (seed, log_dir)
    - [paths]: All pipeline file and directory paths
    - [embedding_model]: Model settings for embedding generation
    - [training_model]: Model architecture settings for training
    - [training]: Training hyperparameters and settings
    - [data]: Data processing settings (splits, neighbors, etc.)
    - [infrastructure]: Device, workers, and performance settings
    - [logging]: Logging and monitoring settings

    Args:
        config_path: Path to config file (must be .toml)
        config_type: Type of config to load

    Returns:
        Configuration object with values from relevant sections merged in

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config file format is not supported
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    if config_path.suffix.lower() != ".toml":
        raise ValueError(f"Config file must be .toml, got: {config_path.suffix}")

    with open(config_path, "rb") as f:
        data = tomllib.load(f)

    # Load global config
    global_data = data.get("global", {})
    global_config = GlobalConfig(**global_data)

    # Build merged data by flattening relevant sections
    merged_data = {}
    
    # Add global settings
    if global_config.seed is not None:
        merged_data["seed"] = global_config.seed
    # Always set log_file from mandatory log_dir
    merged_data["log_file"] = global_config.log_dir / f"{config_type}.log"

    # Merge sections based on config type
    # Handle batch_size conflicts by selectively merging fields
    if config_type == "process_tar":
        merged_data.update(data.get("paths", {}))
        if "tar_file" not in merged_data:
            merged_data["tar_file"] = "data/raw/placeholder.tar"
    elif config_type == "merge_and_split":
        merged_data.update(data.get("paths", {}))
        merged_data.update(data.get("data", {}))
    elif config_type == "generate_embeddings":
        merged_data.update(data.get("paths", {}))
        merged_data.update(data.get("embedding_model", {}))
    elif config_type == "triplet_training":
        merged_data.update(data.get("paths", {}))
        merged_data.update(data.get("training_model", {}))
        merged_data.update(data.get("training", {}))
        # Infrastructure: no fields to merge (device, num_workers, pin_memory are determined dynamically)
        merged_data.update(data.get("logging", {}))
    elif config_type == "difficulty_metadata":
        merged_data.update(data.get("paths", {}))
        merged_data.update(data.get("data", {}))
        # Infrastructure: merge all fields (including batch_size for difficulty_metadata)
        merged_data.update(data.get("infrastructure", {}))
    else:
        raise ValueError(f"Unknown config type: {config_type}")

    # Convert retrieval_metric_top_k to tuple if present
    if "retrieval_metric_top_k" in merged_data:
        merged_data["retrieval_metric_top_k"] = tuple(merged_data["retrieval_metric_top_k"])

    # Instantiate the appropriate config class
    config_classes = {
        "process_tar": ProcessTarConfig,
        "merge_and_split": MergeAndSplitConfig,
        "generate_embeddings": GenerateEmbeddingsConfig,
        "triplet_training": TripletTrainingConfig,
        "difficulty_metadata": DifficultyMetadataConfig,
    }
    return config_classes[config_type](**merged_data)
