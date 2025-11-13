"""
Configuration management for triplet loss training.

Supports loading configuration from:
- Environment variables (with prefixes)
- TOML config files
- Command-line argument to specify config file path
"""

from pathlib import Path
from typing import Literal

import tomllib
from pydantic import Field, PositiveFloat, PositiveInt
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    output_embeddings_dir: Path | None = Field(
        None, description="Directory for saving final embeddings (optional)"
    )

    # Model architecture
    input_dim: int = Field(
        768, description="Input embedding dimension (DINOv2 base = 768)"
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
    samples_per_epoch: PositiveInt = Field(
        250_000,
        description="Maximum number of triplet samples drawn per epoch",
    )

    # UCB sampler configuration
    ucb_exploration_constant: float = Field(
        2.0, ge=0.0, description="UCB exploration constant (c)"
    )
    ucb_warmup_samples: int = Field(
        1000, ge=0, description="Number of warmup samples before UCB kicks in"
    )

    # Training configuration
    device: Literal["cuda", "cpu", "auto"] = Field(
        "auto", description="Device to use (auto detects GPU)"
    )
    num_workers: int = Field(4, ge=0, description="Number of data loader workers")
    pin_memory: bool = Field(True, description="Pin memory for data loader")
    seed: int = Field(42, description="Random seed for reproducibility")

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
        (1, 5, 10),
        description="Top-k values to report for the retrieval metric",
    )
    retrieval_metric_max_queries: int = Field(
        512,
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

    def model_post_init(self, __context):
        """Validate configuration after initialization."""
        if self.device == "auto":
            import torch

            self.device = "cuda" if torch.cuda.is_available() else "cpu"


def load_config_from_file(
    config_path: Path,
    config_type: Literal["triplet_training"],
) -> TripletTrainingConfig:
    """
    Load configuration from a TOML file.

    Args:
        config_path: Path to config file (must be .toml)
        config_type: Type of config to load

    Returns:
        Configuration object

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

    if config_type == "triplet_training":
        config_data = data.get("triplet_training", data)
        return TripletTrainingConfig(**config_data)

    raise ValueError(f"Unknown config type: {config_type}")
