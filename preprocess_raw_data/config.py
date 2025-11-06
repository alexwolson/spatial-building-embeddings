"""
Configuration management using Pydantic Settings.

Supports loading configuration from:
- Environment variables (with prefixes)
- TOML config files
- Command-line argument to specify config file path
"""

from pathlib import Path
from typing import Literal
import tomllib

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ProcessTarConfig(BaseSettings):
    """Configuration for Phase 1: processing a single tar file."""

    model_config = SettingsConfigDict(
        env_prefix="PROCESS_TAR_",
        case_sensitive=False,
        extra="ignore",
    )

    tar_file: Path = Field(..., description="Path to tar file to process")
    output_dir: Path = Field(..., description="Output directory for intermediate Parquet files")
    log_file: Path | None = Field(None, description="Optional log file path")
    temp_dir: Path | None = Field(None, description="Optional temporary directory for extraction")


class MergeAndSplitConfig(BaseSettings):
    """Configuration for Phase 2: merging and splitting intermediate files."""

    model_config = SettingsConfigDict(
        env_prefix="MERGE_",
        case_sensitive=False,
        extra="ignore",
    )

    intermediates_dir: Path = Field(..., description="Directory containing intermediate Parquet files")
    output_dir: Path = Field(..., description="Directory for final output Parquet files")
    train_ratio: float = Field(0.7, ge=0.0, le=1.0, description="Training set ratio")
    val_ratio: float = Field(0.15, ge=0.0, le=1.0, description="Validation set ratio")
    test_ratio: float = Field(0.15, ge=0.0, le=1.0, description="Test set ratio")
    seed: int = Field(42, description="Random seed for deterministic splits")
    log_file: Path | None = Field(None, description="Optional log file path")

    def model_post_init(self, __context):
        """Validate that ratios sum to 1.0."""
        ratio_sum = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(ratio_sum - 1.0) > 1e-6:
            raise ValueError(f"Ratios must sum to 1.0, got {ratio_sum}")


def load_config_from_file(
    config_path: Path,
    config_type: Literal["process_tar", "merge_and_split"],
) -> ProcessTarConfig | MergeAndSplitConfig:
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

    # Extract the relevant section from the config file
    if config_type == "process_tar":
        config_data = data.get("process_tar", data)
        return ProcessTarConfig(**config_data)
    elif config_type == "merge_and_split":
        config_data = data.get("merge_and_split", data)
        return MergeAndSplitConfig(**config_data)
    else:
        raise ValueError(f"Unknown config type: {config_type}")

