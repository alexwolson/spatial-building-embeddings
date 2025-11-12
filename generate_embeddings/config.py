"""
Configuration management for embedding generation.

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


class GenerateEmbeddingsConfig(BaseSettings):
    """Configuration for embedding generation."""

    model_config = SettingsConfigDict(
        env_prefix="GENERATE_EMBEDDINGS_",
        case_sensitive=False,
        extra="ignore",
    )

    parquet_file: Path = Field(
        ..., description="Path to intermediate parquet file to process"
    )
    tar_file: Path | None = Field(
        None, description="Optional: tar file path (auto-detect if None)"
    )
    output_dir: Path = Field(
        ..., description="Output directory for embedding parquet files"
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
        if self.tar_file is None:
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

        # Try absolute path data/raw/ from project root
        # Walk up from parquet file to find project root (has pyproject.toml)
        current = self.parquet_file.parent
        for _ in range(5):  # Limit search depth
            if (current / "pyproject.toml").exists():
                tar_file_project = current / "data" / "raw" / f"{dataset_id:04d}.tar"
                if tar_file_project.exists():
                    return tar_file_project
            if current.parent == current:  # Reached root
                break
            current = current.parent

        raise FileNotFoundError(
            f"Could not find tar file for dataset ID {dataset_id:04d}. "
            f"Tried: {tar_file_same_dir}, {tar_file_raw}, and data/raw/{dataset_id:04d}.tar"
        )


def load_config_from_file(
    config_path: Path,
    config_type: Literal["generate_embeddings"],
) -> GenerateEmbeddingsConfig:
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
    if config_type == "generate_embeddings":
        config_data = data.get("generate_embeddings", data)
        return GenerateEmbeddingsConfig(**config_data)
    else:
        raise ValueError(f"Unknown config type: {config_type}")
