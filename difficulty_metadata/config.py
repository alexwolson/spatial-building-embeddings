"""
Configuration management for difficulty metadata computation.
"""

from pathlib import Path
from typing import Literal

import tomllib
from pydantic import Field, PositiveInt, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DifficultyMetadataConfig(BaseSettings):
    """Configuration for building difficulty metadata generation."""

    model_config = SettingsConfigDict(
        env_prefix="DIFFICULTY_METADATA_",
        case_sensitive=False,
        extra="ignore",
    )

    input_parquet_path: Path = Field(
        ...,
        description="Directory or dataset path containing train/val/test parquet outputs.",
    )
    output_parquet_path: Path = Field(
        Path("difficulty_metadata.parquet"),
        description="Destination parquet file for aggregated difficulty metadata.",
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

    @model_validator(mode="after")
    def validate_parameters(self) -> "DifficultyMetadataConfig":
        if self.k0_for_local_scale >= self.neighbors:
            raise ValueError(
                "k0_for_local_scale must be less than the total neighbours requested."
            )
        return self


def load_config_from_file(
    config_path: Path,
    config_type: Literal["difficulty_metadata"],
) -> DifficultyMetadataConfig:
    """
    Load configuration from a TOML file.

    Args:
        config_path: Path to config file (must be .toml)
        config_type: Type of config to load (only 'difficulty_metadata' supported)

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

    if config_type == "difficulty_metadata":
        config_data = data.get("difficulty_metadata", data)
        return DifficultyMetadataConfig(**config_data)

    raise ValueError(f"Unknown config type: {config_type}")



