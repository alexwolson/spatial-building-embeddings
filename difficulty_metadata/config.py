"""
Configuration management for difficulty metadata precomputation.

This mirrors the configuration style used in other project modules, supporting
environment variables, TOML files, and CLI overrides.
"""

from pathlib import Path
from typing import Literal
import json
import tomllib

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DifficultyConfig(BaseSettings):
    """Configuration for difficulty metadata computation."""

    model_config = SettingsConfigDict(
        env_prefix="DIFFICULTY_",
        case_sensitive=False,
        extra="ignore",
    )

    source_parquet: Path = Field(
        ...,
        description="Input parquet containing per-target metadata after Phase 2 merge.",
    )
    output_dir: Path = Field(
        Path("data/difficulty"),
        description="Directory for writing difficulty metadata artifacts.",
    )
    k_neighbors: int = Field(
        256,
        ge=1,
        description="Number of negative neighbors to retain per target.",
    )
    bands: int = Field(
        5,
        ge=1,
        description="Number of global difficulty bands (quantile based).",
    )
    overwrite: bool = Field(
        False,
        description="Overwrite existing outputs when true.",
    )
    n_jobs: int = Field(
        -1,
        description="Parallel jobs for sklearn NearestNeighbors (-1 => all cores).",
    )
    log_file: Path | None = Field(
        None,
        description="Optional path to a log file (stdout used when omitted).",
    )

    neighbors_filename: str = Field(
        "neighbors.parquet",
        description="Filename for per-target neighbor cache (relative to output_dir).",
    )
    index_map_filename: str = Field(
        "index_map.parquet",
        description="Filename for the anchor index map (relative to output_dir).",
    )

    def model_post_init(self, __context) -> None:
        """Normalise paths after validation."""
        self.output_dir = self.output_dir.expanduser().resolve()
        self.source_parquet = self.source_parquet.expanduser().resolve()
        if self.log_file is not None:
            self.log_file = self.log_file.expanduser().resolve()

    def to_json(self) -> str:
        """Return a JSON string representing the config (with paths as strings)."""
        return json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=True)


def load_config_from_file(
    config_path: Path,
    config_type: Literal["difficulty_metadata"],
) -> DifficultyConfig:
    """Load configuration from a TOML file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    if config_path.suffix.lower() != ".toml":
        raise ValueError(f"Config file must be .toml, got: {config_path.suffix}")

    with config_path.open("rb") as f:
        data = tomllib.load(f)

    if config_type != "difficulty_metadata":
        raise ValueError(f"Unknown config type: {config_type}")

    config_data = data.get("difficulty_metadata", data)
    return DifficultyConfig(**config_data)

