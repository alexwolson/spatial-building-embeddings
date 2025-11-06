#!/usr/bin/env python3
"""
Process a single tar archive containing building street view images and metadata.

This script extracts a tar file, parses metadata, validates image pairs,
and outputs an intermediate Parquet file for later merging and splitting.
"""

import argparse
import logging
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

import pandas as pd
from PIL import Image
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from preprocess_raw_data.config import ProcessTarConfig, load_config_from_file


class MetadataEntry(NamedTuple):
    """Parsed metadata entry from a .txt file."""

    target_id: int
    patch_id: int
    street_view_id: int
    target_lat: float
    target_lon: float
    dataset_id: int
    stem: str


def setup_logging(log_file: Path | None = None) -> logging.Logger:
    """Set up logging with Rich handler."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Remove existing handlers
    logger.handlers.clear()

    # Create Rich handler
    if log_file:
        # When writing to file, use file-aware console
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


def extract_dataset_id(tar_path: Path) -> int:
    """Extract dataset ID from tar filename (e.g., '0088.tar' -> 88)."""
    stem = tar_path.stem
    try:
        return int(stem)
    except ValueError as e:
        raise ValueError(f"Could not extract dataset ID from tar filename: {tar_path}") from e


def parse_metadata_file(txt_path: Path, dataset_id: int) -> MetadataEntry | None:
    """
    Parse a metadata .txt file and extract the 'd' line.

    Returns None if no 'd' line is found or if parsing fails.
    """
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("d "):
                    # Parse the 'd' line: 17 space-separated values
                    parts = line.split()
                    if len(parts) != 18:  # 'd' + 17 values
                        return None

                    try:
                        dataset_id_parsed = int(parts[1])
                        target_id = int(parts[2])
                        patch_id = int(parts[3])
                        street_view_id = int(parts[4])
                        target_lat = float(parts[5])
                        target_lon = float(parts[6])
                        # Remaining fields are discarded

                        # Use dataset_id from filename if parsed one doesn't match
                        if dataset_id_parsed != dataset_id:
                            # Log warning but use filename dataset_id
                            pass

                        stem = txt_path.stem

                        return MetadataEntry(
                            target_id=target_id,
                            patch_id=patch_id,
                            street_view_id=street_view_id,
                            target_lat=target_lat,
                            target_lon=target_lon,
                            dataset_id=dataset_id,
                            stem=stem,
                        )
                    except (ValueError, IndexError):
                        return None

        return None
    except Exception:
        return None


def find_image_file(base_path: Path, stem: str) -> Path | None:
    """Find image file with given stem, trying both .jpg and .jpeg extensions."""
    for ext in [".jpg", ".jpeg"]:
        img_path = base_path / f"{stem}{ext}"
        if img_path.exists():
            return img_path
    return None


def validate_image(image_path: Path) -> bool:
    """Validate that an image file exists and can be opened by PIL."""
    try:
        with Image.open(image_path) as img:
            img.load()  # Load the image to verify it's not corrupted
        return True
    except Exception:
        return False


def construct_relative_image_path(dataset_id: int, stem: str, image_path: Path) -> str:
    """Construct relative image path: {dataset_id:04d}/{dataset_id:04d}/{stem}.{ext}."""
    dataset_dir = f"{dataset_id:04d}"
    ext = image_path.suffix
    return f"{dataset_dir}/{dataset_dir}/{stem}{ext}"


def process_tar_file(
    tar_path: Path,
    output_dir: Path,
    temp_dir: Path | None = None,
    log_file: Path | None = None,
) -> dict[str, int]:
    """
    Process a single tar file and output intermediate Parquet file.

    Returns statistics dictionary.
    """
    logger = setup_logging(log_file)
    logger.info(f"Processing tar file: {tar_path}")

    # Extract dataset ID from filename
    try:
        dataset_id = extract_dataset_id(tar_path)
    except ValueError as e:
        logger.error(f"Failed to extract dataset ID: {e}")
        return {
            "total_metadata_files": 0,
            "valid_entries": 0,
            "invalid_entries": 0,
            "missing_images": 0,
            "corrupted_images": 0,
            "orphan_metadata": 0,
        }

    # Create temporary directory for extraction
    temp_extract_dir = temp_dir or Path(tempfile.mkdtemp())
    should_cleanup = temp_dir is None

    try:
        # Extract tar file
        import tarfile

        logger.info(f"Extracting tar file to: {temp_extract_dir}")
        try:
            with tarfile.open(tar_path, "r") as tar:
                tar.extractall(temp_extract_dir, filter="data")
        except Exception as e:
            logger.error(f"Failed to extract tar file: {e}")
            return {
                "total_metadata_files": 0,
                "valid_entries": 0,
                "invalid_entries": 0,
                "missing_images": 0,
                "corrupted_images": 0,
                "orphan_metadata": 0,
            }

        # Find all .txt metadata files
        txt_files = list(temp_extract_dir.rglob("*.txt"))
        logger.info(f"Found {len(txt_files)} metadata files")

        # Statistics
        stats = {
            "total_metadata_files": len(txt_files),
            "valid_entries": 0,
            "invalid_entries": 0,
            "missing_images": 0,
            "corrupted_images": 0,
            "orphan_metadata": 0,
        }

        # Process metadata files
        entries: list[dict[str, int | float | str]] = []

        # Use Rich progress bar with log-file-friendly updates
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            refresh_per_second=1,  # Update once per second to minimize log file bloat
        ) as progress:
            task = progress.add_task("Processing metadata files...", total=len(txt_files))

            for txt_file in txt_files:
                # Parse metadata
                metadata = parse_metadata_file(txt_file, dataset_id)
                if metadata is None:
                    stats["orphan_metadata"] += 1
                    stats["invalid_entries"] += 1
                    progress.update(task, advance=1)
                    continue

                # Find corresponding image file
                image_path = find_image_file(txt_file.parent, metadata.stem)
                if image_path is None:
                    stats["missing_images"] += 1
                    stats["invalid_entries"] += 1
                    progress.update(task, advance=1)
                    continue

                # Validate image
                if not validate_image(image_path):
                    stats["corrupted_images"] += 1
                    stats["invalid_entries"] += 1
                    progress.update(task, advance=1)
                    continue

                # Construct relative image path
                rel_image_path = construct_relative_image_path(
                    dataset_id, metadata.stem, image_path
                )

                # Add valid entry
                entries.append(
                    {
                        "target_id": metadata.target_id,
                        "patch_id": metadata.patch_id,
                        "street_view_id": metadata.street_view_id,
                        "target_lat": metadata.target_lat,
                        "target_lon": metadata.target_lon,
                        "image_path": rel_image_path,
                        "tar_file": tar_path.name,
                    }
                )
                stats["valid_entries"] += 1
                progress.update(task, advance=1)

        logger.info(f"Processed {len(txt_files)} metadata files")
        logger.info(f"Valid entries: {stats['valid_entries']}")
        logger.info(f"Invalid entries: {stats['invalid_entries']}")
        logger.info(f"  - Missing images: {stats['missing_images']}")
        logger.info(f"  - Corrupted images: {stats['corrupted_images']}")
        logger.info(f"  - Orphan metadata: {stats['orphan_metadata']}")

        # Create DataFrame
        if not entries:
            logger.warning("No valid entries found, skipping Parquet file creation")
            return stats

        df = pd.DataFrame(entries)
        # Ensure correct dtypes (pandas should infer correctly, but be explicit for Parquet)
        df = df.astype({
            "target_id": "int64",
            "patch_id": "int64",
            "street_view_id": "int64",
            "target_lat": "float64",
            "target_lon": "float64",
        })

        # Write Parquet file
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{tar_path.stem}.parquet"
        logger.info(f"Writing Parquet file: {output_file}")

        df.to_parquet(
            output_file,
            engine="pyarrow",
            compression="snappy",
            index=False,
        )

        logger.info(f"Successfully wrote {len(df)} entries to {output_file}")
        return stats

    finally:
        # Clean up temporary directory
        if should_cleanup and temp_extract_dir.exists():
            import shutil

            shutil.rmtree(temp_extract_dir)
            logger.info(f"Cleaned up temporary directory: {temp_extract_dir}")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Process a single tar archive and output intermediate Parquet file"
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to TOML config file. If not provided, config is loaded from environment variables.",
    )

    args = parser.parse_args()

    # Set up logger early (before config loading to catch config errors)
    logger = setup_logging(None)

    # Load configuration
    try:
        if args.config:
            config = load_config_from_file(args.config, "process_tar")
        else:
            # Load from environment variables
            config = ProcessTarConfig()
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return 1

    # Re-setup logger with log file if specified
    if config.log_file:
        logger = setup_logging(config.log_file)

    # Validate inputs
    if not config.tar_file.exists():
        logger.error(f"Tar file not found: {config.tar_file}")
        return 1

    if not config.tar_file.suffix == ".tar":
        logger.warning(f"File does not have .tar extension: {config.tar_file}")

    # Process tar file
    stats = process_tar_file(
        tar_path=config.tar_file,
        output_dir=config.output_dir,
        temp_dir=config.temp_dir,
        log_file=config.log_file,
    )

    # Print summary
    logger.info("")
    logger.info("Processing Summary:")
    logger.info(f"  Total metadata files: {stats['total_metadata_files']}")
    logger.info(f"  Valid entries: {stats['valid_entries']}")
    logger.info(f"  Invalid entries: {stats['invalid_entries']}")
    if stats["invalid_entries"] > 0:
        logger.info(f"    - Missing images: {stats['missing_images']}")
        logger.info(f"    - Corrupted images: {stats['corrupted_images']}")
        logger.info(f"    - Orphan metadata: {stats['orphan_metadata']}")

    return 0 if stats["valid_entries"] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())

