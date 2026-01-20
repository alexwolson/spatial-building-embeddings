#!/usr/bin/env python3
"""
Compute pixel fingerprints for images in an intermediate parquet file.

This script reads an intermediate parquet file, extracts the corresponding tar file,
loads images, resizes them to a small fixed size (e.g., 16x16), and writes output
parquet file with fingerprints.
"""

import argparse
import logging
import sys
import tempfile
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from PIL import Image
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)

from config import ComputeFingerprintsConfig, load_config_from_file


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


def _normalize_relative_path(value: str | Path) -> Path:
    value_str = str(value).strip()
    if not value_str:
        raise ValueError("Empty image path")

    value_str = value_str.replace("\\", "/")
    parts = [part for part in value_str.split("/") if part and part != "."]
    if not parts:
        raise ValueError(f"Invalid relative path: {value}")
    return Path(*parts)


def _drop_duplicate_second_segment(rel_path: Path) -> Path:
    parts = rel_path.parts
    if len(parts) >= 2 and parts[0].lower() == parts[1].lower():
        return Path(parts[0], *parts[2:])
    return rel_path


def _infer_path_transform(
    image_paths: Sequence[str], extract_dir: Path, logger: logging.Logger
) -> Callable[[Path], Path]:
    """Determine how parquet relative paths map to extracted files."""

    samples: list[Path] = []
    for raw in image_paths:
        try:
            samples.append(_normalize_relative_path(raw))
        except ValueError:
            continue
        if len(samples) >= 100:
            break

    if not samples:
        return lambda rel: rel

    def identity(rel: Path) -> Path:
        return rel

    candidates: list[tuple[str, Callable[[Path], Path]]] = [
        ("direct", identity),
        ("drop_duplicate_second", _drop_duplicate_second_segment),
    ]

    for name, transform in candidates:
        for sample in samples:
            candidate_path = extract_dir / transform(sample)
            if candidate_path.exists():
                if name != "direct":
                    logger.info(
                        "Detected '%s' path transform for extracted images", name
                    )
                return transform

    logger.warning(
        "Unable to infer path transform from samples; defaulting to direct paths"
    )
    return identity


def compute_fingerprints_batch(
    images: list[Path],
    extract_dir: Path,
    image_size: int,
    logger: logging.Logger,
) -> np.ndarray:
    """
    Compute fingerprints for a batch of images.

    Args:
        images: List of relative image paths
        extract_dir: Base directory where tar was extracted
        image_size: Size to resize images to (image_size x image_size)
        logger: Logger instance

    Returns:
        Numpy array of fingerprints (shape: [num_images, image_size * image_size * 3])
    """
    fingerprints = []
    num_processed = 0
    total_images = len(images)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        refresh_per_second=1,
    ) as progress:
        task = progress.add_task("Computing fingerprints...", total=total_images)

        for rel_path in images:
            full_path = extract_dir / rel_path
            try:
                with Image.open(full_path) as img:
                    img = img.convert("RGB")
                    img = img.resize(
                        (image_size, image_size), Image.Resampling.BILINEAR
                    )
                    # Flatten to 1D array
                    fingerprint = np.array(img, dtype=np.uint8).flatten()
                    fingerprints.append(fingerprint)
            except Exception as e:
                # Should not happen as paths are pre-validated, but safe guard
                logger.warning(f"Failed to process image {full_path}: {e}")
                # Append zero vector as fallback
                fingerprints.append(
                    np.zeros(image_size * image_size * 3, dtype=np.uint8)
                )

            num_processed += 1
            progress.update(task, advance=1)

    return np.stack(fingerprints)


def process_intermediate_file(
    parquet_path: Path,
    tar_path: Path,
    output_dir: Path,
    image_size: int,
    temp_dir: Path | None = None,
    log_file: Path | None = None,
) -> dict[str, int]:
    """
    Process a single intermediate parquet file and generate fingerprints.

    Args:
        parquet_path: Path to intermediate parquet file
        tar_path: Path to corresponding tar file
        output_dir: Directory for output fingerprint parquet files
        image_size: Size of fingerprint images
        temp_dir: Temporary directory for extraction (optional)
        log_file: Optional log file path

    Returns:
        Statistics dictionary
    """
    logger = setup_logging(log_file)
    logger.info("=" * 60)
    logger.info("Fingerprint generation: starting")
    logger.info("=" * 60)
    logger.info(f"Processing parquet file: {parquet_path}")
    logger.info(f"Tar file: {tar_path}")

    # Statistics
    stats = {
        "total_entries": 0,
        "successful_fingerprints": 0,
        "failed_images": 0,
        "missing_images": 0,
    }

    # Create temporary directory for extraction
    temp_extract_dir = temp_dir or Path(tempfile.mkdtemp())
    should_cleanup = temp_dir is None

    try:
        # Read intermediate parquet file
        logger.info("Reading intermediate parquet file...")
        df = pd.read_parquet(parquet_path, engine="pyarrow").copy()
        stats["total_entries"] = len(df)
        logger.info(f"Found {stats['total_entries']:,} entries")

        # Extract tar file
        import tarfile

        logger.info(f"Extracting tar file to: {temp_extract_dir}")
        try:
            with tarfile.open(tar_path, "r") as tar:
                tar.extractall(temp_extract_dir, filter="data")
        except Exception as e:
            logger.error(f"Failed to extract tar file: {e}")
            raise

        # Determine path transform
        image_paths = df["image_path"].astype(str)
        path_transform = _infer_path_transform(
            image_paths.tolist(), temp_extract_dir, logger
        )

        valid_indices: list[int] = []
        valid_image_paths: list[Path] = []
        missing_file_count = 0
        invalid_path_count = 0

        for idx, raw_rel_path in enumerate(image_paths.tolist()):
            try:
                normalized_rel_path = _normalize_relative_path(raw_rel_path)
            except ValueError:
                invalid_path_count += 1
                continue

            transformed_rel_path = path_transform(normalized_rel_path)
            candidate_paths: list[Path] = [transformed_rel_path]
            if transformed_rel_path != normalized_rel_path:
                candidate_paths.append(normalized_rel_path)

            matched_rel_path: Path | None = None
            for candidate_rel in candidate_paths:
                if (temp_extract_dir / candidate_rel).exists():
                    matched_rel_path = candidate_rel
                    break

            if matched_rel_path is None:
                missing_file_count += 1
                continue

            valid_indices.append(idx)
            valid_image_paths.append(matched_rel_path)

        stats["missing_images"] = missing_file_count + invalid_path_count

        if missing_file_count:
            logger.warning(
                "%s images listed in parquet were not found on disk",
                f"{missing_file_count:,}",
            )

        if not valid_image_paths:
            logger.error("No valid images found")
            return stats

        logger.info(f"Processing {len(valid_image_paths):,} valid images")

        # Generate fingerprints
        try:
            fingerprints = compute_fingerprints_batch(
                valid_image_paths,
                temp_extract_dir,
                image_size,
                logger,
            )
        except Exception as e:
            logger.error(f"Failed to compute fingerprints: {e}", exc_info=True)
            raise

        # Create output dataframe with fingerprints
        df_output = df.iloc[valid_indices].copy()

        # Add identifiers if missing (same logic as merge_and_split)
        dataset_str = df_output["dataset_id"].astype(int).astype(str).str.zfill(4)
        target_str = df_output["target_id"].astype(int).astype(str)
        patch_str = df_output["patch_id"].astype(int).astype(str)

        if "building_id" not in df_output.columns:
            df_output["building_id"] = dataset_str + "_" + target_str
        if "streetview_image_id" not in df_output.columns:
            df_output["streetview_image_id"] = dataset_str + "_" + patch_str

        fingerprints_list = [fp.tolist() for fp in fingerprints]
        df_output["fingerprint"] = fingerprints_list

        stats["successful_fingerprints"] = len(fingerprints_list)
        stats["failed_images"] = (
            stats["total_entries"] - stats["successful_fingerprints"]
        )

        # Write output parquet file
        output_dir.mkdir(parents=True, exist_ok=True)
        output_stem = parquet_path.stem
        output_file = output_dir / f"{output_stem}_fingerprints.parquet"
        logger.info(f"Writing output parquet file: {output_file}")

        df_output.to_parquet(
            output_file,
            engine="pyarrow",
            compression="snappy",
            index=False,
        )

        file_size_mb = output_file.stat().st_size / (1024 * 1024)
        logger.info(
            f"Successfully wrote {len(df_output):,} entries to {output_file} ({file_size_mb:.1f} MB)"
        )

        return stats

    finally:
        if should_cleanup and temp_extract_dir.exists():
            import shutil

            shutil.rmtree(temp_extract_dir)
            logger.info(f"Cleaned up temporary directory: {temp_extract_dir}")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compute pixel fingerprints for a single intermediate parquet file"
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to TOML config file. If not provided, config is loaded from environment variables.",
    )
    # Per-job arguments (override config if provided)
    parser.add_argument(
        "--parquet-file",
        type=Path,
        help="Path to intermediate parquet file to process (overrides config)",
    )
    parser.add_argument(
        "--temp-dir",
        type=Path,
        help="Temporary directory for extraction (overrides config)",
    )

    args = parser.parse_args()
    logger = setup_logging(None)

    try:
        if args.config:
            config = load_config_from_file(args.config, "compute_fingerprints")
        else:
            config = ComputeFingerprintsConfig()
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return 1

    # Override with command-line arguments if provided
    if args.parquet_file is not None:
        config.parquet_file = args.parquet_file
        try:
            config.tar_file = config._auto_detect_tar_file()
        except Exception as e:
            logger.error(
                f"Failed to auto-detect tar file for {config.parquet_file}: {e}"
            )
            return 1

    if args.temp_dir is not None:
        config.temp_dir = args.temp_dir

    if config.parquet_file is None:
        logger.error(
            "Parquet file must be provided via config or command line arguments."
        )
        return 1

    if config.log_file:
        logger = setup_logging(config.log_file)

    if not config.parquet_file.exists():
        logger.error(f"Parquet file not found: {config.parquet_file}")
        return 1

    if not config.tar_file.exists():
        logger.error(f"Tar file not found: {config.tar_file}")
        return 1

    try:
        stats = process_intermediate_file(
            parquet_path=config.parquet_file,
            tar_path=config.tar_file,
            output_dir=config.fingerprints_dir,
            image_size=config.image_size,
            temp_dir=config.temp_dir,
            log_file=config.log_file,
        )

        logger.info("")
        logger.info("Processing Summary:")
        logger.info(f"  Total entries: {stats['total_entries']:,}")
        logger.info(f"  Successful fingerprints: {stats['successful_fingerprints']:,}")
        logger.info(f"  Failed images: {stats['failed_images']:,}")
        if stats["missing_images"] > 0:
            logger.info(f"    - Missing images: {stats['missing_images']:,}")

        return 0 if stats["successful_fingerprints"] > 0 else 1
    except Exception as e:
        logger.error(f"Error processing file: {e}", exc_info=True)
        if config.log_file:
            logger.info(f"See log file for details: {config.log_file}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
