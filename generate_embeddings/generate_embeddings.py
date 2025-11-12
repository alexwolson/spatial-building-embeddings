#!/usr/bin/env python3
"""
Generate image embeddings for a single intermediate parquet file using a pretrained DINOv2 model.

This script reads an intermediate parquet file, extracts the corresponding tar file,
loads images, generates embeddings using timm, and writes output parquet file with embeddings.
"""

import argparse
import logging
import math
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable, NamedTuple, Sequence

# Add project root to sys.path before imports
# This ensures imports work when script is run directly
# When Python runs a script, it adds the script's directory to sys.path[0],
# which can interfere with package imports even when PYTHONPATH is set
script_dir = Path(__file__).parent
project_root = script_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
)
import timm

from generate_embeddings.config import GenerateEmbeddingsConfig, load_config_from_file


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


def validate_gpu_available() -> None:
    """Validate that GPU is available, raise error if not."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "GPU is required but not available. "
            "Please ensure CUDA is available and a GPU is allocated."
        )
    device_count = torch.cuda.device_count()
    if device_count == 0:
        raise RuntimeError("CUDA is available but no GPU devices found.")
    logging.info(
        f"GPU available: {torch.cuda.get_device_name(0)} (device 0 of {device_count})"
    )


def load_model_and_transforms(model_name: str) -> tuple[nn.Module, nn.Module]:
    """
    Load pretrained model and get model-specific transforms.

    Args:
        model_name: Timm model name (e.g., 'vit_base_patch14_dinov2.lvd142m')

    Returns:
        Tuple of (model, transform)
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading model: {model_name}")

    # Validate GPU is available
    validate_gpu_available()
    device = torch.device("cuda")

    # Load model with num_classes=0 to remove classifier and get features directly
    model = timm.create_model(model_name, pretrained=True, num_classes=0)
    model.eval()
    model = model.to(device)

    # Get model-specific transforms using recommended approach
    data_config = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**data_config, is_training=False)
    logger.info(f"Model loaded and moved to {device}")
    logger.info(
        f"Transform config: input_size={data_config.get('input_size', 'unknown')}"
    )

    return model, transform


class ImageDataset(torch.utils.data.Dataset):
    """Dataset for loading images from paths."""

    def __init__(
        self, image_paths: list[Path], transform: nn.Module, extract_dir: Path
    ):
        self.image_paths = image_paths
        self.transform = transform
        self.extract_dir = extract_dir

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        rel_path = self.image_paths[idx]
        full_path = self.extract_dir / rel_path

        if not full_path.exists():
            raise FileNotFoundError(f"Image not found: {full_path}")

        image = Image.open(full_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image


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


def _format_duration(seconds: float) -> str:
    """Return human-readable duration string."""
    if math.isnan(seconds) or math.isinf(seconds):
        return "unknown"
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes:d}m {seconds:02d}s"
    return f"{seconds:d}s"


def generate_embeddings_batch(
    images: list[Path],
    model: nn.Module,
    transform: nn.Module,
    batch_size: int,
    extract_dir: Path,
    logger: logging.Logger,
) -> np.ndarray:
    """
    Generate embeddings for a batch of images.

    Args:
        images: List of relative image paths
        model: Pretrained model (on GPU)
        transform: Image transform pipeline
        batch_size: Batch size for processing
        extract_dir: Base directory where tar was extracted
        logger: Logger instance

    Returns:
        Numpy array of embeddings (shape: [num_images, embedding_dim])
    """
    device = next(model.parameters()).device

    # Create dataset and dataloader
    dataset = ImageDataset(images, transform, extract_dir)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    all_embeddings = []
    num_processed = 0
    total_images = len(images)
    start_time = time.time()
    last_log_time = start_time
    log_interval_seconds = 300  # 5 minutes

    # Log initial progress baseline
    logger.info(
        f"Progress: 0/{total_images:,} (0.0%) processed; ETA {_format_duration(float('nan'))}"
    )

    with torch.no_grad():
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            refresh_per_second=1,
        ) as progress:
            task = progress.add_task("Generating embeddings...", total=total_images)

            for batch_images in dataloader:
                # Move batch to GPU
                batch_images = batch_images.to(device, non_blocking=True)

                # Forward pass - with num_classes=0, model() returns (batch_size, num_features)
                # This is the recommended approach per timm documentation
                output = model(batch_images)
                if isinstance(output, tuple):
                    embeddings = output[0]
                else:
                    embeddings = output

                # Move to CPU and convert to numpy
                embeddings_np = embeddings.cpu().numpy().astype(np.float32)
                all_embeddings.append(embeddings_np)

                num_processed += len(batch_images)
                progress.update(task, advance=len(batch_images))

                now = time.time()
                pct_complete = (num_processed / total_images) * 100
                if (
                    num_processed == total_images
                    or now - last_log_time >= log_interval_seconds
                ):
                    elapsed = now - start_time
                    rate = num_processed / elapsed if elapsed > 0 else 0.0
                    remaining = (
                        (total_images - num_processed) / rate
                        if rate > 0
                        else float("nan")
                    )
                    logger.info(
                        "Progress: %s/%s (%.1f%%) processed; elapsed %s; ETA %s",
                        f"{num_processed:,}",
                        f"{total_images:,}",
                        pct_complete,
                        _format_duration(elapsed),
                        _format_duration(remaining),
                    )
                    last_log_time = now

    # Concatenate all embeddings
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    logger.info(
        f"Generated {len(all_embeddings)} embeddings of dimension {all_embeddings.shape[1]}"
    )
    total_elapsed = time.time() - start_time
    logger.info(
        "Finished embedding generation in %s (avg %.1f images/sec)",
        _format_duration(total_elapsed),
        len(all_embeddings) / total_elapsed if total_elapsed > 0 else 0.0,
    )

    return all_embeddings


def process_intermediate_file(
    parquet_path: Path,
    tar_path: Path,
    output_dir: Path,
    model_name: str,
    batch_size: int,
    temp_dir: Path | None = None,
    log_file: Path | None = None,
) -> dict[str, int]:
    """
    Process a single intermediate parquet file and generate embeddings.

    Args:
        parquet_path: Path to intermediate parquet file
        tar_path: Path to corresponding tar file
        output_dir: Directory for output embedding parquet files
        model_name: Timm model name
        batch_size: Batch size for inference
        temp_dir: Temporary directory for extraction (optional)
        log_file: Optional log file path

    Returns:
        Statistics dictionary
    """
    logger = setup_logging(log_file)
    logger.info("=" * 60)
    logger.info("Embedding generation: starting")
    logger.info("=" * 60)
    logger.info(f"Processing parquet file: {parquet_path}")
    logger.info(f"Tar file: {tar_path}")

    # Validate GPU is available
    validate_gpu_available()
    device = torch.device("cuda")

    # Statistics
    stats = {
        "total_entries": 0,
        "successful_embeddings": 0,
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

        # Debug: Check extraction structure
        extract_subdirs = [d for d in temp_extract_dir.iterdir() if d.is_dir()]
        if extract_subdirs:
            logger.info(
                f"Extracted structure: found {len(extract_subdirs)} top-level directories"
            )
            # Check if we need to adjust path resolution
            sample_dir = extract_subdirs[0]
            sample_files = list(sample_dir.rglob("*.jpg"))[:3]
            if sample_files:
                logger.info(
                    f"Sample extracted image path: {sample_files[0].relative_to(temp_extract_dir)}"
                )

        # Load model and transforms
        model, transform = load_model_and_transforms(model_name)

        # Get image paths from dataframe
        image_paths = df["image_path"].astype(str)
        if not image_paths.empty:
            logger.info(f"Sample parquet image path: {image_paths.iloc[0]}")

        path_transform = _infer_path_transform(
            image_paths.tolist(), temp_extract_dir, logger
        )

        valid_indices: list[int] = []
        valid_image_paths: list[Path] = []
        missing_example: Path | None = None
        invalid_example: str | None = None
        missing_file_count = 0
        invalid_path_count = 0

        for idx, raw_rel_path in enumerate(image_paths.tolist()):
            try:
                normalized_rel_path = _normalize_relative_path(raw_rel_path)
            except ValueError:
                invalid_path_count += 1
                if invalid_example is None:
                    invalid_example = raw_rel_path
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
                if missing_example is None:
                    missing_example = temp_extract_dir / candidate_paths[0]
                continue

            valid_indices.append(idx)
            valid_image_paths.append(matched_rel_path)

        stats["missing_images"] = missing_file_count + invalid_path_count

        if missing_file_count and missing_example is not None:
            logger.warning(
                "%s images listed in parquet were not found on disk; first missing example: %s",
                f"{missing_file_count:,}",
                missing_example,
            )

        if invalid_path_count and invalid_example is not None:
            logger.warning(
                "%s rows contained invalid image paths (first example: %s)",
                f"{invalid_path_count:,}",
                invalid_example,
            )

        if not valid_image_paths:
            logger.error("No valid images found")
            return stats

        logger.info(f"Processing {len(valid_image_paths):,} valid images")

        # Generate embeddings
        try:
            embeddings = generate_embeddings_batch(
                valid_image_paths,
                model,
                transform,
                batch_size,
                temp_extract_dir,
                logger,
            )
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}", exc_info=True)
            raise

        # Create output dataframe with embeddings
        # Only include rows with valid images
        df_output = df.iloc[valid_indices].copy()

        # Convert embeddings to list of lists for parquet storage
        embeddings_list = [emb.tolist() for emb in embeddings]
        df_output["embedding"] = embeddings_list

        stats["successful_embeddings"] = len(embeddings_list)
        stats["failed_images"] = stats["total_entries"] - stats["successful_embeddings"]

        # Write output parquet file
        output_dir.mkdir(parents=True, exist_ok=True)
        output_stem = parquet_path.stem
        output_file = output_dir / f"{output_stem}_embeddings.parquet"
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
        # Clean up temporary directory
        if should_cleanup and temp_extract_dir.exists():
            import shutil

            shutil.rmtree(temp_extract_dir)
            logger.info(f"Cleaned up temporary directory: {temp_extract_dir}")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate embeddings for a single intermediate parquet file"
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
            config = load_config_from_file(args.config, "generate_embeddings")
        else:
            # Load from environment variables
            config = GenerateEmbeddingsConfig()
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return 1

    # Re-setup logger with log file if specified
    if config.log_file:
        logger = setup_logging(config.log_file)

    # Validate inputs
    if not config.parquet_file.exists():
        logger.error(f"Parquet file not found: {config.parquet_file}")
        return 1

    if not config.tar_file.exists():
        logger.error(f"Tar file not found: {config.tar_file}")
        return 1

    if not config.tar_file.suffix == ".tar":
        logger.warning(f"File does not have .tar extension: {config.tar_file}")

    # Process file
    try:
        stats = process_intermediate_file(
            parquet_path=config.parquet_file,
            tar_path=config.tar_file,
            output_dir=config.output_dir,
            model_name=config.model_name,
            batch_size=config.batch_size,
            temp_dir=config.temp_dir,
            log_file=config.log_file,
        )

        # Print summary
        logger.info("")
        logger.info("Processing Summary:")
        logger.info(f"  Total entries: {stats['total_entries']:,}")
        logger.info(f"  Successful embeddings: {stats['successful_embeddings']:,}")
        logger.info(f"  Failed images: {stats['failed_images']:,}")
        if stats["missing_images"] > 0:
            logger.info(f"    - Missing images: {stats['missing_images']:,}")

        return 0 if stats["successful_embeddings"] > 0 else 1
    except Exception as e:
        logger.error(f"Error processing file: {e}", exc_info=True)
        if config.log_file:
            logger.info(f"See log file for details: {config.log_file}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
