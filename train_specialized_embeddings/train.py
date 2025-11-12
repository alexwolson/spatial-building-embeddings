#!/usr/bin/env python3
"""
Train specialized embeddings using triplet loss with UCB-guided negative sampling.

This script loads precomputed embeddings, forms triplets (anchor/positive/negative),
and trains a projection head to learn specialized embeddings optimized for building
identification tasks.
"""

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from torch.utils.data import DataLoader

from train_specialized_embeddings.config import TripletTrainingConfig, load_config_from_file
from train_specialized_embeddings.datasets import TripletDataset
from train_specialized_embeddings.model import EmbeddingProjector, TripletLossWrapper


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


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data(config: TripletTrainingConfig, logger: logging.Logger) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load training, validation, and difficulty metadata dataframes."""
    logger.info("Loading data files...")

    # Load training data
    logger.info(f"Loading training data from: {config.train_parquet_path}")
    train_df = pd.read_parquet(config.train_parquet_path, engine="pyarrow")
    logger.info(f"Loaded {len(train_df):,} training samples")

    # Load validation data
    logger.info(f"Loading validation data from: {config.val_parquet_path}")
    val_df = pd.read_parquet(config.val_parquet_path, engine="pyarrow")
    logger.info(f"Loaded {len(val_df):,} validation samples")

    # Load difficulty metadata
    logger.info(f"Loading difficulty metadata from: {config.difficulty_metadata_path}")
    difficulty_df = pd.read_parquet(config.difficulty_metadata_path, engine="pyarrow")
    logger.info(f"Loaded {len(difficulty_df):,} difficulty metadata entries")

    # Validate required columns
    required_train_cols = {"building_id", "embedding"}
    missing = required_train_cols - set(train_df.columns)
    if missing:
        raise ValueError(f"Missing required columns in training data: {missing}")

    required_val_cols = {"building_id", "embedding"}
    missing = required_val_cols - set(val_df.columns)
    if missing:
        raise ValueError(f"Missing required columns in validation data: {missing}")

    return train_df, val_df, difficulty_df


def collate_triplets(batch: list) -> dict[str, torch.Tensor]:
    """Collate function for triplet batches."""
    anchors = torch.stack([sample.anchor_embedding for sample in batch])
    positives = torch.stack([sample.positive_embedding for sample in batch])
    negatives = torch.stack([sample.negative_embedding for sample in batch])
    bands = torch.tensor([sample.difficulty_band for sample in batch], dtype=torch.long)

    return {
        "anchor": anchors,
        "positive": positives,
        "negative": negatives,
        "band": bands,
    }


def validate(
    model: nn.Module,
    val_dataset: TripletDataset,
    loss_fn: TripletLossWrapper,
    device: torch.device,
    logger: logging.Logger,
    num_samples: int = 1000,
) -> dict[str, float]:
    """
    Run validation on validation dataset.

    Args:
        model: Model to validate
        loss_fn: Loss function
        device: Device to run on
        logger: Logger instance
        num_samples: Number of samples to validate on

    Returns:
        Dictionary of validation metrics
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for _ in range(min(num_samples, len(val_dataset))):
            # Sample random index from validation dataset
            sample_idx = np.random.randint(0, len(val_dataset))
            sample = val_dataset[sample_idx]

            anchor = sample.anchor_embedding.unsqueeze(0).to(device)
            positive = sample.positive_embedding.unsqueeze(0).to(device)
            negative = sample.negative_embedding.unsqueeze(0).to(device)

            # Project embeddings
            anchor_proj = model(anchor)
            positive_proj = model(positive)
            negative_proj = model(negative)

            # Compute loss
            loss = loss_fn(anchor_proj, positive_proj, negative_proj)
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else float("inf")

    metrics = {
        "val_loss": avg_loss,
    }

    logger.info(f"Validation loss: {avg_loss:.6f}")

    return metrics


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
    checkpoint_path: Path,
    logger: logging.Logger,
):
    """Save training checkpoint."""
    checkpoint_dir = checkpoint_path.parent
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }

    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved checkpoint to: {checkpoint_path}")


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    logger: logging.Logger,
) -> int:
    """Load training checkpoint."""
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]

    logger.info(f"Loaded checkpoint from epoch {epoch}")
    if "metrics" in checkpoint:
        logger.info(f"Checkpoint metrics: {checkpoint['metrics']}")

    return epoch


def save_embeddings(
    model: nn.Module,
    embeddings_df: pd.DataFrame,
    output_dir: Path,
    device: torch.device,
    logger: logging.Logger,
    batch_size: int = 1000,
):
    """
    Save projected embeddings to parquet file.

    Args:
        model: Trained model
        embeddings_df: DataFrame with original embeddings
        output_dir: Output directory
        device: Device to run on
        logger: Logger instance
        batch_size: Batch size for processing
    """
    logger.info("Computing and saving projected embeddings...")
    model.eval()

    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert embeddings to numpy
    embeddings_array = np.array([np.array(emb, dtype=np.float32) for emb in embeddings_df["embedding"]])
    projected_embeddings = []

    with torch.no_grad():
        for i in range(0, len(embeddings_array), batch_size):
            batch_embeddings = embeddings_array[i : i + batch_size]
            batch_tensor = torch.from_numpy(batch_embeddings).to(device)
            batch_projected = model(batch_tensor)
            projected_embeddings.append(batch_projected.cpu().numpy())

    projected_embeddings = np.concatenate(projected_embeddings, axis=0)

    # Create output dataframe (retain original embeddings for downstream analysis)
    output_df = embeddings_df.copy()
    output_df["specialized_embedding"] = [emb.tolist() for emb in projected_embeddings]

    # Save to parquet
    output_file = output_dir / "specialized_embeddings.parquet"
    output_df.to_parquet(output_file, engine="pyarrow", compression="snappy", index=False)

    logger.info(f"Saved {len(output_df):,} projected embeddings to: {output_file}")


def train(config: TripletTrainingConfig) -> int:
    """Main training function."""
    logger = setup_logging(config.log_file)
    logger.info("=" * 60)
    logger.info("Triplet Loss Training: Starting")
    logger.info("=" * 60)

    # Set seed
    set_seed(config.seed)
    logger.info(f"Random seed: {config.seed}")

    # Determine device
    if config.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config.device)

    logger.info(f"Using device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load data
    train_df, val_df, difficulty_df = load_data(config, logger)

    # Create datasets
    logger.info("Creating datasets...")
    train_dataset = TripletDataset(train_df, difficulty_df, config)
    val_dataset = TripletDataset(val_df, difficulty_df, config)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=collate_triplets,
    )

    # Initialize model
    logger.info("Initializing model...")
    model = EmbeddingProjector(
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        output_dim=config.output_dim,
        dropout=config.dropout,
        use_residual=config.use_residual,
        use_layer_norm=config.use_layer_norm,
    ).to(device)

    # Initialize loss function
    loss_fn = TripletLossWrapper(margin=config.margin, distance=config.loss_distance)

    # Initialize optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # Resume from checkpoint if specified
    start_epoch = 0
    if config.resume_from_checkpoint:
        start_epoch = load_checkpoint(config.resume_from_checkpoint, model, optimizer, logger)
        start_epoch += 1  # Start from next epoch

    # Training loop
    logger.info("Starting training...")
    logger.info(f"Total epochs: {config.num_epochs}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Learning rate: {config.learning_rate}")

    best_val_loss = float("inf")
    global_step = 0
    val_metrics = {"val_loss": float("inf")}  # Initialize with default value

    for epoch in range(start_epoch, config.num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        epoch_start_time = time.time()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            refresh_per_second=1,
        ) as progress:
            task = progress.add_task(f"Epoch {epoch+1}/{config.num_epochs}", total=len(train_loader))

            for batch_idx, batch in enumerate(train_loader):
                anchor = batch["anchor"].to(device)
                positive = batch["positive"].to(device)
                negative = batch["negative"].to(device)
                bands = batch["band"].to(device)

                # Forward pass
                anchor_proj = model(anchor)
                positive_proj = model(positive)
                negative_proj = model(negative)

                # Compute loss
                loss = loss_fn(anchor_proj, positive_proj, negative_proj)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1
                global_step += 1

                # Update UCB rewards (use negative loss as reward) for valid bands only
                with torch.no_grad():
                    for i, band in enumerate(bands):
                        band_value = int(band.item())
                        if band_value < 0:
                            continue
                        reward = loss_fn.compute_reward(
                            anchor_proj[i : i + 1],
                            positive_proj[i : i + 1],
                            negative_proj[i : i + 1],
                        )
                        train_dataset.update_ucb_reward(band_value, reward)

                # Logging
                if global_step % config.log_every_n_batches == 0:
                    avg_loss = epoch_loss / num_batches
                    logger.info(
                        f"Epoch {epoch+1}, Step {global_step}, Loss: {avg_loss:.6f}"
                    )

                progress.update(task, advance=1)

        epoch_elapsed = time.time() - epoch_start_time
        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else float("inf")

        logger.info(
            f"Epoch {epoch+1}/{config.num_epochs} completed in {epoch_elapsed:.2f}s, "
            f"Average loss: {avg_epoch_loss:.6f}"
        )

        # Validation
        if (epoch + 1) % config.validate_every_n_epochs == 0:
            val_metrics = validate(model, val_dataset, loss_fn, device, logger)
            val_loss = val_metrics["val_loss"]

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                logger.info(f"New best validation loss: {best_val_loss:.6f}")

        # Save checkpoint
        if (epoch + 1) % config.save_every_n_epochs == 0:
            checkpoint_path = config.checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt"
            metrics = {"train_loss": avg_epoch_loss, "val_loss": val_metrics.get("val_loss", float("inf"))}
            save_checkpoint(model, optimizer, epoch, metrics, checkpoint_path, logger)

        # Log UCB statistics
        ucb_stats = train_dataset.ucb_sampler.get_statistics()
        logger.info(f"UCB Statistics: {ucb_stats}")

    # Save final checkpoint
    final_checkpoint_path = config.checkpoint_dir / "checkpoint_final.pt"
    metrics = {"train_loss": avg_epoch_loss, "val_loss": val_metrics.get("val_loss", float("inf"))}
    save_checkpoint(model, optimizer, config.num_epochs - 1, metrics, final_checkpoint_path, logger)

    # Save embeddings if requested
    if config.output_embeddings_dir:
        logger.info("Saving final embeddings...")
        # Save for both train and val splits
        save_embeddings(model, train_df, config.output_embeddings_dir / "train", device, logger)
        save_embeddings(model, val_df, config.output_embeddings_dir / "val", device, logger)

    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info("=" * 60)

    return 0


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train specialized embeddings using triplet loss"
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
            config = load_config_from_file(args.config, "triplet_training")
        else:
            # Load from environment variables
            config = TripletTrainingConfig()
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return 1

    # Re-setup logger with log file if specified
    if config.log_file:
        logger = setup_logging(config.log_file)

    # Validate inputs
    if not config.train_parquet_path.exists():
        logger.error(f"Training parquet file not found: {config.train_parquet_path}")
        return 1

    if not config.val_parquet_path.exists():
        logger.error(f"Validation parquet file not found: {config.val_parquet_path}")
        return 1

    if not config.difficulty_metadata_path.exists():
        logger.error(f"Difficulty metadata file not found: {config.difficulty_metadata_path}")
        return 1

    # Create checkpoint directory
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Run training
    try:
        return train(config)
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        if config.log_file:
            logger.info(f"See log file for details: {config.log_file}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

