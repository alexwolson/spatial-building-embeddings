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
import torch.nn.functional as F
from rich.logging import RichHandler
from torch.utils.data import DataLoader, RandomSampler

from train_specialized_embeddings.config import (
    TripletTrainingConfig,
    load_config_from_file,
)
from train_specialized_embeddings.datasets import TripletDataset
from train_specialized_embeddings.model import EmbeddingProjector, TripletLossWrapper


LOGGER_NAME = "train_specialized_embeddings"


def setup_logging(log_file: Path | None = None) -> logging.Logger:
    """Configure logging with a Rich console handler and optional file output."""
    handlers: list[logging.Handler] = []

    console_handler = RichHandler(rich_tracebacks=True, show_path=False)
    console_handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
    handlers.append(console_handler)

    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        )
        handlers.append(file_handler)

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    for existing in logger.handlers:
        existing.close()
    logger.handlers.clear()
    for handler in handlers:
        logger.addHandler(handler)
    logger.propagate = False
    return logger


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data(
    config: TripletTrainingConfig, logger: logging.Logger
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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


def validate(
    model: nn.Module,
    val_dataset: TripletDataset,
    loss_fn: TripletLossWrapper,
    device: torch.device,
    logger: logging.Logger,
    batch_size: int = 256,
    num_workers: int = 0,
    pin_memory: bool = False,
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
    effective_samples = min(num_samples, len(val_dataset))
    if effective_samples == 0:
        logger.warning("Validation dataset has no samples; skipping validation.")
        return {
            "val_loss": float("inf"),
            "margin_violation_rate": float("nan"),
        }

    sampler = RandomSampler(
        val_dataset,
        replacement=True,
        num_samples=effective_samples,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    total_loss = 0.0
    num_batches = 0
    violation_count = 0
    total_triplets = 0

    with torch.no_grad():
        for batch in val_loader:
            anchor = batch.anchor_embedding.to(device)
            positive = batch.positive_embedding.to(device)
            negative = batch.negative_embedding.to(device)

            # Project embeddings
            anchor_proj = model(anchor)
            positive_proj = model(positive)
            negative_proj = model(negative)

            # Compute loss
            loss = loss_fn(anchor_proj, positive_proj, negative_proj)
            batch_size_actual = anchor.size(0)
            total_loss += loss.item() * batch_size_actual
            num_batches += 1
            total_triplets += batch_size_actual

            # Margin violations
            if loss_fn.distance == "euclidean":
                pos_dist = torch.norm(anchor_proj - positive_proj, p=2, dim=1)
                neg_dist = torch.norm(anchor_proj - negative_proj, p=2, dim=1)
            else:  # cosine distance (1 - cosine similarity)
                pos_dist = 1 - F.cosine_similarity(anchor_proj, positive_proj)
                neg_dist = 1 - F.cosine_similarity(anchor_proj, negative_proj)

            violation_mask = pos_dist + loss_fn.margin > neg_dist
            violation_count += int(violation_mask.sum().item())

    avg_loss = total_loss / total_triplets if total_triplets > 0 else float("inf")
    violation_rate = (
        violation_count / total_triplets if total_triplets > 0 else float("nan")
    )

    metrics = {
        "val_loss": avg_loss,
        "margin_violation_rate": violation_rate,
    }

    logger.info(
        "Validation loss: %.6f, Margin violation rate: %.4f",
        avg_loss,
        violation_rate,
    )

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

    if embeddings_df.empty:
        logger.warning(
            "No embeddings available to project for %s; skipping.", output_dir
        )
        return

    embeddings_array = np.stack(embeddings_df["embedding"].to_numpy()).astype(
        np.float32, copy=False
    )

    projected_chunks: list[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, len(embeddings_array), batch_size):
            batch_embeddings = embeddings_array[start : start + batch_size]
            batch_tensor = torch.from_numpy(batch_embeddings).to(device)
            batch_projected = model(batch_tensor).cpu().numpy()
            projected_chunks.append(batch_projected)

    projected_embeddings = np.concatenate(projected_chunks, axis=0)

    # Create output dataframe (retain original embeddings for downstream analysis)
    output_df = embeddings_df.copy()
    output_df["specialized_embedding"] = projected_embeddings.tolist()

    # Save to parquet
    output_file = output_dir / "specialized_embeddings.parquet"
    output_df.to_parquet(
        output_file, engine="pyarrow", compression="snappy", index=False
    )

    logger.info(f"Saved {len(output_df):,} projected embeddings to: {output_file}")


def train(config: TripletTrainingConfig) -> int:
    """Main training function."""
    wandb_run: Any | None = None
    if config.wandb_enabled:
        import wandb  # type: ignore[import]

        config_dict = json.loads(config.model_dump_json())
        init_kwargs: dict[str, Any] = {
            "project": config.wandb_project,
            "entity": config.wandb_entity,
            "name": config.wandb_run_name,
            "mode": config.wandb_mode,
            "config": config_dict,
        }
        init_kwargs = {k: v for k, v in init_kwargs.items() if v is not None}
        wandb_run = wandb.init(**init_kwargs)

    logger = setup_logging(config.log_file)
    logger.info("=" * 60)
    logger.info("Triplet Loss Training: Starting")
    logger.info("=" * 60)

    if wandb_run:
        logger.info(
            "wandb initialised (mode=%s, run=%s, project=%s)",
            config.wandb_mode,
            wandb_run.name,
            getattr(wandb_run, "project", config.wandb_project),
        )

    try:
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
        loss_fn = TripletLossWrapper(
            margin=config.margin, distance=config.loss_distance
        )

        # Initialize optimizer
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Resume from checkpoint if specified
        start_epoch = 0
        if config.resume_from_checkpoint:
            start_epoch = load_checkpoint(
                config.resume_from_checkpoint, model, optimizer, logger
            )
            start_epoch += 1  # Start from next epoch

        # Training loop
        logger.info("Starting training...")
        logger.info(f"Total epochs: {config.num_epochs}")
        logger.info(f"Batch size: {config.batch_size}")
        logger.info(f"Learning rate: {config.learning_rate}")

        best_val_loss = float("inf")
        no_improvement_epochs = 0
        global_step = 0
        val_metrics = {"val_loss": float("inf")}  # Initialize with default value
        early_stop_triggered = False
        last_epoch_completed = start_epoch - 1

        for epoch in range(start_epoch, config.num_epochs):
            model.train()
            epoch_loss = 0.0
            num_batches = 0

            epoch_start_time = time.time()

            for _, batch in enumerate(train_loader):
                anchor = batch.anchor_embedding.to(device)
                positive = batch.positive_embedding.to(device)
                negative = batch.negative_embedding.to(device)
                bands = batch.difficulty_band

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

                if wandb_run and global_step % config.log_every_n_batches == 0:
                    wandb.log(
                        {
                            "train/batch_loss": loss.item(),
                            "train/epoch": epoch + 1,
                        },
                        step=global_step,
                    )

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

            epoch_elapsed = time.time() - epoch_start_time
            avg_epoch_loss = (
                epoch_loss / num_batches if num_batches > 0 else float("inf")
            )

            logger.info(
                f"Epoch {epoch+1}/{config.num_epochs} completed in {epoch_elapsed:.2f}s, "
                f"Average loss: {avg_epoch_loss:.6f}"
            )
            if wandb_run:
                wandb.log(
                    {
                        "train/epoch_loss": avg_epoch_loss,
                        "train/epoch_duration_s": epoch_elapsed,
                        "train/epoch": epoch + 1,
                    },
                    step=global_step,
                )

            # Validation
            if (epoch + 1) % config.validate_every_n_epochs == 0:
                val_metrics = validate(
                    model,
                    val_dataset,
                    loss_fn,
                    device,
                    logger,
                    batch_size=config.batch_size,
                    num_workers=config.num_workers,
                    pin_memory=config.pin_memory,
                )
                val_loss = val_metrics["val_loss"]

                if wandb_run:
                    val_log = {f"val/{k}": v for k, v in val_metrics.items()}
                    val_log["val/epoch"] = epoch + 1
                    wandb.log(val_log, step=global_step)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    logger.info(f"New best validation loss: {best_val_loss:.6f}")
                    no_improvement_epochs = 0
                else:
                    no_improvement_epochs += 1
                    logger.info(
                        "Validation loss did not improve (%.6f >= %.6f). Patience counter: %d",
                        val_loss,
                        best_val_loss,
                        no_improvement_epochs,
                    )
                    if (
                        config.early_stopping_patience > 0
                        and no_improvement_epochs >= config.early_stopping_patience
                    ):
                        logger.info(
                            "Early stopping triggered after %d validation checks without improvement.",
                            no_improvement_epochs,
                        )
                        early_stop_triggered = True

            last_epoch_completed = epoch

            if early_stop_triggered:
                break

            # Save checkpoint
            if (epoch + 1) % config.save_every_n_epochs == 0:
                checkpoint_path = (
                    config.checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt"
                )
                metrics = {
                    "train_loss": avg_epoch_loss,
                    "val_loss": val_metrics.get("val_loss", float("inf")),
                }
                save_checkpoint(
                    model, optimizer, epoch, metrics, checkpoint_path, logger
                )

            # Log UCB statistics
            ucb_stats = train_dataset.ucb_sampler.get_statistics()
            logger.info(f"UCB Statistics: {ucb_stats}")
            if wandb_run:
                ucb_log: dict[str, Any] = {
                    "ucb/total_samples": ucb_stats.get("total_samples", 0),
                }
                for band, count in sorted(ucb_stats.get("band_counts", {}).items()):
                    ucb_log[f"ucb/band_count/{band}"] = count
                for band, mean_reward in sorted(
                    ucb_stats.get("band_means", {}).items()
                ):
                    ucb_log[f"ucb/band_mean_reward/{band}"] = mean_reward
                wandb.log(ucb_log, step=global_step)

        # Save final checkpoint
        final_epoch = last_epoch_completed if last_epoch_completed >= 0 else 0
        final_checkpoint_path = config.checkpoint_dir / "checkpoint_final.pt"
        metrics = {
            "train_loss": avg_epoch_loss,
            "val_loss": val_metrics.get("val_loss", float("inf")),
        }
        save_checkpoint(
            model, optimizer, final_epoch, metrics, final_checkpoint_path, logger
        )

        # Save embeddings if requested
        if config.output_embeddings_dir:
            logger.info("Saving final embeddings...")
            # Save for both train and val splits
            save_embeddings(
                model, train_df, config.output_embeddings_dir / "train", device, logger
            )
            save_embeddings(
                model, val_df, config.output_embeddings_dir / "val", device, logger
            )

        logger.info("=" * 60)
        logger.info("Training complete!")
        logger.info("=" * 60)

        if wandb_run:
            summary: dict[str, Any] = {
                "training/epochs_completed": last_epoch_completed + 1,
                "training/early_stopped": early_stop_triggered,
            }
            if math.isfinite(avg_epoch_loss):
                summary["training/final_epoch_loss"] = avg_epoch_loss
            if math.isfinite(best_val_loss):
                summary["training/best_val_loss"] = best_val_loss
            final_val_loss = val_metrics.get("val_loss")
            if isinstance(final_val_loss, (int, float)) and math.isfinite(
                final_val_loss
            ):
                summary["training/final_val_loss"] = final_val_loss
            wandb_run.summary.update(summary)

        return 0
    finally:
        if wandb_run:
            wandb_run.finish()


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
        logger = setup_logging(None)
        logger.error(f"Error loading configuration: {e}")
        return 1

    logger = setup_logging(config.log_file)

    # Validate inputs
    if not config.train_parquet_path.exists():
        logger.error(f"Training parquet file not found: {config.train_parquet_path}")
        return 1

    if not config.val_parquet_path.exists():
        logger.error(f"Validation parquet file not found: {config.val_parquet_path}")
        return 1

    if not config.difficulty_metadata_path.exists():
        logger.error(
            f"Difficulty metadata file not found: {config.difficulty_metadata_path}"
        )
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
