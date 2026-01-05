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
import os
import sys
import time
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rich.logging import RichHandler
from torch.utils.data import DataLoader, RandomSampler

from config import (
    TripletTrainingConfig,
    load_config_from_file,
)
from train_specialized_embeddings.datasets import TripletDataset
from publish_model.modeling_spatial_embeddings import EmbeddingProjector
from train_specialized_embeddings.loss import TripletLossWrapper


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


def get_data_loader_config(device: torch.device) -> tuple[int, bool]:
    """
    Determine num_workers and pin_memory based on available CPU cores.
    
    Checks SLURM_CPUS_PER_TASK first (for SLURM jobs), then falls back to
    os.cpu_count(). pin_memory is True for CUDA devices, False otherwise.
    
    Args:
        device: The torch device being used
        
    Returns:
        Tuple of (num_workers, pin_memory)
    """
    # Check SLURM_CPUS_PER_TASK first (for SLURM jobs)
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        try:
            num_workers = int(slurm_cpus)
        except ValueError:
            num_workers = os.cpu_count() or 1
    else:
        num_workers = os.cpu_count() or 1
    
    # Ensure at least 1 worker, but cap at available CPUs
    num_workers = max(1, min(num_workers, os.cpu_count() or 1))
    
    # pin_memory is beneficial for CUDA devices
    pin_memory = device.type == "cuda"
    
    return num_workers, pin_memory


def _validate_parquet_path(
    path: Path | str, label: str, logger: logging.Logger
) -> Path:
    """Validate a parquet path and raise early if it's missing or malformed."""
    resolved = Path(path)
    logger.info(
        "%s parquet path resolved | type=%s | path=%s",
        label,
        type(path).__name__,
        resolved,
    )

    if not resolved.exists():
        raise FileNotFoundError(f"{label} parquet file not found: {resolved}")
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} parquet path is not a file: {resolved}")

    try:
        # Quick footer/magic-byte check so we fail with a clearer message than "<Buffer>"
        pq.ParquetFile(resolved)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"{label} parquet is invalid: {exc}") from exc

    return resolved


def load_data(
    config: TripletTrainingConfig, logger: logging.Logger
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load training, validation, and difficulty metadata dataframes."""
    logger.info("Loading data files...")

    train_path = _validate_parquet_path(
        config.train_parquet_path, "train", logger
    )
    logger.info(f"Loading training data from: {train_path}")
    train_df = pd.read_parquet(train_path, engine="pyarrow")
    logger.info(f"Loaded {len(train_df):,} training samples")

    # Load validation data
    val_path = _validate_parquet_path(config.val_parquet_path, "val", logger)
    logger.info(f"Loading validation data from: {val_path}")
    val_df = pd.read_parquet(val_path, engine="pyarrow")
    logger.info(f"Loaded {len(val_df):,} validation samples")

    # Load difficulty metadata
    difficulty_path = _validate_parquet_path(
        config.difficulty_metadata_path, "difficulty", logger
    )
    logger.info(f"Loading difficulty metadata from: {difficulty_path}")
    difficulty_df = pd.read_parquet(difficulty_path, engine="pyarrow")
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
            total_triplets += batch_size_actual

    avg_loss = total_loss / total_triplets if total_triplets > 0 else float("inf")
    metrics = {"val_loss": avg_loss}

    logger.info("Validation loss: %.6f", avg_loss)

    return metrics


def compute_retrieval_metrics(
    model: nn.Module,
    val_dataset: TripletDataset,
    device: torch.device,
    logger: logging.Logger,
    top_k: tuple[int, ...] = (1, 5, 10, 100, 1000),
    max_queries: int = 512,
    per_building_limit: int = 4,
    seed: int = 0,
) -> dict[str, float]:
    """
    Compute building retrieval Recall@K on the validation split.

    This metric only depends on the learned embeddings and labels, making it
    stable across different hyperparameter choices (e.g., margin, loss type).
    """
    if len(val_dataset.building_ids) == 0:
        logger.warning("Validation dataset has no entries; skipping retrieval metric.")
        return {f"retrieval_recall@{k}": float("nan") for k in top_k}

    # Select candidate indices (buildings with >=2 samples so retrieval is meaningful)
    eligible_indices: list[int] = []
    for building_id, indices in val_dataset.building_to_indices.items():
        if len(indices) < 2:
            continue
        if per_building_limit > 0:
            eligible_indices.extend(indices[:per_building_limit])
        else:
            eligible_indices.extend(indices)

    if not eligible_indices:
        logger.warning(
            "Retrieval metric skipped: no buildings with at least two samples available."
        )
        return {f"retrieval_recall@{k}": float("nan") for k in top_k}

    eligible_array = np.array(eligible_indices, dtype=np.int64)
    rng = np.random.default_rng(seed)
    rng.shuffle(eligible_array)
    if max_queries > 0 and eligible_array.size > max_queries:
        eligible_array = eligible_array[:max_queries]

    num_embeddings = val_dataset.embeddings.size(0)
    if num_embeddings <= 1:
        logger.warning("Retrieval metric skipped: not enough validation embeddings.")
        return {f"retrieval_recall@{k}": float("nan") for k in top_k}

    # Project all validation embeddings once.
    model.eval()
    projected_chunks: list[torch.Tensor] = []
    projection_batch_size = 4096
    with torch.no_grad():
        for start in range(0, num_embeddings, projection_batch_size):
            batch = val_dataset.embeddings[start : start + projection_batch_size].to(
                device
            )
            projected_batch = model(batch).cpu()
            projected_chunks.append(projected_batch)
    projected_embeddings = torch.cat(projected_chunks, dim=0)

    building_ids = val_dataset.building_ids
    sorted_top_k = tuple(sorted(set(top_k)))
    max_k = min(sorted_top_k[-1], num_embeddings - 1)
    if max_k <= 0:
        logger.warning(
            "Retrieval metric skipped: not enough candidates after removing the query entry."
        )
        return {f"retrieval_recall@{k}": float("nan") for k in top_k}

    hits = {k: 0 for k in sorted_top_k}
    total_queries = 0

    for query_idx in eligible_array.tolist():
        total_queries += 1
        query_vec = projected_embeddings[query_idx]
        similarities = torch.matmul(projected_embeddings, query_vec)
        similarities[query_idx] = float("-inf")  # exclude the query itself

        topk_indices = torch.topk(similarities, k=max_k).indices.tolist()
        neighbor_buildings = [building_ids[idx] for idx in topk_indices]
        query_building = building_ids[query_idx]

        for k in sorted_top_k:
            effective_k = min(k, max_k)
            if effective_k == 0:
                continue
            top_neighbors = neighbor_buildings[:effective_k]
            if query_building in top_neighbors:
                hits[k] += 1

    if total_queries == 0:
        logger.warning("Retrieval metric skipped: no eligible validation queries.")
        return {f"retrieval_recall@{k}": float("nan") for k in top_k}

    metrics = {f"retrieval_recall@{k}": hits[k] / total_queries for k in sorted_top_k}
    metrics["retrieval_query_count"] = float(total_queries)

    logger.info(
        "Retrieval recall: "
        + ", ".join(
            f"@{k}={metrics[f'retrieval_recall@{k}']:.4f}" for k in sorted_top_k
        )
        + f" (queries={total_queries})"
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


def train(config: TripletTrainingConfig) -> int:
    """Main training function."""
    wandb_run: Any | None = None
    if config.wandb_enabled:
        try:
            import wandb  # type: ignore[import]
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-specific
            raise RuntimeError(
                "Weights & Biases logging is enabled, but the `wandb` package is not installed. "
                "Install it (e.g., `uv pip install wandb`) or disable wandb logging in the config."
            ) from exc

        config_dict = json.loads(config.model_dump_json())
        init_kwargs: dict[str, Any] = {
            "project": config.wandb_project,
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

        # Determine device (always auto-detect)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        if device.type == "cuda":
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

        # Determine data loader configuration based on available CPU cores
        num_workers, pin_memory = get_data_loader_config(device)
        logger.info(f"Data loader: num_workers={num_workers}, pin_memory={pin_memory}")

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
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        # Initialize model
        logger.info("Initializing model...")
        model = EmbeddingProjector(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            num_hidden_layers=config.num_hidden_layers,
            hidden_dim_multiplier=config.hidden_dim_multiplier,
            activation=config.activation,
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
        best_val_epoch: int | None = None
        no_improvement_epochs = 0
        global_step = 0
        val_metrics = {
            "val_loss": float("inf"),
        }
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
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                )
                val_loss = val_metrics["val_loss"]

                retrieval_metrics = compute_retrieval_metrics(
                    model,
                    val_dataset,
                    device,
                    logger,
                    config.retrieval_metric_top_k,
                    config.retrieval_metric_max_queries,
                    config.retrieval_metric_per_building_limit,
                    seed=config.seed if config.seed is not None else 0,
                )
                val_metrics.update(retrieval_metrics)

                if wandb_run:
                    val_log = {f"val/{k}": v for k, v in val_metrics.items()}
                    val_log["val/epoch"] = epoch + 1
                    wandb.log(val_log, step=global_step)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_epoch = epoch + 1
                    logger.info(f"New best validation loss: {best_val_loss:.6f}")
                    no_improvement_epochs = 0
                    # Save checkpoint on improvement
                    best_checkpoint_path = config.checkpoint_dir / "checkpoint_best.pt"
                    metrics = {
                        "train_loss": avg_epoch_loss,
                        "val_loss": val_loss,
                        "best_val_loss": best_val_loss,
                        "best_val_epoch": best_val_epoch,
                        "epochs_completed": epoch + 1,
                        "early_stopped": False,
                        "val_metrics": val_metrics.copy(),
                    }
                    save_checkpoint(
                        model, optimizer, epoch, metrics, best_checkpoint_path, logger
                    )
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
                    "best_val_loss": best_val_loss,
                    "best_val_epoch": best_val_epoch,
                    "epochs_completed": epoch + 1,
                    "early_stopped": False,
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
        best_epoch_metric = (
            best_val_epoch
            if best_val_epoch is not None
            else (last_epoch_completed + 1 if last_epoch_completed >= 0 else None)
        )
        metrics = {
            "train_loss": avg_epoch_loss,
            "val_loss": val_metrics.get("val_loss", float("inf")),
            "best_val_loss": best_val_loss,
            "best_val_epoch": best_epoch_metric,
            "epochs_completed": last_epoch_completed + 1,
            "early_stopped": early_stop_triggered,
            "val_metrics": val_metrics.copy(),
        }
        save_checkpoint(
            model, optimizer, final_epoch, metrics, final_checkpoint_path, logger
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
            if best_epoch_metric is not None:
                summary["training/best_val_epoch"] = best_epoch_metric
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
