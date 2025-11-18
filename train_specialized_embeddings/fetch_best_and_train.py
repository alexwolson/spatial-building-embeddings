#!/usr/bin/env python3
"""
Fetch best hyperparameters from WandB Optuna trials and train final model.

This script queries WandB to find the best Optuna trial (by retrieval_recall@100),
extracts its hyperparameters, generates a training config, and immediately runs
training with unlimited epochs, early stopping, and checkpoint-on-improvement.
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import Any

import wandb  # type: ignore[import]
from rich.logging import RichHandler

from config import (
    TripletTrainingConfig,
    load_config_from_file,
)
from train_specialized_embeddings.train import train as run_training


LOGGER_NAME = "fetch_best_and_train"


def setup_logging(log_file: Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """Set up logging with Rich handler and optional file output."""
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
    logger.setLevel(level)
    for existing in logger.handlers:
        existing.close()
    logger.handlers.clear()
    for handler in handlers:
        logger.addHandler(handler)
    logger.propagate = False
    return logger


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Fetch best hyperparameters from WandB and train final model."
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to TOML config file. If not provided, config is loaded from environment variables.",
    )
    return parser.parse_args()


def find_best_run(
    project: str,
    trial_name_pattern: str,
    metric_name: str,
    logger: logging.Logger,
) -> wandb.apis.public.Run:
    """
    Find the best run from WandB project based on metric.

    Args:
        project: WandB project name
        trial_name_pattern: Regex pattern to match trial run names
        metric_name: Metric name to use for ranking (e.g., 'val/retrieval_recall@100')
        logger: Logger instance

    Returns:
        Best run based on the specified metric

    Raises:
        RuntimeError: If no runs found or metric not available
    """
    logger.info("Querying WandB for runs in project: %s", project)
    api = wandb.Api()

    try:
        runs = list(
            api.runs(
                project,
                filters={"display_name": {"$regex": trial_name_pattern}},
            )
        )
        logger.info("Found %d runs matching pattern '%s'", len(runs), trial_name_pattern)
    except Exception as exc:
        raise RuntimeError(f"Failed to query WandB project '{project}': {exc}") from exc

    if not runs:
        raise RuntimeError(
            f"No runs found matching pattern '{trial_name_pattern}' in project '{project}'"
        )

    # Sort by metric (highest first)
    def get_metric_value(run: wandb.apis.public.Run) -> float:
        summary = run.summary
        for key_variant in (metric_name, metric_name.replace("val/", "")):
            if key_variant in summary:
                try:
                    return float(summary[key_variant])
                except (ValueError, TypeError):
                    return float("-inf")
        return float("-inf")

    runs.sort(key=get_metric_value, reverse=True)

    best_run = runs[0]
    best_metric_value = get_metric_value(best_run)

    if not math.isfinite(best_metric_value):
        raise RuntimeError(
            f"Best run '{best_run.name}' does not have a valid metric value for '{metric_name}'"
        )

    logger.info(
        "Best run: %s (metric=%s: %.6f)",
        best_run.name,
        metric_name,
        best_metric_value,
    )

    return best_run


def extract_hyperparameters(
    run: wandb.apis.public.Run, base_config: TripletTrainingConfig, logger: logging.Logger
) -> dict[str, Any]:
    """
    Extract hyperparameters from WandB run config.

    Args:
        run: WandB run object
        base_config: Base config to use for non-hyperparameter fields
        logger: Logger instance

    Returns:
        Dictionary of hyperparameters to override in config
    """
    logger.info("Extracting hyperparameters from run: %s", run.name)

    # Get config from run
    run_config = dict(run.config)

    # Map of config keys we want to extract (hyperparameters only)
    hyperparam_keys = {
        "learning_rate",
        "weight_decay",
        "margin",
        "dropout",
        "use_residual",
        "use_layer_norm",
        "loss_distance",
        "hidden_dim",
        "num_hidden_layers",
        "hidden_dim_multiplier",
        "activation",
        "batch_size",
        "ucb_exploration_constant",
        "ucb_warmup_samples",
        "retrieval_metric_max_queries",
    }

    hyperparams: dict[str, Any] = {}

    for key in hyperparam_keys:
        if key in run_config:
            value = run_config[key]
            # Handle nested config (wandb sometimes nests config)
            if isinstance(value, dict) and "value" in value:
                value = value["value"]
            hyperparams[key] = value
            logger.debug("Extracted %s = %s", key, value)
        else:
            logger.warning("Hyperparameter '%s' not found in run config, using base config", key)

    logger.info("Extracted %d hyperparameters", len(hyperparams))
    return hyperparams




def main() -> int:
    """Main entry point."""
    args = parse_args()
    
    # Set up logger early (before config loading to catch config errors)
    logger = setup_logging(None, logging.INFO)

    # Load configuration
    try:
        if args.config:
            config = load_config_from_file(args.config, "triplet_training")
        else:
            # Load from environment variables
            config = TripletTrainingConfig()
    except Exception as exc:
        logger.error("Error loading configuration: %s", exc)
        return 1

    # Re-setup logger with log file if specified
    logger = setup_logging(config.log_file, logging.INFO)

    logger.info("=" * 60)
    logger.info("Fetch Best Hyperparameters and Train")
    logger.info("=" * 60)

    # Validate required config fields
    if not config.wandb_project:
        logger.error("wandb_project is required in config")
        return 1

    # Find best run
    try:
        best_run = find_best_run(
            config.wandb_project,
            config.best_training_trial_name_pattern,
            config.best_training_metric_name,
            logger,
        )
    except Exception as exc:
        logger.error("Failed to find best run: %s", exc)
        return 1

    # Extract hyperparameters
    try:
        hyperparams = extract_hyperparameters(best_run, config, logger)
    except Exception as exc:
        logger.error("Failed to extract hyperparameters: %s", exc)
        return 1

    overrides: dict[str, Any] = {
        **hyperparams,
        "num_epochs": 10000,
        "early_stopping_patience": 10,
        "save_every_n_epochs": 1,
        "resume_from_checkpoint": None,
    }
    if not config.wandb_run_name:
        overrides["wandb_run_name"] = "best-hyperparams-training"

    training_config = config.model_copy(update=overrides)

    logger.info("Training config created:")
    logger.info("  Epochs: unlimited (early stopping after 10 epochs no improvement)")
    logger.info("  Early stopping patience: %d", training_config.early_stopping_patience)
    logger.info("  Checkpoint directory: %s", training_config.checkpoint_dir)
    logger.info("  WandB enabled: %s", training_config.wandb_enabled)

    # Run training
    logger.info("=" * 60)
    logger.info("Starting training with best hyperparameters")
    logger.info("=" * 60)

    try:
        exit_code = run_training(training_config)
        if exit_code != 0:
            logger.error("Training exited with code %d", exit_code)
            return exit_code
    except Exception as exc:
        logger.error("Training failed: %s", exc, exc_info=True)
        return 1

    logger.info("=" * 60)
    logger.info("Training completed successfully!")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

