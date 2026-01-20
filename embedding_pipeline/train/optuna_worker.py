#!/usr/bin/env python3
"""
Optuna worker that runs triplet training trials on SLURM nodes.

Each worker connects to a shared Optuna storage backend (typically a SQLite
database on a shared filesystem) and executes one or more trials sequentially.
For every trial the worker:
  1. Loads a base training configuration.
  2. Samples a set of hyperparameters.
  3. Materialises an isolated working directory for checkpoints and logs.
  4. Runs the standard training loop.
  5. Reports retrieval recall@100 back to Optuna.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

import optuna  # type: ignore[import]
import torch
from sqlalchemy.exc import OperationalError
from rich.logging import RichHandler

from config import (
    TripletTrainingConfig,
    load_config_from_file,
)
from train_specialized_embeddings.train import train as run_training


LOGGER_NAME = "optuna_worker"


def parse_args() -> argparse.Namespace:
    default_base_config = Path(__file__).resolve().parent / "config.toml"
    default_output_root = Path(__file__).resolve().parent / "optuna_trials"

    parser = argparse.ArgumentParser(
        description="Run Optuna-powered hyperparameter tuning trial(s)."
    )
    parser.add_argument(
        "--study-name",
        required=True,
        help="Name of the Optuna study to join.",
    )
    parser.add_argument(
        "--storage-url",
        required=True,
        help="Optuna storage URL (e.g. sqlite:////path/to/optuna.db).",
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=default_base_config,
        help=f"Path to base TOML config. Default: {default_base_config}",
    )
    parser.add_argument(
        "--trial-output-root",
        type=Path,
        default=default_output_root,
        help=f"Directory to store per-trial artefacts. Default: {default_output_root}",
    )
    parser.add_argument(
        "--trials-per-worker",
        type=int,
        default=1,
        help="Number of Optuna trials to execute sequentially in this worker.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Override the number of training epochs for every trial.",
    )
    parser.add_argument(
        "--disable-wandb",
        action="store_true",
        help="Disable Weights & Biases logging for trials.",
    )
    parser.add_argument(
        "--wandb-mode",
        type=str,
        choices=("online", "offline"),
        default=None,
        help="Override wandb mode if logging remains enabled.",
    )
    parser.add_argument(
        "--sqlite-timeout",
        type=float,
        default=60.0,
        help="Timeout (seconds) for SQLite connections if using sqlite storage.",
    )
    parser.add_argument(
        "--worker-id",
        default=os.getenv("SLURM_JOB_ID"),
        help="Identifier for this worker (defaults to SLURM job id when available).",
    )
    parser.add_argument(
        "--verbosity",
        type=int,
        default=20,
        help="Logging level (10=DEBUG, 20=INFO, ...).",
    )
    return parser.parse_args()


def setup_logging(level: int) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    handler = RichHandler(rich_tracebacks=True, show_path=False)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.propagate = False
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return logger


def allocate_trial_directory(root: Path, trial_number: int) -> Path:
    base_dir = root / f"trial_{trial_number:05d}"
    base_dir.mkdir(parents=True, exist_ok=True)
    if not any(base_dir.iterdir()):
        return base_dir

    attempt = 1
    while True:
        candidate = base_dir / f"attempt_{attempt:02d}"
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        attempt += 1


def sample_hyperparameters(
    trial: optuna.trial.Trial,
    base_config: TripletTrainingConfig,
) -> dict[str, Any]:
    """Return a dictionary of config overrides sampled for this trial."""
    params: dict[str, Any] = {}

    params["learning_rate"] = trial.suggest_float("learning_rate", 5e-5, 5e-4, log=True)
    params["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    params["margin"] = trial.suggest_float("margin", 0.2, 1.0)
    params["dropout"] = trial.suggest_float("dropout", 0.0, 0.4)
    params["use_residual"] = trial.suggest_categorical("use_residual", [True, False])
    params["use_layer_norm"] = trial.suggest_categorical(
        "use_layer_norm", [True, False]
    )
    params["loss_distance"] = trial.suggest_categorical(
        "loss_distance", ["euclidean", "cosine"]
    )
    params["hidden_dim"] = trial.suggest_categorical("hidden_dim", [256, 384, 512, 768])
    params["num_hidden_layers"] = trial.suggest_int("num_hidden_layers", 1, 3)
    params["hidden_dim_multiplier"] = trial.suggest_categorical(
        "hidden_dim_multiplier", [0.75, 1.0, 1.25]
    )
    params["activation"] = trial.suggest_categorical(
        "activation", ["gelu", "silu", "relu"]
    )
    params["batch_size"] = trial.suggest_categorical("batch_size", [128, 192, 256, 320])
    params["ucb_exploration_constant"] = trial.suggest_float(
        "ucb_exploration_constant", 0.5, 3.5
    )
    params["ucb_warmup_samples"] = trial.suggest_int(
        "ucb_warmup_samples", 500, 5000, step=500
    )

    # Keep retrieval metric sampling bounded
    params["retrieval_metric_max_queries"] = trial.suggest_int(
        "retrieval_metric_max_queries",
        max(128, base_config.retrieval_metric_max_queries // 2),
        base_config.retrieval_metric_max_queries,
    )

    return params


def build_config_with_updates(
    base_config: TripletTrainingConfig,
    updates: dict[str, Any],
) -> TripletTrainingConfig:
    payload = base_config.model_dump(mode='json')
    payload.update(updates)
    return TripletTrainingConfig(**payload)


def determine_storage(
    storage_url: str,
    sqlite_timeout: float,
    max_retries: int = 5,
    retry_sleep: float = 0.5,
) -> optuna.storages.BaseStorage:
    def _create_storage() -> optuna.storages.BaseStorage:
        if storage_url.startswith("sqlite"):
            engine_kwargs = {"connect_args": {"timeout": sqlite_timeout}}
            return optuna.storages.RDBStorage(
                url=storage_url,
                engine_kwargs=engine_kwargs,
            )
        return optuna.storages.RDBStorage(url=storage_url)

    if not storage_url.startswith("sqlite"):
        return _create_storage()

    logger = logging.getLogger(LOGGER_NAME)
    attempt = 0
    while True:
        try:
            return _create_storage()
        except OperationalError as exc:
            root = exc.orig if hasattr(exc, "orig") else exc
            message = str(root).lower()
            table_exists = isinstance(root, sqlite3.OperationalError) and (
                "already exists" in message or "exists" in message
            )
            if not table_exists or attempt >= max_retries:
                raise
            backoff = retry_sleep * (attempt + 1)
            logger.warning(
                "SQLite storage initialisation race detected (attempt %d/%d). "
                "Retrying in %.2fs...",
                attempt + 1,
                max_retries,
                backoff,
            )
            time.sleep(backoff)
            attempt += 1


def run_trial(
    trial: optuna.trial.Trial,
    base_config: TripletTrainingConfig,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> float:
    params = sample_hyperparameters(trial, base_config)
    trial_dir = allocate_trial_directory(args.trial_output_root, trial.number)
    checkpoints_dir = trial_dir / "checkpoints"
    logs_dir = trial_dir / "logs"

    logs_dir.mkdir(parents=True, exist_ok=True)

    wandb_enabled = base_config.wandb_enabled and not args.disable_wandb
    if wandb_enabled and not base_config.wandb_project:
        logger.warning(
            "wandb is enabled in base config but no project is configured; disabling for trial %d",
            trial.number,
        )
        wandb_enabled = False

    updates: dict[str, Any] = {
        **params,
        "checkpoint_dir": str(checkpoints_dir),
        "log_file": str(logs_dir / "train.log"),
        "resume_from_checkpoint": None,
        "wandb_enabled": wandb_enabled,
    }

    if args.max_epochs is not None:
        updates["num_epochs"] = args.max_epochs
        updates["save_every_n_epochs"] = max(
            1, min(base_config.save_every_n_epochs, args.max_epochs)
        )
        updates["validate_every_n_epochs"] = max(
            1, min(base_config.validate_every_n_epochs, args.max_epochs)
        )

    if args.wandb_mode:
        updates["wandb_mode"] = args.wandb_mode

    if wandb_enabled:
        base_run_name = (
            base_config.wandb_run_name or base_config.wandb_project or args.study_name
        )
        updates["wandb_run_name"] = f"{base_run_name}-trial-{trial.number:05d}"

    config = build_config_with_updates(base_config, updates)

    logger.info(
        "Starting trial %s | worker=%s | directory=%s",
        trial.number,
        args.worker_id,
        trial_dir,
    )
    logger.info("Hyperparameters: %s", {k: params[k] for k in sorted(params)})

    exit_code = run_training(config)
    if exit_code != 0:
        raise RuntimeError(f"Training exited with code {exit_code}")

    checkpoint_path = config.checkpoint_dir / "checkpoint_final.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Expected checkpoint not found for trial {trial.number}: {checkpoint_path}"
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    metrics: dict[str, Any] = checkpoint.get("metrics", {})
    best_val_loss = float(
        metrics.get("best_val_loss", metrics.get("val_loss", math.inf))
    )
    val_metrics = metrics.get("val_metrics", {}) or {}
    recall_metric_name = "retrieval_recall@100"
    recall_at_100 = float(val_metrics.get(recall_metric_name, float("nan")))

    if not math.isfinite(recall_at_100):
        raise optuna.TrialPruned(
            f"Missing or non-finite {recall_metric_name} ({recall_at_100})"
        )

    best_epoch = metrics.get("best_val_epoch")
    early_stopped = bool(metrics.get("early_stopped", False))

    trial.set_user_attr("trial_directory", str(trial_dir))
    trial.set_user_attr("early_stopped", early_stopped)
    trial.set_user_attr("best_val_epoch", best_epoch)
    trial.set_user_attr("best_val_loss", best_val_loss)
    trial.set_user_attr(recall_metric_name, recall_at_100)

    trial.report(recall_at_100, step=int(best_epoch or config.num_epochs))
    if trial.should_prune():
        raise optuna.TrialPruned(f"Pruned at epoch {best_epoch}")

    logger.info(
        (
            "Completed trial %s | recall@100=%.6f | best_val_loss=%.6f "
            "| best_epoch=%s | early_stopped=%s"
        ),
        trial.number,
        recall_at_100,
        best_val_loss,
        best_epoch,
        early_stopped,
    )

    return recall_at_100


def main() -> int:
    args = parse_args()
    logger = setup_logging(args.verbosity)

    if not args.base_config.exists():
        logger.error("Base config not found: %s", args.base_config)
        return 1

    base_config = load_config_from_file(args.base_config, "triplet_training")
    args.trial_output_root.mkdir(parents=True, exist_ok=True)

    storage = determine_storage(args.storage_url, args.sqlite_timeout)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
    )

    active_trials = study.get_trials(
        states=(
            optuna.trial.TrialState.RUNNING,
            optuna.trial.TrialState.WAITING,
        )
    )
    logger.info(
        "Worker %s joined study '%s' (storage=%s). Active/pending trials: %d",
        args.worker_id,
        args.study_name,
        args.storage_url,
        len(active_trials),
    )

    try:
        study.optimize(
            lambda trial: run_trial(trial, base_config, args, logger),
            n_trials=args.trials_per_worker,
            catch=(RuntimeError, FileNotFoundError),
        )
    except optuna.TrialPruned as exc:
        logger.warning("Trial pruned: %s", exc)
    except KeyboardInterrupt:
        logger.warning("Worker interrupted")
        return 1

    logger.info("Worker %s completed requested trials.", args.worker_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
