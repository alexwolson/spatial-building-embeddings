# Triplet Loss Training for Specialized Embeddings

This module implements training of specialized embeddings using triplet loss with UCB-guided negative sampling.

## Overview

The training pipeline:
1. Loads precomputed image embeddings from merged parquet files (train/val splits)
2. Groups embeddings by `building_id` to form positive pairs
3. Uses UCB (Upper Confidence Bound) algorithm to select difficulty bands for negative sampling
4. Trains a two-layer MLP projection head to learn specialized embeddings optimized for building identification

## Architecture

### Model
- **Input**: 768-dimensional DINOv2 embeddings
- **Architecture**: Two-layer MLP (768 → 512 → 256)
- **Features**: GELU activation, LayerNorm, dropout, residual shortcut, L2-normalized output
- **All hyperparameters configurable via TOML config**

### Loss Function
- Uses PyTorch's `TripletMarginLoss` (euclidean or cosine distance)
- Configurable margin parameter

### UCB Sampler
- Balances exploration and exploitation when selecting difficulty bands
- Tracks per-band statistics (counts, mean rewards)
- Configurable exploration constant and warmup period

## Usage

### Local Training

```bash
# Using config file
python train_specialized_embeddings/train.py --config config.toml

# Using environment variables
export TRIPLET_TRAINING_TRAIN_PARQUET_PATH=data/merged/train.parquet
export TRIPLET_TRAINING_VAL_PARQUET_PATH=data/merged/val.parquet
export TRIPLET_TRAINING_DIFFICULTY_METADATA_PATH=data/difficulty/difficulty_metadata.parquet
export TRIPLET_TRAINING_CHECKPOINT_DIR=checkpoints/triplet
python train_specialized_embeddings/train.py
```

### SLURM Cluster Training

```bash
# Basic usage
./train_specialized_embeddings/slurm/submit_training.sh \
    --train-parquet data/merged/train.parquet \
    --val-parquet data/merged/val.parquet \
    --difficulty-metadata data/difficulty/difficulty_metadata.parquet \
    --checkpoint-dir checkpoints/triplet \
    --account <YOUR_ACCOUNT>

# With custom config file
./train_specialized_embeddings/slurm/submit_training.sh \
    --train-parquet data/merged/train.parquet \
    --val-parquet data/merged/val.parquet \
    --difficulty-metadata data/difficulty/difficulty_metadata.parquet \
    --checkpoint-dir checkpoints/triplet \
    --config config.toml \
    --account <YOUR_ACCOUNT>

# With output embeddings directory
./train_specialized_embeddings/slurm/submit_training.sh \
    --train-parquet data/merged/train.parquet \
    --val-parquet data/merged/val.parquet \
    --difficulty-metadata data/difficulty/difficulty_metadata.parquet \
    --checkpoint-dir checkpoints/triplet \
    --output-embeddings-dir data/specialized_embeddings \
    --account <YOUR_ACCOUNT>
```

### Hyperparameter Tuning with Optuna

Leverage Optuna to explore hyperparameters by launching many short training runs on Nibi (or any SLURM cluster). All workers share a single Optuna study and reuse the standard triplet trainer.

1. Choose a study name and a storage location that every node can read/write (for SQLite, place the file on scratch or project space).
2. Launch the workers:

   ```bash
   ./train_specialized_embeddings/slurm/submit_optuna_tuning.sh \
       --account <ACCOUNT> \
       --study-name triplet_tuning \
       --storage-path /home/<user>/scratch/optuna/triplet.db \
       --num-workers 16 \
       --trials-per-worker 2 \
       --max-epochs 15 \
       --disable-wandb
   ```

3. Monitor with `squeue -j <JOB_ID>` and tail logs from `train_specialized_embeddings/logs/optuna`.

Each worker invokes `optuna_worker.py`, samples a configuration, trains, and records results under `train_specialized_embeddings/optuna_trials/trial_#####/`. Use the submit script flags to control concurrency (`--max-concurrent`), WandB behaviour, epoch budgets, or to point at a non-SQLite DSN via `--storage-url`.

## Configuration

See `config.example.toml` for a complete example configuration file. All hyperparameters can be specified via:
- TOML config file (recommended)
- Environment variables with `TRIPLET_TRAINING_` prefix
- Command-line arguments (for SLURM submission script)

### Key Configuration Options

**Model Architecture:**
- `input_dim`: Input embedding dimension (default: 768 for DINOv2 base)
- `hidden_dim`: Hidden layer dimension (default: 512)
- `output_dim`: Output embedding dimension (default: 256)
- `dropout`: Dropout probability (default: 0.1)
- `use_residual`: Use residual shortcut (default: true)
- `use_layer_norm`: Use LayerNorm (default: true)

**Training:**
- `batch_size`: Batch size (default: 256)
- `num_epochs`: Number of epochs (default: 50)
- `learning_rate`: Learning rate (default: 1e-4)
- `weight_decay`: Weight decay (default: 1e-5)
- `margin`: Triplet loss margin (default: 0.5)
- `loss_distance`: Distance metric: "euclidean" or "cosine" (default: "euclidean")
- Epoch length now always covers every valid anchor candidate once per epoch.

**UCB Sampler:**
- `ucb_exploration_constant`: UCB exploration constant c (default: 2.0)
- `ucb_warmup_samples`: Warmup samples before UCB (default: 1000)

**Checkpointing:**
- `save_every_n_epochs`: Save checkpoint frequency (default: 5)
- `validate_every_n_epochs`: Validation frequency (default: 1)
- `resume_from_checkpoint`: Path to checkpoint to resume from (optional)

## Output

### Checkpoints
- Saved to `checkpoint_dir` as `checkpoint_epoch_{N}.pt`
- Final checkpoint saved as `checkpoint_final.pt`
- Each checkpoint contains:
  - Model state dict
  - Optimizer state dict
  - Epoch number
  - Training/validation metrics, including `best_val_loss`, `best_val_epoch`, total epochs completed, and whether early stopping fired

### Specialized Embeddings (Optional)
- If `output_embeddings_dir` is specified, final projected embeddings are saved
- Saved as `{output_embeddings_dir}/train/specialized_embeddings.parquet` and `{output_embeddings_dir}/val/specialized_embeddings.parquet`
- Contains all original columns plus an additional `specialized_embedding` column (the original `embedding` column is retained)

## Requirements

- Python 3.11+
- PyTorch 2.9.0+
- pandas, numpy, rich, pydantic, pydantic-settings
- GPU recommended for training (CUDA required)

## Data Requirements

**Input Files:**
- Training parquet: Must contain `building_id` and `embedding` columns
- Validation parquet: Must contain `building_id` and `embedding` columns
- Difficulty metadata: Must contain `target_coord_hash`, `neighbor_building_ids`, `neighbor_bands` columns

**Note**: The training parquet files should have `target_coord_hash` column for matching with difficulty metadata. This is added by `preprocess_raw_data/merge_and_split.py`.

## Monitoring

### Local Training
- Logs printed to console (or log file if `log_file` is specified)
- Metrics logged every `log_every_n_batches` batches
- Validation metrics logged every `validate_every_n_epochs` epochs

### SLURM Training
- Monitor job: `squeue -j <JOB_ID>`
- View logs: `tail -f train_specialized_embeddings/logs/train_triplet_<JOB_ID>.out`
- Check errors: `tail -f train_specialized_embeddings/logs/train_triplet_<JOB_ID>.err`

## Troubleshooting

**GPU not available:**
- Ensure CUDA is available: `nvidia-smi`
- Check SLURM GPU allocation: `--gres=gpu:1` in batch script

**Out of memory:**
- Reduce `batch_size` in config
- Reduce `num_workers` in config

**Difficulty metadata not matching:**
- Ensure training parquet files have `target_coord_hash` column
- Check that `difficulty_metadata.parquet` was generated from the same dataset

**Checkpoint loading fails:**
- Ensure checkpoint path is correct
- Check that model architecture matches checkpoint

