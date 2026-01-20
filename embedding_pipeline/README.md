# Embedding Pipeline

This directory contains the complete pipeline for generating and training specialized building embeddings from the 3D Street View dataset.

## Pipeline Stages

### 1. Preprocess (`preprocess/`)
Extracts and normalizes raw tar files into structured Parquet format:
- `process_tar.py` - Extract images and metadata from tar archives
- `merge_and_split.py` - Merge parquet files and create train/val/test splits
- `compute_fingerprints.py` - Generate perceptual hashes for deduplication
- `fix_intermediate_schema.py` - Fix schema issues in intermediate files
- `summarize_ids.py` - Generate building ID summaries

**Input**: Raw tar archives  
**Output**: `data/intermediates/*.parquet`

### 2. Generate (`generate/`)
Generate embeddings using pre-trained Vision Transformer models:
- `generate_embeddings.py` - Extract embeddings using DINOv2/DINOv3 models

**Input**: Images from preprocessed data  
**Output**: `data/embeddings/*.parquet` (768-4096D vectors)

### 3. Difficulty (`difficulty/`)
Compute visual similarity bands for intelligent negative sampling:
- `compute_visual_neighbors.py` - Build BallTree index and compute difficulty bands

**Input**: Merged embeddings  
**Output**: `data/difficulty/difficulty_metadata.parquet`

### 4. Train (`train/`)
Train specialized projection head using triplet loss:
- `train.py` - Main training script with triplet loss
- `loss.py` - Triplet loss implementation with UCB-guided sampling
- `datasets.py` - PyTorch dataset implementations
- `optuna_worker.py` - Hyperparameter tuning with Optuna
- `fetch_best_and_train.py` - Retrieve best hyperparameters and train final model

**Input**: Merged data + difficulty metadata  
**Output**: Model checkpoints in `checkpoints/`

### 5. Publish (`publish/`)
Export trained models to Hugging Face format:
- `convert_to_hf.py` - Convert PyTorch checkpoint to HF format
- `generate_embeddings_from_hf.py` - Inference using published model
- `generate_embeddings_from_parquet.py` - Batch inference from parquet files
- `demo_usage.py` - Example usage of published model
- `estimate_model_size.py` - Calculate model size metrics
- `configuration_spatial_embeddings.py` - HF configuration class
- `modeling_spatial_embeddings.py` - HF model class

**Input**: Trained checkpoint  
**Output**: Hugging Face compatible model

## Execution Order

The pipeline stages must be executed in this order:

```
1. preprocess/process_tar.py        → data/intermediates/
2. generate/generate_embeddings.py  → data/embeddings/
3. preprocess/merge_and_split.py    → data/merged/
4. difficulty/compute_visual_neighbors.py → data/difficulty/
5. train/optuna_worker.py (optional) → hyperparameter tuning
6. train/train.py or fetch_best_and_train.py → checkpoints/
7. publish/convert_to_hf.py         → Hugging Face model
```

See the main README.md for the complete dependency graph and SLURM submission scripts.

## Configuration

All workflows use the unified `config.toml` file in the repository root. Each stage reads from relevant configuration sections:
- `[global]` - Shared settings (seed, log directory)
- `[paths]` - Data directory paths
- `[embedding_model]` - DINOv2/v3 model settings
- `[training_model]` - Projection head architecture
- `[training]` - Training hyperparameters
- `[data]` - Data processing settings
- `[infrastructure]` - Device and performance settings
- `[logging]` - Logging and monitoring (W&B)
