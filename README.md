## Spatial Building Embeddings

A unified pipeline for generating and training specialized building embeddings from the 3D Street View dataset.

- **Dataset scale**: 1,006,160 unique buildings and 5,772,534 unique images across the combined train, validation, and test partitions.

## Project Structure

This repository is organized into two main components:

### 1. **`embedding_pipeline/`** - Embedding Model Development (Current Phase)
Complete pipeline for generating, training, and publishing specialized building embeddings:
- **`preprocess/`** - Extract and normalize raw tar files into structured Parquet format
- **`generate/`** - Generate embeddings using pre-trained Vision Transformer models (DINOv2/DINOv3)
- **`difficulty/`** - Compute visual similarity bands for intelligent negative sampling
- **`train/`** - Train specialized projection head using triplet loss
- **`publish/`** - Export trained models to Hugging Face format

See [`embedding_pipeline/README.md`](embedding_pipeline/README.md) for detailed documentation.

### 2. **`clustering/`** - Clustering Phase (Future Work)
Reserved for the next phase: clustering the specialized embeddings with additional datasets. See [`clustering/README.md`](clustering/README.md) for planned functionality.

### 3. **`download_raw_data/`** - Shared Utilities
Scripts for downloading the 3D Street View dataset using aria2c.

## Pipeline Execution Order

The following graph shows the dependency relationships between all `submit_*.sh` scripts. Execute them in the order indicated by the arrows (scripts at the top/left must complete before those below/right):

```mermaid
flowchart TD
    A[embedding_pipeline/preprocess/slurm/submit_tar_jobs.sh<br/>Process raw tar files]
    
    B[embedding_pipeline/preprocess/slurm/submit_fingerprint_jobs.sh<br/>Compute fingerprints]
    C[embedding_pipeline/generate/slurm/submit_embedding_jobs.sh<br/>Generate embeddings]
    
    D[embedding_pipeline/preprocess/slurm/submit_merge.sh<br/>Merge & create splits]
    
    E[embedding_pipeline/difficulty/slurm/submit_visual_neighbors.sh<br/>Compute visual neighbors]
    
    F[embedding_pipeline/train/slurm/submit_optuna_tuning.sh<br/>Hyperparameter tuning]
    G[embedding_pipeline/train/slurm/submit_best_training.sh<br/>Train with best params]
    
    A --> B
    A --> C
    A --> D
    C --> D
    
    B --> E
    
    D --> F
    D --> G
    
    E --> F
    E --> G
    
    style A fill:#e1f5e1
    style B fill:#e1f0f5
    style C fill:#e1f0f5
    style D fill:#fff4e1
    style E fill:#f5e1f0
    style F fill:#f0e1f5
    style G fill:#f0e1f5
```

## Configuration

All workflows use the unified `config.toml` file in the repository root. See the file for configuration options organized by section:
- `[global]` - Shared settings (seed, log directory)
- `[paths]` - Data directory paths
- `[embedding_model]` - DINOv2/v3 model settings
- `[training_model]` - Projection head architecture
- `[training]` - Training hyperparameters
- `[data]` - Data processing settings
- `[infrastructure]` - Device and performance settings
- `[logging]` - Logging and monitoring (W&B)


