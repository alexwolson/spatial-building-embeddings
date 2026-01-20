# Migration Guide: Code Organization Update

This document describes the code reorganization performed to better separate the embedding pipeline from future clustering work.

## What Changed

The repository has been reorganized to clearly distinguish between different phases of the project:

### Before (Old Structure)
```
.
├── preprocess_raw_data/
├── generate_embeddings/
├── difficulty_metadata/
├── train_specialized_embeddings/
├── publish_model/
└── download_raw_data/
```

### After (New Structure)
```
.
├── embedding_pipeline/         # Phase 1: Embedding model development
│   ├── preprocess/            # (was preprocess_raw_data/)
│   ├── generate/              # (was generate_embeddings/)
│   ├── difficulty/            # (was difficulty_metadata/)
│   ├── train/                 # (was train_specialized_embeddings/)
│   └── publish/               # (was publish_model/)
├── clustering/                # Phase 2: Future clustering work (placeholder)
└── download_raw_data/         # Shared utilities (unchanged)
```

## Path Changes

### SLURM Submit Scripts
All submit scripts have moved but remain in their respective `slurm/` subdirectories:

| Old Path | New Path |
|----------|----------|
| `preprocess_raw_data/slurm/submit_*.sh` | `embedding_pipeline/preprocess/slurm/submit_*.sh` |
| `generate_embeddings/slurm/submit_*.sh` | `embedding_pipeline/generate/slurm/submit_*.sh` |
| `difficulty_metadata/slurm/submit_*.sh` | `embedding_pipeline/difficulty/slurm/submit_*.sh` |
| `train_specialized_embeddings/slurm/submit_*.sh` | `embedding_pipeline/train/slurm/submit_*.sh` |
| `publish_model/slurm/submit_*.sh` | `embedding_pipeline/publish/slurm/submit_*.sh` |

### Python Scripts
All Python scripts have moved to their new directories:

| Old Path | New Path |
|----------|----------|
| `preprocess_raw_data/*.py` | `embedding_pipeline/preprocess/*.py` |
| `generate_embeddings/*.py` | `embedding_pipeline/generate/*.py` |
| `difficulty_metadata/*.py` | `embedding_pipeline/difficulty/*.py` |
| `train_specialized_embeddings/*.py` | `embedding_pipeline/train/*.py` |
| `publish_model/*.py` | `embedding_pipeline/publish/*.py` |

### Python Module Imports
Internal imports have been updated:

```python
# Old imports
from train_specialized_embeddings.datasets import TripletDataset
from publish_model.modeling_spatial_embeddings import EmbeddingProjector
from train_specialized_embeddings.loss import TripletLossWrapper

# New imports
from embedding_pipeline.train.datasets import TripletDataset
from embedding_pipeline.publish.modeling_spatial_embeddings import EmbeddingProjector
from embedding_pipeline.train.loss import TripletLossWrapper
```

### Logger Names
Logger names have been updated to reflect the new structure:

| Old Logger Name | New Logger Name |
|----------------|----------------|
| `train_specialized_embeddings` | `embedding_pipeline.train` |
| `fetch_best_and_train` | `embedding_pipeline.train.fetch_best_and_train` |
| `optuna_worker` | `embedding_pipeline.train.optuna_worker` |

## What Stayed the Same

1. **Configuration**: The `config.toml` file remains in the repository root with the same structure
2. **Data Paths**: All data directory paths in `config.toml` remain unchanged
3. **CLI Arguments**: All command-line arguments for scripts remain the same
4. **Functionality**: No functional changes were made - this is purely a reorganization

## Impact on Workflows

### For SLURM Users
Update your workflow scripts to use the new paths:

```bash
# Old
cd preprocess_raw_data/slurm
./submit_tar_jobs.sh --account def-user

# New
cd embedding_pipeline/preprocess/slurm
./submit_tar_jobs.sh --account def-user
```

The scripts themselves handle all path updates internally, so you don't need to change any arguments.

### For Python Developers
If you have custom scripts that import from these modules, update your imports:

```python
# Old
from train_specialized_embeddings.datasets import TripletDataset

# New
from embedding_pipeline.train.datasets import TripletDataset
```

### For CI/CD Pipelines
Update any automation that references the old paths:

```yaml
# Old
- python preprocess_raw_data/process_tar.py --config config.toml

# New
- python -m embedding_pipeline.preprocess.process_tar --config config.toml
```

## Benefits of This Organization

1. **Clear Phase Separation**: The embedding pipeline and future clustering work are now clearly separated
2. **Better Navigation**: Related components are grouped together under `embedding_pipeline/`
3. **Scalability**: Easy to add new phases (e.g., `analysis/`, `visualization/`) alongside `embedding_pipeline/` and `clustering/`
4. **Maintainability**: Clearer ownership and purpose for each directory

## Troubleshooting

### "Module not found" errors
- Ensure you're using the new import paths: `embedding_pipeline.<submodule>` instead of the old module names
- Check that your `PYTHONPATH` includes the repository root

### SLURM job failures
- Update your working directory to the new script locations
- The scripts themselves have been updated, so as long as you're calling the right path, they should work

### Configuration issues
- `config.toml` location and structure are unchanged - no updates needed there

## Questions?

See the README files in each directory for detailed documentation:
- `embedding_pipeline/README.md` - Complete pipeline documentation
- `clustering/README.md` - Future clustering phase plans
- Main `README.md` - Overall project structure and execution order
