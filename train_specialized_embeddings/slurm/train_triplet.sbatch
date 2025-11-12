#!/bin/bash
#SBATCH --account=<ACCOUNT>          # Required: Resource Allocation Project (set via submit script)
#SBATCH --time=48:00:00              # Default: 48 hours (training can take a while)
#SBATCH --mem=64G                     # Memory for data loading and model
#SBATCH --cpus-per-task=8             # Multiple CPUs for data loading
#SBATCH --gres=gpu:1                  # GPU required for training
#SBATCH --job-name=train_triplet
#SBATCH --output=%x_%j.out            # %x = job name, %j = job ID
#SBATCH --error=%x_%j.err

# SLURM batch script for training specialized embeddings using triplet loss.
# This script is called by submit_training.sh.
#
# Environment variables expected:
#   TRAIN_PARQUET_PATH: Path to training split parquet file
#   VAL_PARQUET_PATH: Path to validation split parquet file
#   DIFFICULTY_METADATA_PATH: Path to difficulty_metadata.parquet
#   CHECKPOINT_DIR: Directory for saving checkpoints
#   OUTPUT_EMBEDDINGS_DIR: Optional directory for saving final embeddings
#   PROJECT_ROOT: Path to project root directory (optional, will auto-detect if not set)
#   VENV_PATH: Path to Python virtual environment (optional, if using venv)
#   PYTHON_MODULE: Python module to load (e.g., python/3.12)
#   ARROW_MODULE: Arrow module to load for PyArrow (optional, e.g., arrow/17.0.0)
#   CONFIG_FILE: Path to TOML config file (optional, can use env vars)

set -euo pipefail

# Log job information
echo "=========================================="
echo "Triplet Training Job Information"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

# Step 1: Clean environment (best practice)
module purge

# Step 2: Load CUDA module (required for GPU)
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA/nvidia-smi available"
else
    echo "Error: CUDA not available - GPU is required" >&2
    exit 1
fi

# Step 3: Load Arrow module (required for PyArrow, must be loaded before Python/venv)
if [ -n "${ARROW_MODULE:-}" ]; then
    echo "Loading Arrow module: ${ARROW_MODULE}"
    module load gcc 2>/dev/null || true
    module load "${ARROW_MODULE}" || echo "Warning: Failed to load Arrow module - PyArrow may not be available"
fi

# Step 4: Load Python module
if [ -n "${PYTHON_MODULE:-}" ]; then
    echo "Loading Python module: ${PYTHON_MODULE}"
    module load "${PYTHON_MODULE}"
else
    echo "Warning: PYTHON_MODULE not set, using system Python"
fi

# Step 5: Activate Python virtual environment (if using venv)
if [ -n "${VENV_PATH:-}" ]; then
    if [ ! -d "${VENV_PATH}" ]; then
        echo "Error: Virtual environment not found at ${VENV_PATH}" >&2
        exit 1
    fi
    echo "Activating virtual environment: ${VENV_PATH}"
    source "${VENV_PATH}/bin/activate"
else
    echo "Using system Python (no virtual environment)"
fi

# Step 6: Verify GPU is available
echo "Verifying GPU availability..."
if ! python -c "import torch; assert torch.cuda.is_available(), 'GPU not available'; print(f'GPU available: {torch.cuda.get_device_name(0)}')" 2>/dev/null; then
    echo "Error: GPU is not available - this job requires a GPU" >&2
    exit 1
fi

# Step 7: Determine project root
if [ -n "${PROJECT_ROOT:-}" ]; then
    PROJECT_ROOT="$(cd "${PROJECT_ROOT}" && pwd)"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi

echo "Using project root: ${PROJECT_ROOT}"

if [ ! -f "${PROJECT_ROOT}/train_specialized_embeddings/train.py" ]; then
    echo "Error: train.py not found at ${PROJECT_ROOT}/train_specialized_embeddings/train.py" >&2
    exit 1
fi

# Add project root to PYTHONPATH so imports work
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# Step 8: Create config file if not provided
if [ -n "${CONFIG_FILE:-}" ] && [ -f "${CONFIG_FILE}" ]; then
    echo "Using provided config file: ${CONFIG_FILE}"
    TRAIN_CONFIG_FILE="${CONFIG_FILE}"
else
    echo "Creating temporary config file from environment variables..."
    CONFIG_DIR="${TMPDIR:-/tmp}"
    TRAIN_CONFIG_FILE="${CONFIG_DIR}/train_triplet_${SLURM_JOB_ID}.toml"

    # Ensure cleanup on exit
    cleanup() {
        rm -f "${TRAIN_CONFIG_FILE}"
    }
    trap cleanup EXIT

    # Validate required environment variables
    if [ -z "${TRAIN_PARQUET_PATH:-}" ]; then
        echo "Error: TRAIN_PARQUET_PATH not set" >&2
        exit 1
    fi

    if [ -z "${VAL_PARQUET_PATH:-}" ]; then
        echo "Error: VAL_PARQUET_PATH not set" >&2
        exit 1
    fi

    if [ -z "${DIFFICULTY_METADATA_PATH:-}" ]; then
        echo "Error: DIFFICULTY_METADATA_PATH not set" >&2
        exit 1
    fi

    if [ -z "${CHECKPOINT_DIR:-}" ]; then
        echo "Error: CHECKPOINT_DIR not set" >&2
        exit 1
    fi

    # Create config file with environment variables
    cat > "${TRAIN_CONFIG_FILE}" << EOF
[triplet_training]
train_parquet_path = "${TRAIN_PARQUET_PATH}"
val_parquet_path = "${VAL_PARQUET_PATH}"
difficulty_metadata_path = "${DIFFICULTY_METADATA_PATH}"
checkpoint_dir = "${CHECKPOINT_DIR}"
EOF

    # Add optional output embeddings directory if set
    if [ -n "${OUTPUT_EMBEDDINGS_DIR:-}" ]; then
        echo "output_embeddings_dir = \"${OUTPUT_EMBEDDINGS_DIR}\"" >> "${TRAIN_CONFIG_FILE}"
    fi

    # Add optional log file if set
    if [ -n "${LOG_FILE:-}" ]; then
        echo "log_file = \"${LOG_FILE}\"" >> "${TRAIN_CONFIG_FILE}"
    fi

    # Add other optional config from environment (with defaults)
    if [ -n "${BATCH_SIZE:-}" ]; then
        echo "batch_size = ${BATCH_SIZE}" >> "${TRAIN_CONFIG_FILE}"
    fi

    if [ -n "${NUM_EPOCHS:-}" ]; then
        echo "num_epochs = ${NUM_EPOCHS}" >> "${TRAIN_CONFIG_FILE}"
    fi

    if [ -n "${LEARNING_RATE:-}" ]; then
        echo "learning_rate = ${LEARNING_RATE}" >> "${TRAIN_CONFIG_FILE}"
    fi

    if [ -n "${RESUME_FROM_CHECKPOINT:-}" ]; then
        echo "resume_from_checkpoint = \"${RESUME_FROM_CHECKPOINT}\"" >> "${TRAIN_CONFIG_FILE}"
    fi
fi

# Step 9: Run training script
echo "Starting training at $(date '+%Y-%m-%d %H:%M:%S')"

python "${PROJECT_ROOT}/train_specialized_embeddings/train.py" \
    --config "${TRAIN_CONFIG_FILE}"

EXIT_CODE=$?

# Step 10: Error handling
if [ ${EXIT_CODE} -ne 0 ]; then
    echo "Error: Training failed with exit code ${EXIT_CODE}" >&2
    echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
    exit ${EXIT_CODE}
fi

echo "Training completed successfully at $(date '+%Y-%m-%d %H:%M:%S')"
echo "Checkpoints saved to: ${CHECKPOINT_DIR}"

exit 0

