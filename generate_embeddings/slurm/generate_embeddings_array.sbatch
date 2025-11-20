#!/bin/bash
#SBATCH --account=<ACCOUNT>          # Required: Resource Allocation Project (set via submit script)
#SBATCH --time=24:00:00                    # Default: 4 hours (model inference takes time)
#SBATCH --mem-per-cpu=48G                   # Model + batch processing
#SBATCH --cpus-per-task=8                  # Multiple CPUs for data loading
#SBATCH --gres=gpu:1                       # GPU required for efficient inference
#SBATCH --array=1-1%10                     # Default: 1 task (overridden by submit script)
#SBATCH --job-name=generate_embeddings
#SBATCH --output=%x_%A_%a.out  # %x = job name, %A = array job ID, %a = task ID
#SBATCH --error=%x_%A_%a.err

# SLURM batch script for generating embeddings for a single intermediate parquet file.
# This script is called by submit_embedding_jobs.sh and processes one parquet file per task.
#
# Environment variables expected:
#   PARQUET_LIST_FILE: Path to file containing list of parquet files (one per line)
#   RAW_DIR: Directory containing tar files
#   OUTPUT_DIR: Directory for output embedding Parquet files
#   PROJECT_ROOT: Path to project root directory (optional, will auto-detect if not set)
#   VENV_PATH: Path to Python virtual environment (optional, if using venv)
#   PYTHON_MODULE: Python module to load (e.g., python/3.12)
#   ARROW_MODULE: Arrow module to load for PyArrow (optional, e.g., arrow/17.0.0)
#   Note: MODEL_NAME and BATCH_SIZE come from config.toml only

set -euo pipefail

# Log job information
echo "=========================================="
echo "Job Array Task Information"
echo "=========================================="
echo "Job ID: ${SLURM_ARRAY_JOB_ID}"
echo "Task ID: ${SLURM_ARRAY_TASK_ID}"
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

# Step 7: Get parquet file path from list file using SLURM_ARRAY_TASK_ID
if [ ! -f "${PARQUET_LIST_FILE}" ]; then
    echo "Error: Parquet list file not found: ${PARQUET_LIST_FILE}" >&2
    exit 1
fi

PARQUET_FILE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${PARQUET_LIST_FILE}")

if [ -z "${PARQUET_FILE}" ]; then
    echo "Error: No parquet file found at task ID ${SLURM_ARRAY_TASK_ID}" >&2
    exit 1
fi

echo "Processing parquet file: ${PARQUET_FILE}"

# Step 8: Determine project root - prefer explicit env, otherwise fall back to script location
if [ -n "${PROJECT_ROOT:-}" ]; then
    PROJECT_ROOT="$(cd "${PROJECT_ROOT}" && pwd)"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi

echo "Using project root: ${PROJECT_ROOT}"

if [ ! -f "${PROJECT_ROOT}/generate_embeddings/generate_embeddings.py" ]; then
    echo "Error: generate_embeddings.py not found at ${PROJECT_ROOT}/generate_embeddings/generate_embeddings.py" >&2
    exit 1
fi

# Add project root to PYTHONPATH so imports work
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# Locate base config file
BASE_CONFIG="${PROJECT_ROOT}/config.toml"
if [ ! -f "${BASE_CONFIG}" ]; then
    echo "Error: Base config file not found: ${BASE_CONFIG}" >&2
    exit 1
fi

# Use SLURM_TMPDIR (node-local storage) for extraction if available
# This reduces memory pressure and improves I/O performance for large tar files
EXTRACT_DIR="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/extract_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${EXTRACT_DIR}"

# Step 9: Check if output already exists (resume capability)
# Extract embeddings_dir from config.toml for checking existing files
EMBEDDINGS_DIR=$(awk -F'=' '/^embeddings_dir/ {gsub(/^[[:space:]]*["'\'']|["'\'']$/, "", $2); gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2); print $2}' "${BASE_CONFIG}" | head -1)
if [ -z "${EMBEDDINGS_DIR}" ]; then
    # Fallback to default if not found
    EMBEDDINGS_DIR="data/embeddings"
fi
PARQUET_BASENAME=$(basename "${PARQUET_FILE}" .parquet)
OUTPUT_FILE="${EMBEDDINGS_DIR}/${PARQUET_BASENAME}_embeddings.parquet"

if [ -f "${OUTPUT_FILE}" ]; then
    echo "Skipping ${PARQUET_FILE} - output already exists: ${OUTPUT_FILE}"
    exit 0
fi

# Step 10: Run processing script with base config and CLI args
echo "Starting processing at $(date '+%Y-%m-%d %H:%M:%S')"

# Ensure cleanup on exit
cleanup() {
    # Clean up extraction directory if it exists
    if [ -d "${EXTRACT_DIR}" ]; then
        rm -rf "${EXTRACT_DIR}"
    fi
}
trap cleanup EXIT

srun python -m generate_embeddings.generate_embeddings \
    --config "${BASE_CONFIG}" \
    --parquet-file "${PARQUET_FILE}" \
    --temp-dir "${EXTRACT_DIR}"

EXIT_CODE=$?

# Step 10: Error handling
if [ ${EXIT_CODE} -ne 0 ]; then
    echo "Error: Processing failed with exit code ${EXIT_CODE}" >&2
    echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
    exit ${EXIT_CODE}
fi

echo "Processing completed successfully at $(date '+%Y-%m-%d %H:%M:%S')"
echo "Output file: ${OUTPUT_FILE}"

exit 0
