#!/bin/bash
#SBATCH --account=<ACCOUNT>          # Required: Resource Allocation Project (set via submit script)
#SBATCH --time=4:00:00                    # Default: 4 hours (model inference takes time)
#SBATCH --mem-per-cpu=8G                   # Model + batch processing
#SBATCH --cpus-per-task=4                  # Multiple CPUs for data loading
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
#   RUST_MODULE: Rust module to load for building packages (optional, e.g., rust)
#   MODEL_NAME: Timm model name (default: vit_base_patch14_dinov2.lvd142m)
#   BATCH_SIZE: Batch size for inference (default: 128)

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

# Step 3b: Load Rust module if available (needed for some Python packages like hf-xet)
if [ -n "${RUST_MODULE:-}" ]; then
    if module avail "${RUST_MODULE}" 2>/dev/null | grep -q "${RUST_MODULE}"; then
        echo "Loading Rust module: ${RUST_MODULE}"
        module load "${RUST_MODULE}" || echo "Warning: Failed to load Rust module - some packages may fail to build"
    else
        echo "Warning: Rust module not available: ${RUST_MODULE}"
    fi
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

# Step 8: Check if output already exists (resume capability)
# Normalize OUTPUT_DIR to remove trailing slashes
OUTPUT_DIR="${OUTPUT_DIR%/}"
PARQUET_BASENAME=$(basename "${PARQUET_FILE}" .parquet)
OUTPUT_FILE="${OUTPUT_DIR}/${PARQUET_BASENAME}_embeddings.parquet"

if [ -f "${OUTPUT_FILE}" ]; then
    echo "Skipping ${PARQUET_FILE} - output already exists: ${OUTPUT_FILE}"
    exit 0
fi

# Step 9: Create config file and run processing script
echo "Starting processing at $(date '+%Y-%m-%d %H:%M:%S')"

# Determine project root - use environment variable if set, otherwise auto-detect
if [ -z "${PROJECT_ROOT:-}" ]; then
    # Get the directory where this script is located
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
else
    PROJECT_ROOT="$(cd "${PROJECT_ROOT}" && pwd)"
fi

if [ ! -f "${PROJECT_ROOT}/generate_embeddings/generate_embeddings.py" ]; then
    echo "Error: generate_embeddings.py not found at ${PROJECT_ROOT}/generate_embeddings/generate_embeddings.py" >&2
    exit 1
fi

# Add project root to PYTHONPATH so imports work
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# Use SLURM_TMPDIR (node-local storage) for extraction if available
# This reduces memory pressure and improves I/O performance for large tar files
EXTRACT_DIR="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/extract_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${EXTRACT_DIR}"

# Create temporary config file in temp directory
CONFIG_DIR="${TMPDIR:-/tmp}"
CONFIG_FILE="${CONFIG_DIR}/generate_embeddings_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.toml"

# Set defaults for model name and batch size
MODEL_NAME="${MODEL_NAME:-vit_base_patch14_dinov2.lvd142m}"
BATCH_SIZE="${BATCH_SIZE:-128}"

# Ensure cleanup on exit
cleanup() {
    rm -f "${CONFIG_FILE}"
    # Clean up extraction directory if it exists
    if [ -d "${EXTRACT_DIR}" ]; then
        rm -rf "${EXTRACT_DIR}"
    fi
}
trap cleanup EXIT

cat > "${CONFIG_FILE}" << EOF
[generate_embeddings]
parquet_file = "${PARQUET_FILE}"
tar_file = null  # Auto-detect from parquet filename
output_dir = "${OUTPUT_DIR}"
model_name = "${MODEL_NAME}"
batch_size = ${BATCH_SIZE}
temp_dir = "${EXTRACT_DIR}"
EOF

python "${PROJECT_ROOT}/generate_embeddings/generate_embeddings.py" \
    --config "${CONFIG_FILE}"

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

