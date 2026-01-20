#!/bin/bash
#SBATCH --account=<ACCOUNT>          # Required: Resource Allocation Project (set via submit script)
#SBATCH --time=24:00:00                    # Default: 24 hours (model inference takes time)
#SBATCH --mem-per-cpu=21G                   # CPU memory: 124GB total / 6 cores (recommended for H100-3g.40gb on Nibi)
#SBATCH --cpus-per-task=6                  # Recommended for H100-3g.40gb MIG instance on Nibi
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1  # MIG instance: 40GB GPU memory, 6.1 RGU (50% cost vs full GPU)
#SBATCH --array=1-1%10                     # Default: 1 task (overridden by submit script)
#SBATCH --job-name=generate_embeddings
#SBATCH --output=%x_%A_%a.out  # %x = job name, %A = array job ID, %a = task ID
#SBATCH --error=%x_%A_%a.err
#
# GPU Configuration for DINOv3 ViT-7B:
#   - Using H100-3g.40gb MIG instance (40GB GPU memory, sufficient for 7B model with batch_size=16)
#   - RGU cost: 6.1 RGU (vs 12.2 RGU for full H100-80gb = 50% cost reduction)
#   - Recommended resources per Alliance docs: 6 cores, 124 GB system memory
#   - See: https://docs.alliancecan.ca/wiki/Multi-Instance_GPU/en
#   - See: https://docs.alliancecan.ca/wiki/Allocations_and_compute_scheduling/en

# SLURM batch script for generating embeddings for a single intermediate parquet file.
# This script is called by submit_embedding_jobs.sh and processes one parquet file per task.
#
# Environment variables expected:
#   PARQUET_LIST_FILE: Path to file containing list of parquet files (one per line)
#   OUTPUT_DIR: Directory for output embedding Parquet files
#   PROJECT_ROOT: Path to project root directory (optional, will auto-detect if not set)
#   VENV_PATH: Path to Python virtual environment (optional, if using venv)
#   PYTHON_MODULE: Python module to load (e.g., python/3.12)
#   ARROW_MODULE: Arrow module to load for PyArrow (optional, e.g., arrow/17.0.0)
#   Note: MODEL_NAME and BATCH_SIZE come from config.toml only

set -euo pipefail

# Load shared functions
source "${PROJECT_ROOT}/slurm/common.sh"

# Log job information
echo "=========================================="
echo "Job Array Task Information"
echo "=========================================="
echo "Job ID: ${SLURM_ARRAY_JOB_ID}"
echo "Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

# Step 1: Setup environment
load_python_env "${VENV_PATH:-}" "${PYTHON_MODULE:-}" "${ARROW_MODULE:-}"

# Step 2: Load CUDA module (required for GPU)
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA/nvidia-smi available"
else
    error_exit "CUDA not available - GPU is required"
fi

# Step 3: Verify GPU is available
echo "Verifying GPU availability..."
if ! python -c "import torch; assert torch.cuda.is_available(), 'GPU not available'; print(f'GPU available: {torch.cuda.get_device_name(0)}')" 2>/dev/null; then
    error_exit "GPU is not available - this job requires a GPU"
fi

# Step 4: Get parquet file path from list file using SLURM_ARRAY_TASK_ID
if [ ! -f "${PARQUET_LIST_FILE}" ]; then
    error_exit "Parquet list file not found: ${PARQUET_LIST_FILE}"
fi

PARQUET_FILE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${PARQUET_LIST_FILE}")

if [ -z "${PARQUET_FILE}" ]; then
    error_exit "No parquet file found at task ID ${SLURM_ARRAY_TASK_ID}"
fi

echo "Processing parquet file: ${PARQUET_FILE}"

if [ ! -f "${PROJECT_ROOT}/embedding_pipeline/generate/generate_embeddings.py" ]; then
    error_exit "generate_embeddings.py not found at ${PROJECT_ROOT}/embedding_pipeline/generate/generate_embeddings.py"
fi

# Add project root to PYTHONPATH so imports work
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# Locate base config file
BASE_CONFIG="${PROJECT_ROOT}/config.toml"
if [ ! -f "${BASE_CONFIG}" ]; then
    error_exit "Base config file not found: ${BASE_CONFIG}"
fi

# Use SLURM_TMPDIR (node-local storage) for extraction if available
# This reduces memory pressure and improves I/O performance for large tar files
EXTRACT_DIR="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/extract_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${EXTRACT_DIR}"

# Step 5: Check if output already exists (resume capability)
# Extract embeddings_dir from config.toml for checking existing files
EMBEDDINGS_DIR=$(read_toml_value "${BASE_CONFIG}" "paths" "embeddings_dir")
if [ -z "${EMBEDDINGS_DIR}" ]; then
    EMBEDDINGS_DIR="data/embeddings"
fi

PARQUET_BASENAME=$(basename "${PARQUET_FILE}" .parquet)
OUTPUT_FILE="${EMBEDDINGS_DIR}/${PARQUET_BASENAME}_embeddings.parquet"

if [ -f "${OUTPUT_FILE}" ]; then
    echo "Skipping ${PARQUET_FILE} - output already exists: ${OUTPUT_FILE}"
    exit 0
fi

# Step 6: Run processing script with base config and CLI args
echo "Starting processing at $(date '+%Y-%m-%d %H:%M:%S')"

# Ensure cleanup on exit
cleanup() {
    # Clean up extraction directory if it exists
    if [ -d "${EXTRACT_DIR}" ]; then
        rm -rf "${EXTRACT_DIR}"
    fi
}
trap cleanup EXIT

srun python -m embedding_pipeline.generate.generate_embeddings \
    --config "${BASE_CONFIG}" \
    --parquet-file "${PARQUET_FILE}" \
    --temp-dir "${EXTRACT_DIR}"

EXIT_CODE=$?

# Step 7: Error handling
if [ ${EXIT_CODE} -ne 0 ]; then
    echo "Error: Processing failed with exit code ${EXIT_CODE}" >&2
    echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
    exit ${EXIT_CODE}
fi

echo "Processing completed successfully at $(date '+%Y-%m-%d %H:%M:%S')"
echo "Output file: ${OUTPUT_FILE}"

exit 0
