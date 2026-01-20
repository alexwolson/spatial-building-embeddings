#!/bin/bash
#SBATCH --account=<ACCOUNT>          # Required: Resource Allocation Project (set via submit script)
#SBATCH --time=12:00:00              # Default: 12 hours (merge can be long)
#SBATCH --mem=100G                   # High memory for large dataset merge
#SBATCH --cpus-per-task=8            # Multiple CPUs for parallel operations
#SBATCH --job-name=merge_and_split
#SBATCH --output=%x_%j.out           # %x = job name, %j = job ID
#SBATCH --error=%x_%j.err

# SLURM batch script for merging intermediate Parquet files and creating final splits.
# This script is called by submit_merge.sh and processes all intermediate files.
#
# Environment variables expected:
#   INTERMEDIATES_DIR: Directory containing intermediate Parquet files
#   OUTPUT_DIR: Directory for final output Parquet files
#   EMBEDDINGS_DIR: Directory containing per-tar embedding Parquet files
#   PROJECT_ROOT: Path to project root directory (optional, will auto-detect if not set)
#   VENV_PATH: Path to Python virtual environment (optional, if using venv)
#   PYTHON_MODULE: Python module to load (e.g., python/3.12)
#   ARROW_MODULE: Arrow module to load for PyArrow (optional, e.g., arrow/17.0.0)
#   TRAIN_RATIO: Training set ratio (default: 0.7)
#   VAL_RATIO: Validation set ratio (default: 0.15)
#   TEST_RATIO: Test set ratio (default: 0.15)
#   SEED: Random seed for splits (default: 42)

set -euo pipefail

# Load shared functions
source "${PROJECT_ROOT}/slurm/common.sh"

# Log job information
echo "=========================================="
echo "Merge and Split Job Information"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

# Step 1: Setup environment
load_python_env "${VENV_PATH:-}" "${PYTHON_MODULE:-}" "${ARROW_MODULE:-}"

# Step 2: Validate required environment variables
if [ -z "${INTERMEDIATES_DIR:-}" ]; then
    error_exit "INTERMEDIATES_DIR not set"
fi

if [ -z "${OUTPUT_DIR:-}" ]; then
    error_exit "OUTPUT_DIR not set"
fi

if [ -z "${EMBEDDINGS_DIR:-}" ]; then
    error_exit "EMBEDDINGS_DIR not set"
fi

echo "Intermediates directory: ${INTERMEDIATES_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Embeddings directory: ${EMBEDDINGS_DIR}"

# Step 3: Check if intermediates directory exists and contains files
if [ ! -d "${INTERMEDIATES_DIR}" ]; then
    error_exit "Intermediates directory not found: ${INTERMEDIATES_DIR}"
fi

PARQUET_COUNT=$(find "${INTERMEDIATES_DIR}" -name "*.parquet" -type f | wc -l)
if [ "${PARQUET_COUNT}" -eq 0 ]; then
    error_exit "No Parquet files found in ${INTERMEDIATES_DIR}"
fi

echo "Found ${PARQUET_COUNT} intermediate Parquet files"

EMBEDDING_COUNT=$(find "${EMBEDDINGS_DIR}" -name "*_embeddings.parquet" -type f | wc -l)
if [ "${EMBEDDING_COUNT}" -eq 0 ]; then
    error_exit "No embedding Parquet files found in ${EMBEDDINGS_DIR}"
fi

if [ "${EMBEDDING_COUNT}" -ne "${PARQUET_COUNT}" ]; then
    warning "Found ${EMBEDDING_COUNT} embedding files but ${PARQUET_COUNT} intermediate files"
fi

# Step 4: Verify script exists
if [ ! -f "${PROJECT_ROOT}/embedding_pipeline/preprocess/merge_and_split.py" ]; then
    error_exit "merge_and_split.py not found at ${PROJECT_ROOT}/embedding_pipeline/preprocess/merge_and_split.py"
fi

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
echo "Project root: ${PROJECT_ROOT}"

# Step 5: Locate base config file
BASE_CONFIG="${PROJECT_ROOT}/config.toml"
if [ ! -f "${BASE_CONFIG}" ]; then
    error_exit "Base config file not found: ${BASE_CONFIG}"
fi

echo "Using base config file: ${BASE_CONFIG}"

# Step 6: Run merge and split script with base config
echo "Starting merge and split at $(date '+%Y-%m-%d %H:%M:%S')"

python "${PROJECT_ROOT}/embedding_pipeline/preprocess/merge_and_split.py" \
    --config "${BASE_CONFIG}"

EXIT_CODE=$?

# Step 7: Error handling
if [ ${EXIT_CODE} -ne 0 ]; then
    echo "Error: Merge and split failed with exit code ${EXIT_CODE}" >&2
    echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
    exit ${EXIT_CODE}
fi

echo "Merge and split completed successfully at $(date '+%Y-%m-%d %H:%M:%S')"

exit 0
