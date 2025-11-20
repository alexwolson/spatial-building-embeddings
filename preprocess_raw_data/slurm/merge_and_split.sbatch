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

# Log job information
echo "=========================================="
echo "Merge and Split Job Information"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

# Step 1: Clean environment (best practice)
module purge

# Step 2: Load Arrow module (required for PyArrow, must be loaded before Python/venv)
if [ -n "${ARROW_MODULE:-}" ]; then
    echo "Loading Arrow module: ${ARROW_MODULE}"
    module load gcc 2>/dev/null || true
    module load "${ARROW_MODULE}" || echo "Warning: Failed to load Arrow module - PyArrow may not be available"
fi

# Step 3: Load Python module
if [ -n "${PYTHON_MODULE:-}" ]; then
    echo "Loading Python module: ${PYTHON_MODULE}"
    module load "${PYTHON_MODULE}"
else
    echo "Warning: PYTHON_MODULE not set, using system Python"
fi

# Step 4: Activate Python virtual environment (if using venv)
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

# Step 5: Validate required environment variables
if [ -z "${INTERMEDIATES_DIR:-}" ]; then
    echo "Error: INTERMEDIATES_DIR not set" >&2
    exit 1
fi

if [ -z "${OUTPUT_DIR:-}" ]; then
    echo "Error: OUTPUT_DIR not set" >&2
    exit 1
fi

if [ -z "${EMBEDDINGS_DIR:-}" ]; then
    echo "Error: EMBEDDINGS_DIR not set" >&2
    exit 1
fi

echo "Intermediates directory: ${INTERMEDIATES_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Embeddings directory: ${EMBEDDINGS_DIR}"

# Step 6: Check if intermediates directory exists and contains files
if [ ! -d "${INTERMEDIATES_DIR}" ]; then
    echo "Error: Intermediates directory not found: ${INTERMEDIATES_DIR}" >&2
    exit 1
fi

PARQUET_COUNT=$(find "${INTERMEDIATES_DIR}" -name "*.parquet" -type f | wc -l)
if [ "${PARQUET_COUNT}" -eq 0 ]; then
    echo "Error: No Parquet files found in ${INTERMEDIATES_DIR}" >&2
    exit 1
fi

echo "Found ${PARQUET_COUNT} intermediate Parquet files"

EMBEDDING_COUNT=$(find "${EMBEDDINGS_DIR}" -name "*_embeddings.parquet" -type f | wc -l)
if [ "${EMBEDDING_COUNT}" -eq 0 ]; then
    echo "Error: No embedding Parquet files found in ${EMBEDDINGS_DIR}" >&2
    exit 1
fi

if [ "${EMBEDDING_COUNT}" -ne "${PARQUET_COUNT}" ]; then
    echo "Warning: Found ${EMBEDDING_COUNT} embedding files but ${PARQUET_COUNT} intermediate files"
fi

# Step 7: Resolve project root and configure Python path
if [ -n "${PROJECT_ROOT:-}" ]; then
    PROJECT_ROOT="$(cd "${PROJECT_ROOT}" && pwd)"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/pyproject.toml" ]; then
    PROJECT_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
    SCRIPT_DIR_SOURCE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(cd "${SCRIPT_DIR_SOURCE}/../.." && pwd)"
fi

if [ ! -f "${PROJECT_ROOT}/preprocess_raw_data/merge_and_split.py" ]; then
    echo "Error: merge_and_split.py not found at ${PROJECT_ROOT}/preprocess_raw_data/merge_and_split.py" >&2
    exit 1
fi

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
echo "Project root: ${PROJECT_ROOT}"

# Step 8: Locate base config file
BASE_CONFIG="${PROJECT_ROOT}/config.toml"
if [ ! -f "${BASE_CONFIG}" ]; then
    echo "Error: Base config file not found: ${BASE_CONFIG}" >&2
    exit 1
fi

echo "Using base config file: ${BASE_CONFIG}"

# Step 9: Run merge and split script with base config
echo "Starting merge and split at $(date '+%Y-%m-%d %H:%M:%S')"

python "${PROJECT_ROOT}/preprocess_raw_data/merge_and_split.py" \
    --config "${BASE_CONFIG}"

EXIT_CODE=$?

# Step 10: Error handling
if [ ${EXIT_CODE} -ne 0 ]; then
    echo "Error: Merge and split failed with exit code ${EXIT_CODE}" >&2
    echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
    exit ${EXIT_CODE}
fi

echo "Merge and split completed successfully at $(date '+%Y-%m-%d %H:%M:%S')"

exit 0

