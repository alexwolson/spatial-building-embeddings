#!/bin/bash
#SBATCH --account=<ACCOUNT>          # Required: Resource Allocation Project (set via submit script)
#SBATCH --time=2:00:00                     # Default: 2 hours (resizing images is fast)
#SBATCH --mem-per-cpu=8G                   # CPU memory: 8GB
#SBATCH --cpus-per-task=1                  # 1 core is sufficient
#SBATCH --array=1-1%10                     # Default: 1 task (overridden by submit script)
#SBATCH --job-name=compute_fingerprints
#SBATCH --output=%x_%A_%a.out  # %x = job name, %A = array job ID, %a = task ID
#SBATCH --error=%x_%A_%a.err

# SLURM batch script for computing fingerprints for a single intermediate parquet file.
# This script is called by submit_fingerprint_jobs.sh and processes one parquet file per task.

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

# Step 2: Get parquet file path from list file using SLURM_ARRAY_TASK_ID
if [ ! -f "${PARQUET_LIST_FILE}" ]; then
    error_exit "Parquet list file not found: ${PARQUET_LIST_FILE}"
fi

PARQUET_FILE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${PARQUET_LIST_FILE}")

if [ -z "${PARQUET_FILE}" ]; then
    error_exit "No parquet file found at task ID ${SLURM_ARRAY_TASK_ID}"
fi

echo "Processing parquet file: ${PARQUET_FILE}"

if [ ! -f "${PROJECT_ROOT}/preprocess_raw_data/compute_fingerprints.py" ]; then
    error_exit "compute_fingerprints.py not found at ${PROJECT_ROOT}/preprocess_raw_data/compute_fingerprints.py"
fi

# Add project root to PYTHONPATH so imports work
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# Locate base config file
BASE_CONFIG="${PROJECT_ROOT}/config.toml"
if [ ! -f "${BASE_CONFIG}" ]; then
    error_exit "Base config file not found: ${BASE_CONFIG}"
fi

# Use SLURM_TMPDIR (node-local storage) for extraction if available
EXTRACT_DIR="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/extract_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${EXTRACT_DIR}"

# Step 3: Run processing script with base config and CLI args
echo "Starting processing at $(date '+%Y-%m-%d %H:%M:%S')"

# Ensure cleanup on exit
cleanup() {
    # Clean up extraction directory if it exists
    if [ -d "${EXTRACT_DIR}" ]; then
        rm -rf "${EXTRACT_DIR}"
    fi
}
trap cleanup EXIT

srun python -m preprocess_raw_data.compute_fingerprints \
    --config "${BASE_CONFIG}" \
    --parquet-file "${PARQUET_FILE}" \
    --temp-dir "${EXTRACT_DIR}"

EXIT_CODE=$?

# Step 4: Error handling
if [ ${EXIT_CODE} -ne 0 ]; then
    echo "Error: Processing failed with exit code ${EXIT_CODE}" >&2
    echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
    exit ${EXIT_CODE}
fi

echo "Processing completed successfully at $(date '+%Y-%m-%d %H:%M:%S')"

exit 0

