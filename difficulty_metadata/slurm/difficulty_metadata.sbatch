#!/bin/bash
#SBATCH --account=<ACCOUNT>
#SBATCH --time=08:00:00
#SBATCH --mem=220G
#SBATCH --cpus-per-task=32
#SBATCH --job-name=difficulty_metadata
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# SLURM batch script for computing global difficulty metadata.
# Expected environment variables (set by submit_difficulty_metadata.sh):
#   INPUT_DATASET_PATH  Directory containing merged parquet files
#   OUTPUT_FILE         Destination parquet file
#   PROJECT_ROOT        Project root directory (auto-detected if unset)
#   PYTHON_MODULE       Python module to load (e.g., python/3.12)
#   ARROW_MODULE        Optional Arrow module (e.g., arrow/17.0.0)
#   VENV_PATH           Optional virtual environment path
#   Note: All hyperparameters (neighbors, k0_for_local_scale, sample_fraction_for_bands,
#         batch_size, row_group_size, distance_dtype) come from config.toml only
#   LOG_DIR             Directory for log files (optional)

set -euo pipefail

# Load shared functions
source "${PROJECT_ROOT}/slurm/common.sh"

echo "=========================================="
echo "Difficulty Metadata Job Information"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

# Setup environment
load_python_env "${VENV_PATH:-}" "${PYTHON_MODULE:-}" "${ARROW_MODULE:-}"

if [ -z "${INPUT_DATASET_PATH:-}" ]; then
    error_exit "INPUT_DATASET_PATH not set"
fi

if [ -z "${OUTPUT_FILE:-}" ]; then
    error_exit "OUTPUT_FILE not set"
fi

if [ ! -d "${INPUT_DATASET_PATH}" ]; then
    error_exit "Input dataset path not found: ${INPUT_DATASET_PATH}"
fi

mkdir -p "$(dirname "${OUTPUT_FILE}")"

if [ ! -f "${PROJECT_ROOT}/difficulty_metadata/compute_neighbors.py" ]; then
    error_exit "compute_neighbors.py not found at ${PROJECT_ROOT}/difficulty_metadata/compute_neighbors.py"
fi

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
echo "Project root: ${PROJECT_ROOT}"
echo "Input dataset path: ${INPUT_DATASET_PATH}"
echo "Output file: ${OUTPUT_FILE}"

# Locate base config file
BASE_CONFIG="${PROJECT_ROOT}/config.toml"
if [ ! -f "${BASE_CONFIG}" ]; then
    error_exit "Base config file not found: ${BASE_CONFIG}"
fi

echo "Using base config file: ${BASE_CONFIG}"

LOG_FILE="${LOG_DIR:-.}/difficulty_metadata_${SLURM_JOB_ID}.log"

echo "Running difficulty metadata computation at $(date '+%Y-%m-%d %H:%M:%S')"
python "${PROJECT_ROOT}/difficulty_metadata/compute_neighbors.py" --config "${BASE_CONFIG}" | tee "${LOG_FILE}"
EXIT_CODE=${PIPESTATUS[0]}

if [ ${EXIT_CODE} -ne 0 ]; then
    echo "Error: difficulty metadata computation failed with exit code ${EXIT_CODE}" >&2
    echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
    exit ${EXIT_CODE}
fi

echo "Difficulty metadata computation completed successfully at $(date '+%Y-%m-%d %H:%M:%S')"
echo "Log file: ${LOG_FILE}"

exit 0
