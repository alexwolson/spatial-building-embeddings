#!/bin/bash
#SBATCH --account=<ACCOUNT>
#SBATCH --time=01:00:00
#SBATCH --mem=124G
#SBATCH --cpus-per-task=16
#SBATCH --job-name=compute_visual_neighbors
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# SLURM batch script for computing visual difficulty metadata.

set -euo pipefail

# Load shared functions
source "${PROJECT_ROOT}/slurm/common.sh"

echo "=========================================="
echo "Visual Neighbors Job Information"
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

if [ ! -f "${PROJECT_ROOT}/embedding_pipeline/difficulty/compute_visual_neighbors.py" ]; then
    error_exit "compute_visual_neighbors.py not found at ${PROJECT_ROOT}/embedding_pipeline/difficulty/compute_visual_neighbors.py"
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

LOG_FILE="${LOG_DIR:-.}/visual_neighbors_${SLURM_JOB_ID}.log"

echo "Running visual neighbors computation at $(date '+%Y-%m-%d %H:%M:%S')"
python -m embedding_pipeline.difficulty.compute_visual_neighbors --config "${BASE_CONFIG}" | tee "${LOG_FILE}"
EXIT_CODE=${PIPESTATUS[0]}

if [ ${EXIT_CODE} -ne 0 ]; then
    echo "Error: computation failed with exit code ${EXIT_CODE}" >&2
    echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
    exit ${EXIT_CODE}
fi

echo "Computation completed successfully at $(date '+%Y-%m-%d %H:%M:%S')"
echo "Log file: ${LOG_FILE}"

exit 0

