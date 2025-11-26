#!/bin/bash
# Job submission script for computing visual difficulty metadata on Nibi cluster.
#
# This script validates fingerprint inputs, prepares the Python environment,
# and submits a SLURM job that runs difficulty_metadata/compute_visual_neighbors.py.
#
# Usage:
#   ./submit_visual_neighbors.sh --account <ACCOUNT>
#
# Options:
#   --account <ACCOUNT>        SLURM account name (required, or use SLURM_ACCOUNT env var)
#   --time <TIME>              Time limit (default: 01:00:00)
#   --mem <MEM>                Memory requirement (default: 124G)
#   --cpus <N>                 CPUs to request (default: 16)
#   --dependency <JOB_ID>      Job ID(s) to wait for before starting
#   --venv-path <DIR>          Path to Python virtual environment (default: ~/venv/spatial-building-embeddings)
#   --python-module <MODULE>   Python module to load (default: python/3.11.5)
#   --arrow-module <MODULE>    Arrow module to load (default: arrow/17.0.0)
#   --project-root <DIR>       Path to project root directory (default: auto-detect)
#   --no-venv                  Use system Python instead of creating/using venv
#   --help                     Show this help message
#
# Exit codes mirror other project submit scripts:
#   0: Success
#   1: Invalid arguments or missing required options
#   2: Input validation failure
#   3: Python module not available
#   4: Virtual environment setup failure
#   5: Project root not found
#   6: Job submission failed

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../../slurm/common.sh"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default values
ACCOUNT="${SLURM_ACCOUNT:-}"
TIME="01:00:00"
MEM="124G"
CPUS=16
DEPENDENCY=""
VENV_PATH="${HOME}/venv/spatial-building-embeddings"
PYTHON_MODULE="python/3.11.5"
ARROW_MODULE="arrow/17.0.0"
PROJECT_ROOT=""
NO_VENV=false

show_usage() {
    grep "^# " "${0}" | sed 's/^# //' | head -n 60
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --account) ACCOUNT="$2"; shift 2 ;;
        --time) TIME="$2"; shift 2 ;;
        --mem) MEM="$2"; shift 2 ;;
        --cpus) CPUS="$2"; shift 2 ;;
        --dependency) DEPENDENCY="$2"; shift 2 ;;
        --venv-path) VENV_PATH="$2"; shift 2 ;;
        --python-module) PYTHON_MODULE="$2"; shift 2 ;;
        --arrow-module) ARROW_MODULE="$2"; shift 2 ;;
        --project-root) PROJECT_ROOT="$2"; shift 2 ;;
        --no-venv) NO_VENV=true; shift ;;
        --help) show_usage ;;
        *) error_exit "Unknown option: $1" 1 ;;
    esac
done

if [ -z "${ACCOUNT}" ]; then
    error_exit "--account is required (or set SLURM_ACCOUNT environment variable)" 1
fi

# Step 1: Resolve project root
PROJECT_ROOT=$(resolve_project_root "${PROJECT_ROOT}" "${SCRIPT_DIR}")
if [ -z "${PROJECT_ROOT}" ] || [ ! -f "${PROJECT_ROOT}/pyproject.toml" ]; then
    error_exit "Could not locate project root (pyproject.toml). Use --project-root." 5
fi
info "Project root: ${PROJECT_ROOT}"

# Step 2: Read paths from config.toml
CONFIG_FILE="${PROJECT_ROOT}/config.toml"
INPUT_DIR=$(read_toml_value "${CONFIG_FILE}" "paths" "fingerprints_dir")
OUTPUT_FILE=$(read_toml_value "${CONFIG_FILE}" "paths" "difficulty_metadata_path")

if [ -z "${INPUT_DIR}" ]; then
    error_exit "fingerprints_dir not found in [paths] section of config.toml" 1
fi
if [ -z "${OUTPUT_FILE}" ]; then
    error_exit "difficulty_metadata_path not found in [paths] section of config.toml" 1
fi

info "Input directory (from config): ${INPUT_DIR}"
info "Output file (from config): ${OUTPUT_FILE}"

if [ ! -d "${INPUT_DIR}" ]; then
    error_exit "Input directory not found: ${INPUT_DIR}" 2
fi

PARQUET_COUNT=$(find "${INPUT_DIR}" -maxdepth 1 -type f -name "*.parquet" | wc -l | tr -d ' ')
if [ "${PARQUET_COUNT}" -eq 0 ]; then
    error_exit "No parquet files found in fingerprints directory: ${INPUT_DIR}" 2
else
    info "Found ${PARQUET_COUNT} parquet file(s) under ${INPUT_DIR}"
fi

OUTPUT_PARENT="$(dirname "${OUTPUT_FILE}")"
mkdir -p "${OUTPUT_PARENT}" || error_exit "Failed to create output directory: ${OUTPUT_PARENT}" 2

# Step 3: Setup Python environment
setup_python_env "${PROJECT_ROOT}" "${VENV_PATH}" "${PYTHON_MODULE}" "${ARROW_MODULE}" "${NO_VENV}" "numpy,pandas,rich,sklearn"

# Step 4: Extract log_dir from config.toml
LOG_DIR=$(get_log_dir "${PROJECT_ROOT}" "")
mkdir -p "${LOG_DIR}" || error_exit "Failed to create log directory: ${LOG_DIR}" 2

info "Log directory: ${LOG_DIR}"

# Step 5: Assemble export variables
SBATCH_SCRIPT="${SCRIPT_DIR}/visual_neighbors.sbatch"

# Pass paths to sbatch for logging/checking, but python script uses config
EXPORT_VARS="ALL,INPUT_DATASET_PATH=${INPUT_DIR},OUTPUT_FILE=${OUTPUT_FILE},PYTHON_MODULE=${PYTHON_MODULE},PROJECT_ROOT=${PROJECT_ROOT}"

if [ "${NO_VENV}" = false ] && [ -n "${VENV_PATH:-}" ]; then
    EXPORT_VARS="${EXPORT_VARS},VENV_PATH=${VENV_PATH}"
fi
if [ -n "${ARROW_MODULE:-}" ]; then
    EXPORT_VARS="${EXPORT_VARS},ARROW_MODULE=${ARROW_MODULE}"
fi
EXPORT_VARS="${EXPORT_VARS},LOG_DIR=${LOG_DIR}"

# Step 6: Submit job
SBATCH_CMD=(
    sbatch
    --account="${ACCOUNT}"
    --time="${TIME}"
    --mem="${MEM}"
    --cpus-per-task="${CPUS}"
    --job-name=compute_visual_neighbors
    --output="${LOG_DIR}/visual_neighbors_%j.out"
    --error="${LOG_DIR}/visual_neighbors_%j.err"
    --export="${EXPORT_VARS}"
)

if [ -n "${DEPENDENCY}" ]; then
    SBATCH_CMD+=(--dependency="afterok:${DEPENDENCY}")
    info "Job will wait for job(s): ${DEPENDENCY}"
fi

SBATCH_CMD+=("${SBATCH_SCRIPT}")

SUBMIT_OUTPUT=$("${SBATCH_CMD[@]}" 2>&1) || true

if echo "${SUBMIT_OUTPUT}" | grep -q "Submitted batch job"; then
    JOB_ID=$(echo "${SUBMIT_OUTPUT}" | grep -oE '[0-9]+' | head -n 1)
    echo ""
    echo "=========================================="
    echo "Visual Neighbors Submission Summary"
    echo "=========================================="
    echo "Job ID: ${JOB_ID}"
    echo "Time limit: ${TIME}"
    echo "Memory: ${MEM}"
    echo "CPUs: ${CPUS}"
    echo "Input directory: ${INPUT_DIR}"
    echo "Output file: ${OUTPUT_FILE}"
    echo "Hyperparameters: from config.toml"
    if [ -n "${DEPENDENCY}" ]; then
        echo "Dependency: afterok:${DEPENDENCY}"
    fi
    if [ "${NO_VENV}" = false ] && [ -n "${VENV_PATH}" ]; then
        echo "Virtual environment: ${VENV_PATH}"
    else
        echo "Python: System Python (${PYTHON_MODULE})"
    fi
    echo ""
    echo "Monitor job status:"
    echo "  squeue -j ${JOB_ID}"
    echo ""
    echo "View logs:"
    echo "  tail -f ${LOG_DIR}/visual_neighbors_${JOB_ID}.out"
    echo "  tail -f ${LOG_DIR}/visual_neighbors_${JOB_ID}.err"
    echo "=========================================="
    exit 0
else
    error_exit "Job submission failed: ${SUBMIT_OUTPUT}" 6
fi

