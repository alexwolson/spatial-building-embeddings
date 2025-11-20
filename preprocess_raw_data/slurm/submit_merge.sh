#!/bin/bash
# Job submission script for merging intermediate Parquet files and creating final splits on Nibi cluster.
#
# This script sets up Python environment if needed and submits a SLURM job to merge
# all intermediate Parquet files, filter singletons, create splits, and write final files.
#
# Usage:
#   ./submit_merge.sh --account <ACCOUNT>
#
# Options:
#   --account <ACCOUNT>        SLURM account name (required, or use SLURM_ACCOUNT env var)
#   --time <TIME>              Time limit (default: 12:00:00)
#   --mem <MEM>                Memory requirement (default: 100G)
#   --cpus <N>                 Number of CPUs (default: 8)
#   --dependency <JOB_ID>      Job ID(s) to wait for (optional, for tar preprocessing completion)
#   Note: train_ratio, val_ratio, test_ratio, and seed come from config.toml only
#   --venv-path <PATH>         Path to Python virtual environment (default: ~/venv/spatial-building-embeddings)
#   --python-module <MODULE>   Python module to load (default: python/3.12)
#   --arrow-module <MODULE>    Arrow module to load (default: arrow/17.0.0)
#   --project-root <DIR>       Path to project root directory (default: auto-detect)
#   --no-venv                  Use system Python instead of creating/using venv
#   --help                     Show this help message
#
# Exit codes:
#   0: Success
#   1: Invalid arguments or missing required options
#   2: Intermediates directory not found or contains no Parquet files
#   3: Python module not available
#   4: Virtual environment creation failed
#   5: Project root not found
#   6: Job submission failed

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../../slurm/common.sh"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default values
ACCOUNT="${SLURM_ACCOUNT:-}"
TIME="00:30:00"
MEM="100G"
CPUS=32
DEPENDENCY=""
VENV_PATH="${HOME}/venv/spatial-building-embeddings"
PYTHON_MODULE="python/3.12"
ARROW_MODULE="arrow/17.0.0"
PROJECT_ROOT=""
NO_VENV=false

show_usage() {
    grep "^# " "${0}" | sed 's/^# //' | head -n 40
    exit 0
}

# Parse command-line arguments
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

if [ -z "${ACCOUNT}" ]; then error_exit "--account is required (or set SLURM_ACCOUNT environment variable)" 1; fi

# Step 1: Auto-detect project root
PROJECT_ROOT=$(resolve_project_root "${PROJECT_ROOT}" "${SCRIPT_DIR}")
if [ -z "${PROJECT_ROOT}" ] || [ ! -f "${PROJECT_ROOT}/pyproject.toml" ]; then
    error_exit "Could not find project root (pyproject.toml). Use --project-root to specify." 5
fi
info "Project root: ${PROJECT_ROOT}"

# Step 2: Read paths from config.toml
CONFIG_FILE="${PROJECT_ROOT}/config.toml"
INTERMEDIATES_DIR=$(read_toml_value "${CONFIG_FILE}" "paths" "intermediates_dir")
OUTPUT_DIR=$(read_toml_value "${CONFIG_FILE}" "paths" "merged_dir")
EMBEDDINGS_DIR=$(read_toml_value "${CONFIG_FILE}" "paths" "embeddings_dir")

if [ -z "${INTERMEDIATES_DIR}" ]; then error_exit "intermediates_dir not found in [paths] section of config.toml" 1; fi
if [ -z "${OUTPUT_DIR}" ]; then error_exit "merged_dir not found in [paths] section of config.toml" 1; fi
if [ -z "${EMBEDDINGS_DIR}" ]; then error_exit "embeddings_dir not found in [paths] section of config.toml" 1; fi

# Step 3: Validate inputs
if [ ! -d "${INTERMEDIATES_DIR}" ]; then
    error_exit "Intermediates directory not found: ${INTERMEDIATES_DIR}" 2
fi
PARQUET_COUNT=$(find "${INTERMEDIATES_DIR}" -name "*.parquet" -type f | wc -l)
if [ "${PARQUET_COUNT}" -eq 0 ]; then
    error_exit "No Parquet files found in ${INTERMEDIATES_DIR}" 2
fi
info "Found ${PARQUET_COUNT} intermediate Parquet files in ${INTERMEDIATES_DIR}"

# Check output directory
if [ ! -d "${OUTPUT_DIR}" ]; then
    info "Creating output directory: ${OUTPUT_DIR}"
    mkdir -p "${OUTPUT_DIR}" || error_exit "Failed to create output directory: ${OUTPUT_DIR}" 2
fi
if [ ! -w "${OUTPUT_DIR}" ]; then
    error_exit "Output directory is not writable: ${OUTPUT_DIR}" 2
fi

if [ ! -d "${EMBEDDINGS_DIR}" ]; then
    error_exit "Embeddings directory not found: ${EMBEDDINGS_DIR}" 2
fi
EMBEDDING_COUNT=$(find "${EMBEDDINGS_DIR}" -name "*_embeddings.parquet" -type f | wc -l)
if [ "${EMBEDDING_COUNT}" -eq 0 ]; then
    error_exit "No embedding Parquet files found in ${EMBEDDINGS_DIR}" 2
fi

if [ "${EMBEDDING_COUNT}" -ne "${PARQUET_COUNT}" ]; then
    warning "Found ${EMBEDDING_COUNT} embedding files but ${PARQUET_COUNT} intermediate files"
fi

# Step 4: Setup Python environment
setup_python_env "${PROJECT_ROOT}" "${VENV_PATH}" "${PYTHON_MODULE}" "${ARROW_MODULE}" "${NO_VENV}" "pandas,pyarrow"

# Step 5: Extract log_dir from config.toml
LOG_DIR=$(get_log_dir "${PROJECT_ROOT}" "")
mkdir -p "${LOG_DIR}" || error_exit "Failed to create log directory: ${LOG_DIR}" 2

info "Output directory: ${OUTPUT_DIR}"
info "Log directory: ${LOG_DIR}"
info "Embeddings directory: ${EMBEDDINGS_DIR}"

# Step 6: Submit job
SBATCH_SCRIPT="${SCRIPT_DIR}/merge_and_split.sbatch"

EXPORT_VARS="ALL,INTERMEDIATES_DIR=${INTERMEDIATES_DIR},OUTPUT_DIR=${OUTPUT_DIR},PYTHON_MODULE=${PYTHON_MODULE},EMBEDDINGS_DIR=${EMBEDDINGS_DIR},PROJECT_ROOT=${PROJECT_ROOT}"
if [ "${NO_VENV}" = false ] && [ -n "${VENV_PATH:-}" ]; then
    EXPORT_VARS="${EXPORT_VARS},VENV_PATH=${VENV_PATH}"
fi
if [ -n "${ARROW_MODULE:-}" ]; then
    EXPORT_VARS="${EXPORT_VARS},ARROW_MODULE=${ARROW_MODULE}"
fi

SBATCH_CMD=(
    sbatch
    --account="${ACCOUNT}"
    --time="${TIME}"
    --mem="${MEM}"
    --cpus-per-task="${CPUS}"
    --job-name=merge_and_split
    --output="${LOG_DIR}/merge_%j.out"
    --error="${LOG_DIR}/merge_%j.err"
    --export="${EXPORT_VARS}"
)

if [ -n "${DEPENDENCY}" ]; then
    SBATCH_CMD+=(--dependency="afterok:${DEPENDENCY}")
    info "Job will wait for job(s): ${DEPENDENCY}"
fi

SBATCH_CMD+=("${SBATCH_SCRIPT}")

SUBMIT_OUTPUT=$("${SBATCH_CMD[@]}" 2>&1)

if echo "${SUBMIT_OUTPUT}" | grep -q "Submitted batch job"; then
    JOB_ID=$(echo "${SUBMIT_OUTPUT}" | grep -oE '[0-9]+' | head -n 1)
    echo ""
    echo "=========================================="
    echo "Job Submission Summary"
    echo "=========================================="
    echo "Job ID: ${JOB_ID}"
    echo "Time limit: ${TIME}"
    echo "Memory: ${MEM}"
    echo "CPUs: ${CPUS}"
    echo "Intermediates directory: ${INTERMEDIATES_DIR}"
    echo "Output directory: ${OUTPUT_DIR}"
    echo "Parquet files found: ${PARQUET_COUNT}"
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
    echo "  tail -f ${LOG_DIR}/merge_${JOB_ID}.out"
    echo "  tail -f ${LOG_DIR}/merge_${JOB_ID}.err"
    echo "=========================================="
    exit 0
else
    error_exit "Job submission failed: ${SUBMIT_OUTPUT}" 6
fi
