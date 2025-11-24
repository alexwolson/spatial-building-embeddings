#!/bin/bash
# Job submission script for processing tar files on Nibi cluster.
#
# This script discovers tar files, sets up Python environment if needed,
# and submits a SLURM job array to process them in parallel.
#
# Usage:
#   ./submit_tar_jobs.sh --input-dir /path/to/tars --account <ACCOUNT>
#
# Options:
#   --input-dir <DIR>        Directory containing tar files (required)
#   --account <ACCOUNT>       SLURM account name (required, or use SLURM_ACCOUNT env var)
#   --time <TIME>            Time limit per job (default: 2:00:00)
#   --max-concurrent <N>      Maximum concurrent jobs (default: unlimited, let SLURM decide)
#   --mem-per-cpu <MEM>      Memory per CPU (default: 32G, for large tar files up to 18GB)
#   --resume                  Skip already-processed tar files
#   --venv-path <PATH>        Path to Python virtual environment (default: ~/venv/spatial-building-embeddings)
#   --python-module <MODULE>  Python module to load (default: python/3.12.4)
#   --project-root <DIR>       Path to project root directory (default: auto-detect)
#   --no-venv                 Use system Python instead of creating/using venv
#   --help                    Show this help message
#
# Exit codes:
#   0: Success
#   1: Invalid arguments or missing required options
#   2: Input directory not found or contains no tar files
#   3: Python module not available
#   4: Virtual environment creation failed
#   5: Project root not found
#   6: Job submission failed

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../../slurm/common.sh"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default values
INPUT_DIR=""
ACCOUNT="${SLURM_ACCOUNT:-}"
TIME="2:00:00"
MAX_CONCURRENT=""
MEM_PER_CPU="32G"
RESUME=false
VENV_PATH="${HOME}/venv/spatial-building-embeddings"
PYTHON_MODULE="python/3.12.4"
ARROW_MODULE="arrow/17.0.0"
PROJECT_ROOT=""
NO_VENV=false

show_usage() {
    grep "^# " "${0}" | sed 's/^# //' | head -n 30
    exit 0
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input-dir) INPUT_DIR="$2"; shift 2 ;;
        --account) ACCOUNT="$2"; shift 2 ;;
        --time) TIME="$2"; shift 2 ;;
        --max-concurrent)
            if [ -z "$2" ] || [ "$2" = "unlimited" ]; then
                MAX_CONCURRENT=""
            else
                MAX_CONCURRENT="$2"
            fi
            shift 2
            ;;
        --mem-per-cpu) MEM_PER_CPU="$2"; shift 2 ;;
        --resume) RESUME=true; shift ;;
        --venv-path) VENV_PATH="$2"; shift 2 ;;
        --python-module) PYTHON_MODULE="$2"; shift 2 ;;
        --project-root) PROJECT_ROOT="$2"; shift 2 ;;
        --no-venv) NO_VENV=true; shift ;;
        --help) show_usage ;;
        *) error_exit "Unknown option: $1" 1 ;;
    esac
done

if [ -z "${INPUT_DIR}" ]; then error_exit " --input-dir is required" 1; fi
if [ -z "${ACCOUNT}" ]; then error_exit " --account is required (or set SLURM_ACCOUNT environment variable)" 1; fi

# Step 1: Validate input directory exists and contains .tar files
if [ ! -d "${INPUT_DIR}" ]; then
    error_exit "Input directory does not exist: ${INPUT_DIR}" 2
fi
TAR_COUNT=$(find "${INPUT_DIR}" -name "*.tar" -type f | wc -l)
if [ "${TAR_COUNT}" -eq 0 ]; then
    error_exit "No .tar files found in input directory: ${INPUT_DIR}" 2
fi
info "Found ${TAR_COUNT} tar file(s) in ${INPUT_DIR}"

# Step 2: Detect project root
PROJECT_ROOT=$(resolve_project_root "${PROJECT_ROOT}" "${SCRIPT_DIR}")
if [ -z "${PROJECT_ROOT}" ] || [ ! -f "${PROJECT_ROOT}/pyproject.toml" ]; then
    error_exit "Could not find project root (pyproject.toml). Use --project-root to specify." 5
fi
info "Project root: ${PROJECT_ROOT}"

# Step 3: Read output dir from config
CONFIG_FILE="${PROJECT_ROOT}/config.toml"
if [ ! -f "${CONFIG_FILE}" ]; then
    error_exit "Config file not found: ${CONFIG_FILE}" 1
fi
info "Reading config from: ${CONFIG_FILE}"
set +e
OUTPUT_DIR=$(read_toml_value "${CONFIG_FILE}" "paths" "intermediates_dir" 2>&1)
READ_TOML_EXIT=$?
set -e
if [ ${READ_TOML_EXIT} -ne 0 ]; then
    ERROR_MSG="${OUTPUT_DIR:-unknown error}"
    error_exit "Failed to read intermediates_dir from [paths] section of config.toml: ${ERROR_MSG}" 1
fi
if [ -z "${OUTPUT_DIR}" ]; then 
    error_exit "intermediates_dir is empty in [paths] section of config.toml" 1
fi

# Step 4: Setup Python environment
info "Setting up Python environment..."
setup_python_env "${PROJECT_ROOT}" "${VENV_PATH}" "${PYTHON_MODULE}" "${ARROW_MODULE}" "${NO_VENV}" "pandas,rich,pydantic,PIL" || {
    error_exit "Python environment setup failed" 4
}

info "Environment setup complete, proceeding to job submission..."

# Step 5: Create output and log directories
mkdir -p "${OUTPUT_DIR}" || error_exit "Failed to create output directory: ${OUTPUT_DIR}" 2

set +e
LOG_DIR=$(get_log_dir "${PROJECT_ROOT}" "" 2>&1)
GET_LOG_EXIT=$?
set -e
if [ ${GET_LOG_EXIT} -ne 0 ]; then
    ERROR_MSG="${LOG_DIR:-unknown error}"
    error_exit "Failed to get log directory from config: ${ERROR_MSG}" 1
fi
mkdir -p "${LOG_DIR}" || error_exit "Failed to create log directory: ${LOG_DIR}" 2

info "Output directory: ${OUTPUT_DIR}"
info "Log directory: ${LOG_DIR}"

# Step 6: Generate tar file list
TAR_LIST_FILE="${SCRIPT_DIR}/.tar_list_$$.txt"
find "${INPUT_DIR}" -name "*.tar" -type f | sort > "${TAR_LIST_FILE}"

NUM_TARS=$(wc -l < "${TAR_LIST_FILE}")

# Step 7: If --resume flag, filter out already-processed files
if [ "${RESUME}" = true ]; then
    info "Resume mode: checking for already-processed files..."
    TEMP_LIST="${TAR_LIST_FILE}.temp"
    > "${TEMP_LIST}"
    
    PROCESSED=0
    while IFS= read -r TAR_FILE; do
        TAR_BASENAME=$(basename "${TAR_FILE}")
        OUTPUT_FILE="${OUTPUT_DIR}/${TAR_BASENAME}.parquet"
        if [ ! -f "${OUTPUT_FILE}" ]; then
            echo "${TAR_FILE}" >> "${TEMP_LIST}"
        else
            ((PROCESSED++)) || true
        fi
    done < "${TAR_LIST_FILE}"
    
    mv "${TEMP_LIST}" "${TAR_LIST_FILE}"
    NUM_TARS=$(wc -l < "${TAR_LIST_FILE}")
    
    if [ "${PROCESSED}" -gt 0 ]; then
        info "Skipping ${PROCESSED} already-processed file(s)"
    fi
fi

if [ "${NUM_TARS}" -eq 0 ]; then
    rm -f "${TAR_LIST_FILE}"
    info "No files to process (all already processed or no tar files found)"
    exit 0
fi

info "Submitting job array for ${NUM_TARS} tar file(s)"

# Step 8: Submit job array
SBATCH_SCRIPT="${SCRIPT_DIR}/process_tar_array.sbatch"

EXPORT_VARS="ALL,TAR_LIST_FILE=${TAR_LIST_FILE},OUTPUT_DIR=${OUTPUT_DIR},PYTHON_MODULE=${PYTHON_MODULE},LOG_DIR=${LOG_DIR},PROJECT_ROOT=${PROJECT_ROOT}"
if [ "${NO_VENV}" = false ] && [ -n "${VENV_PATH:-}" ]; then
    EXPORT_VARS="${EXPORT_VARS},VENV_PATH=${VENV_PATH}"
fi
if [ -n "${ARROW_MODULE:-}" ]; then
    EXPORT_VARS="${EXPORT_VARS},ARROW_MODULE=${ARROW_MODULE}"
fi

if [ -n "${MAX_CONCURRENT}" ]; then
    ARRAY_SPEC="1-${NUM_TARS}%${MAX_CONCURRENT}"
else
    ARRAY_SPEC="1-${NUM_TARS}"
fi

SUBMIT_OUTPUT=$(sbatch \
    --account="${ACCOUNT}" \
    --time="${TIME}" \
    --mem-per-cpu="${MEM_PER_CPU}" \
    --array="${ARRAY_SPEC}" \
    --output="${LOG_DIR}/process_tar_%A_%a.out" \
    --error="${LOG_DIR}/process_tar_%A_%a.err" \
    --export="${EXPORT_VARS}" \
    "${SBATCH_SCRIPT}" 2>&1)

if echo "${SUBMIT_OUTPUT}" | grep -q "Submitted batch job"; then
    JOB_ID=$(echo "${SUBMIT_OUTPUT}" | grep -oE '[0-9]+' | head -n 1)
    
    echo ""
    echo "=========================================="
    echo "Job Submission Summary"
    echo "=========================================="
    echo "Job ID: ${JOB_ID}"
    echo "Number of tasks: ${NUM_TARS}"
    echo "Max concurrent: ${MAX_CONCURRENT}"
    echo "Time limit: ${TIME}"
    echo "Input directory: ${INPUT_DIR}"
    echo "Output directory: ${OUTPUT_DIR}"
    if [ "${NO_VENV}" = false ] && [ -n "${VENV_PATH}" ]; then
        echo "Virtual environment: ${VENV_PATH}"
    else
        echo "Python: System Python (${PYTHON_MODULE})"
    fi
    echo "Tar list file: ${TAR_LIST_FILE}"
    echo ""
    echo "Monitor job status:"
    echo "  squeue -j ${JOB_ID}"
    echo ""
    echo "View logs:"
    echo "  tail -f ${LOG_DIR}/process_tar_${JOB_ID}_*.out"
    echo "  tail -f ${LOG_DIR}/process_tar_${JOB_ID}_*.err"
    echo "=========================================="
    exit 0
else
    error_exit "Job submission failed: ${SUBMIT_OUTPUT}" 6
fi
