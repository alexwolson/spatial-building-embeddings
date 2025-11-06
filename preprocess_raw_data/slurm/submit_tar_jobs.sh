#!/bin/bash
# Job submission script for processing tar files on Nibi cluster.
#
# This script discovers tar files, sets up Python environment if needed,
# and submits a SLURM job array to process them in parallel.
#
# Usage:
#   ./submit_tar_jobs.sh --input-dir /path/to/tars --output-dir /path/to/output --account <ACCOUNT>
#
# Options:
#   --input-dir <DIR>        Directory containing tar files (required)
#   --output-dir <DIR>        Directory for intermediate Parquet files (required)
#   --account <ACCOUNT>       SLURM account name (required, or use SLURM_ACCOUNT env var)
#   --time <TIME>            Time limit per job (default: 2:00:00)
#   --max-concurrent <N>      Maximum concurrent jobs (default: 10)
#   --resume                  Skip already-processed tar files
#   --venv-path <PATH>        Path to Python virtual environment (default: ~/venv/spatial-building-embeddings)
#   --python-module <MODULE>  Python module to load (default: python/3.12)
#   --project-root <DIR>       Path to project root directory (default: auto-detect)
#   --no-venv                 Use system Python instead of creating/using venv
#   --help                    Show this help message
#
# Examples:
#   # Basic usage with automatic venv creation
#   ./submit_tar_jobs.sh --input-dir /scratch/user/tars --output-dir /scratch/user/output --account <ACCOUNT>
#
#   # Use existing venv
#   ./submit_tar_jobs.sh --input-dir /scratch/user/tars --output-dir /scratch/user/output \
#       --account <ACCOUNT> --venv-path ~/venv/myenv
#
#   # Use system Python
#   ./submit_tar_jobs.sh --input-dir /scratch/user/tars --output-dir /scratch/user/output \
#       --account <ACCOUNT> --no-venv
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

# Default values
INPUT_DIR=""
OUTPUT_DIR=""
ACCOUNT="${SLURM_ACCOUNT:-}"
TIME="2:00:00"
MAX_CONCURRENT=10
RESUME=false
VENV_PATH="${HOME}/venv/spatial-building-embeddings"
PYTHON_MODULE="python/3.12"
ARROW_MODULE="arrow/17.0.0"  # Arrow module for PyArrow (required on Alliance clusters)
PROJECT_ROOT=""
NO_VENV=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print error and exit
error_exit() {
    echo -e "${RED}Error:${NC} $1" >&2
    exit "${2:-1}"
}

# Function to print info
info() {
    echo -e "${GREEN}Info:${NC} $1"
}

# Function to print warning
warning() {
    echo -e "${YELLOW}Warning:${NC} $1"
}

# Function to show usage
show_usage() {
    grep "^# " "${0}" | sed 's/^# //' | head -n 30
    exit 0
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input-dir)
            INPUT_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --account)
            ACCOUNT="$2"
            shift 2
            ;;
        --time)
            TIME="$2"
            shift 2
            ;;
        --max-concurrent)
            MAX_CONCURRENT="$2"
            shift 2
            ;;
        --resume)
            RESUME=true
            shift
            ;;
        --venv-path)
            VENV_PATH="$2"
            shift 2
            ;;
        --python-module)
            PYTHON_MODULE="$2"
            shift 2
            ;;
        --project-root)
            PROJECT_ROOT="$2"
            shift 2
            ;;
        --no-venv)
            NO_VENV=true
            shift
            ;;
        --help)
            show_usage
            ;;
        *)
            error_exit "Unknown option: $1" 1
            ;;
    esac
done

# Validate required arguments
if [ -z "${INPUT_DIR}" ]; then
    error_exit " --input-dir is required" 1
fi

if [ -z "${OUTPUT_DIR}" ]; then
    error_exit " --output-dir is required" 1
fi

if [ -z "${ACCOUNT}" ]; then
    error_exit " --account is required (or set SLURM_ACCOUNT environment variable)" 1
fi

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
if [ -z "${PROJECT_ROOT}" ]; then
    # Try to find project root by looking for pyproject.toml
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    CURRENT_DIR="$(pwd)"
    
    # Check script directory and parent
    for dir in "${SCRIPT_DIR}/../.." "${SCRIPT_DIR}/.." "${CURRENT_DIR}"; do
        ABS_DIR="$(cd "${dir}" 2>/dev/null && pwd || echo "")"
        if [ -n "${ABS_DIR}" ] && [ -f "${ABS_DIR}/pyproject.toml" ]; then
            PROJECT_ROOT="${ABS_DIR}"
            break
        fi
    done
    
    if [ -z "${PROJECT_ROOT}" ]; then
        error_exit "Could not find project root (pyproject.toml). Use --project-root to specify." 5
    fi
fi

PROJECT_ROOT="$(cd "${PROJECT_ROOT}" && pwd)"
if [ ! -f "${PROJECT_ROOT}/pyproject.toml" ]; then
    error_exit "Project root does not contain pyproject.toml: ${PROJECT_ROOT}" 5
fi

info "Project root: ${PROJECT_ROOT}"

# Step 3: Setup Python environment (if not using --no-venv)
if [ "${NO_VENV}" = false ]; then
    # Check if Python module is available
    if ! module avail "${PYTHON_MODULE}" 2>/dev/null | grep -q "${PYTHON_MODULE}"; then
        error_exit "Python module not available: ${PYTHON_MODULE}. Check with 'module avail python'" 3
    fi
    
    # Load Arrow module before venv operations (required for PyArrow)
    if [ -n "${ARROW_MODULE}" ]; then
        info "Loading Arrow module: ${ARROW_MODULE}"
        # Load gcc first (required by Arrow module)
        module load gcc 2>/dev/null || true
        module load "${ARROW_MODULE}" || warning "Failed to load Arrow module - PyArrow may not be available"
    fi
    
    # Check if venv exists
    if [ -d "${VENV_PATH}" ]; then
        info "Virtual environment exists: ${VENV_PATH}"
        info "Verifying dependencies..."
        
        # Activate and check for key packages
        source "${VENV_PATH}/bin/activate" || error_exit "Failed to activate virtual environment" 4
        
        # Check for all required dependencies (not just pandas/pyarrow since pyarrow comes from Arrow module)
        if ! python -c "import pandas, rich, pydantic, PIL" 2>/dev/null; then
            warning "Key packages missing, reinstalling dependencies..."
            pip install --upgrade pip
            cd "${PROJECT_ROOT}"
            # Install dependencies directly (pyarrow is provided by Arrow module, so skip it)
            # Use --no-index to prefer Alliance wheelhouse packages and --ignore-installed to force installation in venv
            pip install --no-index --ignore-installed "pandas>=2.3.3" "pillow>=12.0.0" "pydantic>=2.12.4" "pydantic-settings>=2.11.0" "rich>=14.2.0" || error_exit "Failed to reinstall dependencies" 4
            info "Dependencies reinstalled"
        else
            info "Dependencies verified"
        fi
        deactivate
    else
        info "Creating virtual environment: ${VENV_PATH}"
        
        # Load Python module
        module load "${PYTHON_MODULE}"
        
        # Create venv (Arrow module should already be loaded)
        python -m venv "${VENV_PATH}" || error_exit "Failed to create virtual environment" 4
        
        # Activate and install dependencies
        source "${VENV_PATH}/bin/activate" || error_exit "Failed to activate virtual environment" 4
        
        pip install --upgrade pip
        cd "${PROJECT_ROOT}"
        # Install dependencies directly (pyarrow is provided by Arrow module, so skip it)
        # Use --no-index to prefer Alliance wheelhouse packages and --ignore-installed to force installation in venv
        pip install --no-index --ignore-installed "pandas>=2.3.3" "pillow>=12.0.0" "pydantic>=2.12.4" "pydantic-settings>=2.11.0" "rich>=14.2.0" || error_exit "Failed to install project dependencies" 4
        
        deactivate
        
        info "Virtual environment created and dependencies installed"
    fi
else
    info "Using system Python (--no-venv flag set)"
    if ! module avail "${PYTHON_MODULE}" 2>/dev/null | grep -q "${PYTHON_MODULE}"; then
        error_exit "Python module not available: ${PYTHON_MODULE}" 3
    fi
    VENV_PATH=""  # Clear VENV_PATH for system Python
fi

# Step 5: Create output and log directories
mkdir -p "${OUTPUT_DIR}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/../logs"
mkdir -p "${LOG_DIR}"

info "Output directory: ${OUTPUT_DIR}"
info "Log directory: ${LOG_DIR}"

# Step 5: Generate tar file list
TAR_LIST_FILE="${SCRIPT_DIR}/.tar_list_$$.txt"
find "${INPUT_DIR}" -name "*.tar" -type f | sort > "${TAR_LIST_FILE}"

# Step 6: Count tar files
NUM_TARS=$(wc -l < "${TAR_LIST_FILE}")

# Step 8: If --resume flag, filter out already-processed files
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

# Step 9: Submit job array
SBATCH_SCRIPT="${SCRIPT_DIR}/process_tar_array.sbatch"

# Build export string - include PROJECT_ROOT and other variables
EXPORT_VARS="ALL,TAR_LIST_FILE=${TAR_LIST_FILE},OUTPUT_DIR=${OUTPUT_DIR},PYTHON_MODULE=${PYTHON_MODULE},LOG_DIR=${LOG_DIR},PROJECT_ROOT=${PROJECT_ROOT}"
if [ -n "${VENV_PATH:-}" ]; then
    EXPORT_VARS="${EXPORT_VARS},VENV_PATH=${VENV_PATH}"
fi
if [ -n "${ARROW_MODULE:-}" ]; then
    EXPORT_VARS="${EXPORT_VARS},ARROW_MODULE=${ARROW_MODULE}"
fi

# Submit job
# Note: --account, --time, --array, and log paths are specified on command line and override script defaults
SUBMIT_OUTPUT=$(sbatch \
    --account="${ACCOUNT}" \
    --time="${TIME}" \
    --array=1-${NUM_TARS}%${MAX_CONCURRENT} \
    --output="${LOG_DIR}/process_tar_%A_%a.out" \
    --error="${LOG_DIR}/process_tar_%A_%a.err" \
    --export="${EXPORT_VARS}" \
    "${SBATCH_SCRIPT}" 2>&1)

# Extract job ID
if echo "${SUBMIT_OUTPUT}" | grep -q "Submitted batch job"; then
    JOB_ID=$(echo "${SUBMIT_OUTPUT}" | grep -oE '[0-9]+' | head -n 1)
    
    # Step 9: Print submission summary
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
    if [ -n "${VENV_PATH}" ]; then
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

