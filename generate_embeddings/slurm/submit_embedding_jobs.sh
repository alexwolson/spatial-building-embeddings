#!/bin/bash
# Job submission script for generating embeddings on Nibi cluster.
#
# This script discovers intermediate parquet files, sets up Python environment if needed,
# and submits a SLURM job array to generate embeddings in parallel.
#
# Usage:
#   ./submit_embedding_jobs.sh --intermediates-dir /path/to/intermediates --raw-dir /path/to/raw --output-dir /path/to/output --account <ACCOUNT>
#
# Options:
#   --intermediates-dir <DIR>  Directory containing intermediate parquet files (default: data/intermediates)
#   --raw-dir <DIR>            Directory containing tar files (default: data/raw)
#   --output-dir <DIR>         Directory for output embedding parquet files (default: data/embeddings)
#   --account <ACCOUNT>        SLURM account name (required, or use SLURM_ACCOUNT env var)
#   --time <TIME>              Time limit per job (default: 4:00:00)
#   --max-concurrent <N>       Maximum concurrent jobs (default: unlimited, let SLURM decide)
#   --mem-per-cpu <MEM>        Memory per CPU (default: 8G)
#   --model-name <NAME>        Timm model name (default: vit_base_patch14_dinov2.lvd142m)
#   --batch-size <N>           Batch size for inference (default: 128)
#   --resume                    Skip already-processed parquet files
#   --venv-path <PATH>         Path to Python virtual environment (default: ~/venv/spatial-building-embeddings)
#   --python-module <MODULE>   Python module to load (default: python/3.12)
#   --project-root <DIR>       Path to project root directory (default: auto-detect)
#   --no-venv                  Use system Python instead of creating/using venv
#   --test                     Submit only the first parquet file for testing
#   --help                     Show this help message
#
# Examples:
#   # Basic usage with automatic venv creation
#   ./submit_embedding_jobs.sh --intermediates-dir data/intermediates --raw-dir data/raw --output-dir data/embeddings --account <ACCOUNT>
#
#   # Use existing venv
#   ./submit_embedding_jobs.sh --intermediates-dir data/intermediates --raw-dir data/raw --output-dir data/embeddings \
#       --account <ACCOUNT> --venv-path ~/venv/myenv
#
# Exit codes:
#   0: Success
#   1: Invalid arguments or missing required options
#   2: Input directory not found or contains no parquet files
#   3: Python module not available
#   4: Virtual environment creation failed
#   5: Project root not found
#   6: Job submission failed

set -euo pipefail

# Default values
INTERMEDIATES_DIR=""
RAW_DIR=""
OUTPUT_DIR=""
ACCOUNT="${SLURM_ACCOUNT:-}"
TIME="4:00:00"
MAX_CONCURRENT=""  # Empty = unlimited (let SLURM scheduler decide based on resources)
MEM_PER_CPU="8G"
MODEL_NAME="vit_base_patch14_dinov2.lvd142m"
BATCH_SIZE="128"
RESUME=false
VENV_PATH="${HOME}/venv/spatial-building-embeddings"
PYTHON_MODULE="python/3.12"
ARROW_MODULE="arrow/17.0.0"  # Arrow module for PyArrow (required on Alliance clusters)
# Note: huggingface_hub is pinned to <1.0.0 in pyproject.toml to avoid hf-xet dependency
PROJECT_ROOT=""
NO_VENV=false
TEST_MODE=false

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
        --intermediates-dir)
            INTERMEDIATES_DIR="$2"
            shift 2
            ;;
        --raw-dir)
            RAW_DIR="$2"
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
            if [ -z "$2" ] || [ "$2" = "unlimited" ]; then
                MAX_CONCURRENT=""
            else
                MAX_CONCURRENT="$2"
            fi
            shift 2
            ;;
        --mem-per-cpu)
            MEM_PER_CPU="$2"
            shift 2
            ;;
        --model-name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
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
        --test)
            TEST_MODE=true
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
if [ -z "${ACCOUNT}" ]; then
    error_exit " --account is required (or set SLURM_ACCOUNT environment variable)" 1
fi

# Step 1: Detect project root
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

# Step 2: Set default directories if not provided
if [ -z "${INTERMEDIATES_DIR}" ]; then
    INTERMEDIATES_DIR="${PROJECT_ROOT}/data/intermediates"
fi

if [ -z "${RAW_DIR}" ]; then
    RAW_DIR="${PROJECT_ROOT}/data/raw"
fi

if [ -z "${OUTPUT_DIR}" ]; then
    OUTPUT_DIR="${PROJECT_ROOT}/data/embeddings"
fi

# Step 3: Validate input directories
if [ ! -d "${INTERMEDIATES_DIR}" ]; then
    error_exit "Intermediates directory does not exist: ${INTERMEDIATES_DIR}" 2
fi

PARQUET_COUNT=$(find "${INTERMEDIATES_DIR}" -name "*.parquet" -type f | wc -l)
if [ "${PARQUET_COUNT}" -eq 0 ]; then
    error_exit "No .parquet files found in intermediates directory: ${INTERMEDIATES_DIR}" 2
fi

info "Found ${PARQUET_COUNT} parquet file(s) in ${INTERMEDIATES_DIR}"

if [ ! -d "${RAW_DIR}" ]; then
    error_exit "Raw directory does not exist: ${RAW_DIR}" 2
fi

# Step 4: Setup Python environment (if not using --no-venv)
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
        
        # Check for all required dependencies including torch and timm
        if ! python -c "import pandas, rich, pydantic, PIL, torch, timm" 2>/dev/null; then
            warning "Key packages missing, reinstalling dependencies..."
            # Install uv if not available
            if ! command -v uv &> /dev/null; then
                info "Installing uv..."
                pip install uv || error_exit "Failed to install uv" 4
            fi
            cd "${PROJECT_ROOT}"
            # Use Alliance wheelhouse directories if available (same as pip does)
            WHEELHOUSE_DIRS="/cvmfs/soft.computecanada.ca/custom/python/wheelhouse/gentoo2023/x86-64-v4 /cvmfs/soft.computecanada.ca/custom/python/wheelhouse/gentoo2023/x86-64-v3 /cvmfs/soft.computecanada.ca/custom/python/wheelhouse/gentoo2023/generic /cvmfs/soft.computecanada.ca/custom/python/wheelhouse/generic"
            uv pip install -e . --find-links ${WHEELHOUSE_DIRS} || error_exit "Failed to reinstall dependencies" 4
            info "Dependencies reinstalled"
        else
            info "Dependencies verified"
        fi
        deactivate
    else
        info "Creating virtual environment: ${VENV_PATH}"
        
        # Load Python module
        module load "${PYTHON_MODULE}"
        
        # Install uv if not available
        if ! command -v uv &> /dev/null; then
            info "Installing uv..."
            pip install uv || error_exit "Failed to install uv" 4
        fi
        
        # Create venv using uv (Arrow module should already be loaded)
        uv venv "${VENV_PATH}" || error_exit "Failed to create virtual environment" 4
        
        # Activate and install dependencies
        source "${VENV_PATH}/bin/activate" || error_exit "Failed to activate virtual environment" 4
        
        cd "${PROJECT_ROOT}"
        # Use Alliance wheelhouse directories if available (same as pip does)
        WHEELHOUSE_DIRS="/cvmfs/soft.computecanada.ca/custom/python/wheelhouse/gentoo2023/x86-64-v4 /cvmfs/soft.computecanada.ca/custom/python/wheelhouse/gentoo2023/x86-64-v3 /cvmfs/soft.computecanada.ca/custom/python/wheelhouse/gentoo2023/generic /cvmfs/soft.computecanada.ca/custom/python/wheelhouse/generic"
        uv pip install -e . --find-links ${WHEELHOUSE_DIRS} || error_exit "Failed to install project dependencies" 4
        
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

info "Environment setup complete, proceeding to job submission..."

# Step 5: Create output and log directories
mkdir -p "${OUTPUT_DIR}" || error_exit "Failed to create output directory: ${OUTPUT_DIR}" 2
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${PROJECT_ROOT}/generate_embeddings/logs"
mkdir -p "${LOG_DIR}" || error_exit "Failed to create log directory: ${LOG_DIR}" 2

# Verify directories were created
if [ ! -d "${OUTPUT_DIR}" ]; then
    error_exit "Output directory does not exist: ${OUTPUT_DIR}" 2
fi
if [ ! -d "${LOG_DIR}" ]; then
    error_exit "Log directory does not exist: ${LOG_DIR}" 2
fi

info "Intermediates directory: ${INTERMEDIATES_DIR}"
info "Raw directory: ${RAW_DIR}"
info "Output directory: ${OUTPUT_DIR}"
info "Log directory: ${LOG_DIR}"

# Step 6: Generate parquet file list
PARQUET_LIST_FILE="${SCRIPT_DIR}/.parquet_list_$$.txt"
find "${INTERMEDIATES_DIR}" -name "*.parquet" -type f | sort > "${PARQUET_LIST_FILE}"

# Step 7: Count parquet files
NUM_PARQUETS=$(wc -l < "${PARQUET_LIST_FILE}")

# Step 8: If --resume flag, filter out already-processed files
if [ "${RESUME}" = true ]; then
    info "Resume mode: checking for already-processed files..."
    TEMP_LIST="${PARQUET_LIST_FILE}.temp"
    > "${TEMP_LIST}"
    
    PROCESSED=0
    while IFS= read -r PARQUET_FILE; do
        PARQUET_BASENAME=$(basename "${PARQUET_FILE}" .parquet)
        OUTPUT_FILE="${OUTPUT_DIR}/${PARQUET_BASENAME}_embeddings.parquet"
        if [ ! -f "${OUTPUT_FILE}" ]; then
            echo "${PARQUET_FILE}" >> "${TEMP_LIST}"
        else
            ((PROCESSED++)) || true
        fi
    done < "${PARQUET_LIST_FILE}"
    
    mv "${TEMP_LIST}" "${PARQUET_LIST_FILE}"
    NUM_PARQUETS=$(wc -l < "${PARQUET_LIST_FILE}")
    
    if [ "${PROCESSED}" -gt 0 ]; then
        info "Skipping ${PROCESSED} already-processed file(s)"
    fi
fi

if [ "${NUM_PARQUETS}" -eq 0 ]; then
    rm -f "${PARQUET_LIST_FILE}"
    info "No files to process (all already processed or no parquet files found)"
    exit 0
fi

# Step 8.5: If --test mode, limit to first file only
if [ "${TEST_MODE}" = true ]; then
    info "Test mode: limiting to first parquet file only"
    TEMP_LIST="${PARQUET_LIST_FILE}.test"
    head -n 1 "${PARQUET_LIST_FILE}" > "${TEMP_LIST}"
    mv "${TEMP_LIST}" "${PARQUET_LIST_FILE}"
    NUM_PARQUETS=1
fi

info "Submitting job array for ${NUM_PARQUETS} parquet file(s)"

# Step 9: Submit job array
SBATCH_SCRIPT="${SCRIPT_DIR}/generate_embeddings_array.sbatch"

# Build export string - include PROJECT_ROOT and other variables
EXPORT_VARS="ALL,PARQUET_LIST_FILE=${PARQUET_LIST_FILE},RAW_DIR=${RAW_DIR},OUTPUT_DIR=${OUTPUT_DIR},PYTHON_MODULE=${PYTHON_MODULE},LOG_DIR=${LOG_DIR},PROJECT_ROOT=${PROJECT_ROOT},MODEL_NAME=${MODEL_NAME},BATCH_SIZE=${BATCH_SIZE}"
if [ -n "${VENV_PATH:-}" ]; then
    EXPORT_VARS="${EXPORT_VARS},VENV_PATH=${VENV_PATH}"
fi
if [ -n "${ARROW_MODULE:-}" ]; then
    EXPORT_VARS="${EXPORT_VARS},ARROW_MODULE=${ARROW_MODULE}"
fi

# Submit job
# Note: --account, --time, --array, --mem-per-cpu, and log paths are specified on command line and override script defaults
# Build array specification: with or without concurrency limit
if [ -n "${MAX_CONCURRENT}" ]; then
    ARRAY_SPEC="1-${NUM_PARQUETS}%${MAX_CONCURRENT}"
else
    ARRAY_SPEC="1-${NUM_PARQUETS}"  # No limit - let SLURM scheduler decide
fi

SUBMIT_OUTPUT=$(sbatch \
    --account="${ACCOUNT}" \
    --time="${TIME}" \
    --mem-per-cpu="${MEM_PER_CPU}" \
    --array="${ARRAY_SPEC}" \
    --output="${LOG_DIR}/generate_embeddings_%A_%a.out" \
    --error="${LOG_DIR}/generate_embeddings_%A_%a.err" \
    --export="${EXPORT_VARS}" \
    "${SBATCH_SCRIPT}" 2>&1)

# Extract job ID
if echo "${SUBMIT_OUTPUT}" | grep -q "Submitted batch job"; then
    JOB_ID=$(echo "${SUBMIT_OUTPUT}" | grep -oE '[0-9]+' | head -n 1)
    
    # Step 10: Print submission summary
    echo ""
    echo "=========================================="
    echo "Job Submission Summary"
    echo "=========================================="
    echo "Job ID: ${JOB_ID}"
    echo "Number of tasks: ${NUM_PARQUETS}"
    echo "Max concurrent: ${MAX_CONCURRENT:-unlimited}"
    echo "Time limit: ${TIME}"
    echo "Model: ${MODEL_NAME}"
    echo "Batch size: ${BATCH_SIZE}"
    echo "Intermediates directory: ${INTERMEDIATES_DIR}"
    echo "Raw directory: ${RAW_DIR}"
    echo "Output directory: ${OUTPUT_DIR}"
    if [ -n "${VENV_PATH}" ]; then
        echo "Virtual environment: ${VENV_PATH}"
    else
        echo "Python: System Python (${PYTHON_MODULE})"
    fi
    echo "Parquet list file: ${PARQUET_LIST_FILE}"
    echo ""
    echo "Monitor job status:"
    echo "  squeue -j ${JOB_ID}"
    echo ""
    echo "View logs:"
    echo "  tail -f ${LOG_DIR}/generate_embeddings_${JOB_ID}_*.out"
    echo "  tail -f ${LOG_DIR}/generate_embeddings_${JOB_ID}_*.err"
    echo "=========================================="
    
    exit 0
else
    error_exit "Job submission failed: ${SUBMIT_OUTPUT}" 6
fi

