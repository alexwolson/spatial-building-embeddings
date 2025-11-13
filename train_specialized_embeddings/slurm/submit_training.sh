#!/bin/bash
# Job submission script for training specialized embeddings on SLURM cluster.
#
# This script sets up Python environment if needed and submits a SLURM job
# to train specialized embeddings using triplet loss.
#
# Usage:
#   ./submit_training.sh --train-parquet <PATH> --val-parquet <PATH> --difficulty-metadata <PATH> --checkpoint-dir <DIR> --account <ACCOUNT>
#
# Options:
#   --train-parquet <PATH>        Path to training split parquet file (required)
#   --val-parquet <PATH>          Path to validation split parquet file (required)
#   --difficulty-metadata <PATH>  Path to difficulty_metadata.parquet (required)
#   --checkpoint-dir <DIR>        Directory for saving checkpoints (required)
#   --output-embeddings-dir <DIR> Optional directory for saving final embeddings
#   --config <PATH>                Path to TOML config file (optional, can use env vars)
#   --account <ACCOUNT>            SLURM account name (required, or use SLURM_ACCOUNT env var)
#   --time <TIME>                  Time limit per job (default: 48:00:00)
#   --mem <MEM>                    Memory requirement (default: 64G)
#   --cpus <N>                     Number of CPUs (default: 8)
#   --venv-path <PATH>             Path to Python virtual environment (default: ~/venv/spatial-building-embeddings)
#   --python-module <MODULE>       Python module to load (default: python/3.12)
#   --project-root <DIR>           Path to project root directory (default: auto-detect)
#   --no-venv                      Use system Python instead of creating/using venv
#   --help                         Show this help message
#
# Examples:
#   # Basic usage with automatic venv creation
#   ./submit_training.sh --train-parquet data/merged/train.parquet \
#       --val-parquet data/merged/val.parquet \
#       --difficulty-metadata data/difficulty/difficulty_metadata.parquet \
#       --checkpoint-dir checkpoints/triplet \
#       --account <ACCOUNT>
#
#   # Use existing venv
#   ./submit_training.sh --train-parquet data/merged/train.parquet \
#       --val-parquet data/merged/val.parquet \
#       --difficulty-metadata data/difficulty/difficulty_metadata.parquet \
#       --checkpoint-dir checkpoints/triplet \
#       --account <ACCOUNT> --venv-path ~/venv/myenv
#
# Exit codes:
#   0: Success
#   1: Invalid arguments or missing required options
#   2: Input files not found
#   3: Python module not available
#   4: Virtual environment creation failed
#   5: Project root not found
#   6: Job submission failed

set -euo pipefail

# Default values
TRAIN_PARQUET_PATH=""
VAL_PARQUET_PATH=""
DIFFICULTY_METADATA_PATH=""
CHECKPOINT_DIR=""
OUTPUT_EMBEDDINGS_DIR=""
CONFIG_FILE=""
ACCOUNT="${SLURM_ACCOUNT:-}"
TIME="48:00:00"
MEM="256G"
CPUS="16"
VENV_PATH="${HOME}/venv/spatial-building-embeddings"
PYTHON_MODULE="python/3.12"
ARROW_MODULE="arrow/17.0.0"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
NO_VENV=false

# Colours for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Colour

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
    grep "^# " "${0}" | sed 's/^# //' | head -n 40
    exit 0
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --train-parquet)
            TRAIN_PARQUET_PATH="$2"
            shift 2
            ;;
        --val-parquet)
            VAL_PARQUET_PATH="$2"
            shift 2
            ;;
        --difficulty-metadata)
            DIFFICULTY_METADATA_PATH="$2"
            shift 2
            ;;
        --checkpoint-dir)
            CHECKPOINT_DIR="$2"
            shift 2
            ;;
        --output-embeddings-dir)
            OUTPUT_EMBEDDINGS_DIR="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
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
        --mem)
            MEM="$2"
            shift 2
            ;;
        --cpus)
            CPUS="$2"
            shift 2
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
if [ -z "${ACCOUNT}" ]; then
    error_exit " --account is required (or set SLURM_ACCOUNT environment variable)" 1
fi

if [ -z "${TRAIN_PARQUET_PATH}" ]; then
    error_exit "--train-parquet is required" 1
fi

if [ -z "${VAL_PARQUET_PATH}" ]; then
    error_exit "--val-parquet is required" 1
fi

if [ -z "${DIFFICULTY_METADATA_PATH}" ]; then
    error_exit "--difficulty-metadata is required" 1
fi

if [ -z "${CHECKPOINT_DIR}" ]; then
    error_exit "--checkpoint-dir is required" 1
fi

# Step 1: Validate project root
PROJECT_ROOT="$(cd "${PROJECT_ROOT}" && pwd)"
if [ ! -f "${PROJECT_ROOT}/pyproject.toml" ]; then
    error_exit "Project root does not contain pyproject.toml: ${PROJECT_ROOT}" 5
fi

info "Project root: ${PROJECT_ROOT}"

# Step 2: Validate input files
if [ ! -f "${TRAIN_PARQUET_PATH}" ]; then
    error_exit "Training parquet file not found: ${TRAIN_PARQUET_PATH}" 2
fi

if [ ! -f "${VAL_PARQUET_PATH}" ]; then
    error_exit "Validation parquet file not found: ${VAL_PARQUET_PATH}" 2
fi

if [ ! -f "${DIFFICULTY_METADATA_PATH}" ]; then
    error_exit "Difficulty metadata file not found: ${DIFFICULTY_METADATA_PATH}" 2
fi

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
        
        # Check for all required dependencies including torch
        if ! python -c "import pandas, rich, pydantic, torch, wandb" 2>/dev/null; then
            warning "Key packages missing, reinstalling dependencies..."
            # Install uv if not available
            if ! command -v uv &> /dev/null; then
                info "Installing uv..."
                pip install uv || error_exit "Failed to install uv" 4
            fi
            cd "${PROJECT_ROOT}"
            uv pip install -e . || error_exit "Failed to reinstall dependencies" 4
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

# Step 4: Create checkpoint and log directories
mkdir -p "${CHECKPOINT_DIR}" || error_exit "Failed to create checkpoint directory: ${CHECKPOINT_DIR}" 2
LOG_DIR="${PROJECT_ROOT}/train_specialized_embeddings/logs"
mkdir -p "${LOG_DIR}" || error_exit "Failed to create log directory: ${LOG_DIR}" 2

# Verify directories were created
if [ ! -d "${CHECKPOINT_DIR}" ]; then
    error_exit "Checkpoint directory does not exist: ${CHECKPOINT_DIR}" 2
fi
if [ ! -d "${LOG_DIR}" ]; then
    error_exit "Log directory does not exist: ${LOG_DIR}" 2
fi

info "Training parquet: ${TRAIN_PARQUET_PATH}"
info "Validation parquet: ${VAL_PARQUET_PATH}"
info "Difficulty metadata: ${DIFFICULTY_METADATA_PATH}"
info "Checkpoint directory: ${CHECKPOINT_DIR}"
if [ -n "${OUTPUT_EMBEDDINGS_DIR}" ]; then
    info "Output embeddings directory: ${OUTPUT_EMBEDDINGS_DIR}"
fi
info "Log directory: ${LOG_DIR}"

# Step 5: Submit job
SBATCH_SCRIPT="${SCRIPT_DIR}/train_triplet.sbatch"

# Build export string - include all required variables
EXPORT_VARS="ALL,TRAIN_PARQUET_PATH=${TRAIN_PARQUET_PATH},VAL_PARQUET_PATH=${VAL_PARQUET_PATH},DIFFICULTY_METADATA_PATH=${DIFFICULTY_METADATA_PATH},CHECKPOINT_DIR=${CHECKPOINT_DIR},PYTHON_MODULE=${PYTHON_MODULE},LOG_DIR=${LOG_DIR},PROJECT_ROOT=${PROJECT_ROOT}"

if [ -n "${VENV_PATH:-}" ]; then
    EXPORT_VARS="${EXPORT_VARS},VENV_PATH=${VENV_PATH}"
fi

if [ -n "${ARROW_MODULE:-}" ]; then
    EXPORT_VARS="${EXPORT_VARS},ARROW_MODULE=${ARROW_MODULE}"
fi

if [ -n "${OUTPUT_EMBEDDINGS_DIR:-}" ]; then
    EXPORT_VARS="${EXPORT_VARS},OUTPUT_EMBEDDINGS_DIR=${OUTPUT_EMBEDDINGS_DIR}"
fi

if [ -n "${CONFIG_FILE:-}" ]; then
    EXPORT_VARS="${EXPORT_VARS},CONFIG_FILE=${CONFIG_FILE}"
fi

# Submit job
SUBMIT_OUTPUT=$(sbatch \
    --account="${ACCOUNT}" \
    --time="${TIME}" \
    --mem="${MEM}" \
    --cpus-per-task="${CPUS}" \
    --job-name=train_triplet \
    --output="${LOG_DIR}/train_triplet_%j.out" \
    --error="${LOG_DIR}/train_triplet_%j.err" \
    --export="${EXPORT_VARS}" \
    "${SBATCH_SCRIPT}" 2>&1)

# Extract job ID
if echo "${SUBMIT_OUTPUT}" | grep -q "Submitted batch job"; then
    JOB_ID=$(echo "${SUBMIT_OUTPUT}" | grep -oE '[0-9]+' | head -n 1)
    
    # Step 6: Print submission summary
    echo ""
    echo "=========================================="
    echo "Job Submission Summary"
    echo "=========================================="
    echo "Job ID: ${JOB_ID}"
    echo "Time limit: ${TIME}"
    echo "Memory: ${MEM}"
    echo "CPUs: ${CPUS}"
    echo "Training parquet: ${TRAIN_PARQUET_PATH}"
    echo "Validation parquet: ${VAL_PARQUET_PATH}"
    echo "Difficulty metadata: ${DIFFICULTY_METADATA_PATH}"
    echo "Checkpoint directory: ${CHECKPOINT_DIR}"
    if [ -n "${OUTPUT_EMBEDDINGS_DIR}" ]; then
        echo "Output embeddings directory: ${OUTPUT_EMBEDDINGS_DIR}"
    fi
    if [ -n "${VENV_PATH}" ]; then
        echo "Virtual environment: ${VENV_PATH}"
    else
        echo "Python: System Python (${PYTHON_MODULE})"
    fi
    echo ""
    echo "Monitor job status:"
    echo "  squeue -j ${JOB_ID}"
    echo ""
    echo "View logs:"
    echo "  tail -f ${LOG_DIR}/train_triplet_${JOB_ID}.out"
    echo "  tail -f ${LOG_DIR}/train_triplet_${JOB_ID}.err"
    echo "=========================================="
    
    exit 0
else
    error_exit "Job submission failed: ${SUBMIT_OUTPUT}" 6
fi

