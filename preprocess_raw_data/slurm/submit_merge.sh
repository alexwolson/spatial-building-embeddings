#!/bin/bash
# Job submission script for merging intermediate Parquet files and creating final splits on Nibi cluster.
#
# This script sets up Python environment if needed and submits a SLURM job to merge
# all intermediate Parquet files, filter singletons, create splits, and write final files.
#
# Usage:
#   ./submit_merge.sh --intermediates-dir /path/to/intermediates --output-dir /path/to/output --account <ACCOUNT>
#
# Options:
#   --intermediates-dir <DIR>  Directory containing intermediate Parquet files (required)
#   --output-dir <DIR>         Directory for final output Parquet files (required)
#   --account <ACCOUNT>        SLURM account name (required, or use SLURM_ACCOUNT env var)
#   --time <TIME>              Time limit (default: 12:00:00)
#   --mem <MEM>                Memory requirement (default: 100G)
#   --cpus <N>                 Number of CPUs (default: 8)
#   --dependency <JOB_ID>      Job ID(s) to wait for (optional, for tar preprocessing completion)
#   --seed <N>                 Random seed for splits (default: 42)
#   --train-ratio <RATIO>      Training set ratio (default: 0.7)
#   --val-ratio <RATIO>        Validation set ratio (default: 0.15)
#   --test-ratio <RATIO>       Test set ratio (default: 0.15)
#   --venv-path <PATH>         Path to Python virtual environment (default: ~/venv/spatial-building-embeddings)
#   --python-module <MODULE>   Python module to load (default: python/3.12)
#   --arrow-module <MODULE>    Arrow module to load (default: arrow/17.0.0)
#   --project-root <DIR>       Path to project root directory (default: auto-detect)
#   --no-venv                  Use system Python instead of creating/using venv
#   --help                     Show this help message
#
# Examples:
#   # Basic usage with automatic venv creation
#   ./submit_merge.sh --intermediates-dir /scratch/user/intermediates --output-dir /scratch/user/final --account <ACCOUNT>
#
#   # Wait for tar preprocessing jobs to complete
#   ./submit_merge.sh --intermediates-dir /scratch/user/intermediates --output-dir /scratch/user/final \
#       --account <ACCOUNT> --dependency 12345
#
#   # Custom split ratios
#   ./submit_merge.sh --intermediates-dir /scratch/user/intermediates --output-dir /scratch/user/final \
#       --account <ACCOUNT> --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1
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

# Default values
INTERMEDIATES_DIR=""
OUTPUT_DIR=""
ACCOUNT="${SLURM_ACCOUNT:-}"
TIME="12:00:00"
MEM="100G"
CPUS=8
DEPENDENCY=""
SEED=42
TRAIN_RATIO=0.7
VAL_RATIO=0.15
TEST_RATIO=0.15
VENV_PATH="${HOME}/venv/spatial-building-embeddings"
PYTHON_MODULE="python/3.12"
ARROW_MODULE="arrow/17.0.0"
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
    grep "^# " "${0}" | sed 's/^# //' | head -n 40
    exit 0
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --intermediates-dir)
            INTERMEDIATES_DIR="$2"
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
        --mem)
            MEM="$2"
            shift 2
            ;;
        --cpus)
            CPUS="$2"
            shift 2
            ;;
        --dependency)
            DEPENDENCY="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --train-ratio)
            TRAIN_RATIO="$2"
            shift 2
            ;;
        --val-ratio)
            VAL_RATIO="$2"
            shift 2
            ;;
        --test-ratio)
            TEST_RATIO="$2"
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
        --arrow-module)
            ARROW_MODULE="$2"
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
if [ -z "${INTERMEDIATES_DIR}" ]; then
    error_exit "--intermediates-dir is required" 1
fi

if [ -z "${OUTPUT_DIR}" ]; then
    error_exit "--output-dir is required" 1
fi

if [ -z "${ACCOUNT}" ]; then
    error_exit "--account is required (or set SLURM_ACCOUNT environment variable)" 1
fi

# Validate ratios sum to 1.0 (using Python for floating point comparison)
if ! python3 -c "import sys; r1, r2, r3 = ${TRAIN_RATIO}, ${VAL_RATIO}, ${TEST_RATIO}; sys.exit(0 if abs(r1 + r2 + r3 - 1.0) < 0.0001 else 1)" 2>/dev/null; then
    RATIO_SUM=$(python3 -c "print(${TRAIN_RATIO} + ${VAL_RATIO} + ${TEST_RATIO})")
    error_exit "Ratios must sum to 1.0, got ${RATIO_SUM}" 1
fi

# Step 1: Auto-detect project root if not provided
if [ -z "${PROJECT_ROOT}" ]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    # Look for pyproject.toml starting from script directory
    CURRENT_DIR="${SCRIPT_DIR}/../.."
    if [ -f "${CURRENT_DIR}/pyproject.toml" ]; then
        PROJECT_ROOT="${CURRENT_DIR}"
    elif [ -f "${PWD}/pyproject.toml" ]; then
        PROJECT_ROOT="${PWD}"
    else
        error_exit "Could not find project root (pyproject.toml). Use --project-root to specify." 5
    fi
fi

if [ ! -f "${PROJECT_ROOT}/pyproject.toml" ]; then
    error_exit "Project root does not contain pyproject.toml: ${PROJECT_ROOT}" 5
fi

info "Project root: ${PROJECT_ROOT}"

# Step 2: Validate inputs
if [ ! -d "${INTERMEDIATES_DIR}" ]; then
    error_exit "Intermediates directory not found: ${INTERMEDIATES_DIR}" 2
fi

PARQUET_COUNT=$(find "${INTERMEDIATES_DIR}" -name "*.parquet" -type f | wc -l)
if [ "${PARQUET_COUNT}" -eq 0 ]; then
    error_exit "No Parquet files found in ${INTERMEDIATES_DIR}" 2
fi

info "Found ${PARQUET_COUNT} intermediate Parquet files"

# Check output directory is writable
if [ ! -d "${OUTPUT_DIR}" ]; then
    info "Creating output directory: ${OUTPUT_DIR}"
    mkdir -p "${OUTPUT_DIR}" || error_exit "Failed to create output directory: ${OUTPUT_DIR}" 2
fi

if [ ! -w "${OUTPUT_DIR}" ]; then
    error_exit "Output directory is not writable: ${OUTPUT_DIR}" 2
fi

# Step 3: Setup Python environment (similar to tar preprocessing submission script)
if [ "${NO_VENV}" = false ]; then
    # Load Arrow module if specified (needed for PyArrow)
    if [ -n "${ARROW_MODULE:-}" ]; then
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
        
        if ! python -c "import pandas, pyarrow" 2>/dev/null; then
            warning "Key packages missing, reinstalling..."
            pip install --upgrade pip
            cd "${PROJECT_ROOT}"
            pip install -e .
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
        pip install -e . || error_exit "Failed to install project dependencies" 4
        
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

# Step 4: Create log directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/../logs"
mkdir -p "${LOG_DIR}"

info "Output directory: ${OUTPUT_DIR}"
info "Log directory: ${LOG_DIR}"

# Step 5: Submit job
SBATCH_SCRIPT="${SCRIPT_DIR}/merge_and_split.sbatch"

# Build export string - only include VENV_PATH and ARROW_MODULE if they're set
EXPORT_VARS="ALL,INTERMEDIATES_DIR=${INTERMEDIATES_DIR},OUTPUT_DIR=${OUTPUT_DIR},PYTHON_MODULE=${PYTHON_MODULE},TRAIN_RATIO=${TRAIN_RATIO},VAL_RATIO=${VAL_RATIO},TEST_RATIO=${TEST_RATIO},SEED=${SEED}"
if [ -n "${VENV_PATH:-}" ]; then
    EXPORT_VARS="${EXPORT_VARS},VENV_PATH=${VENV_PATH}"
fi
if [ -n "${ARROW_MODULE:-}" ]; then
    EXPORT_VARS="${EXPORT_VARS},ARROW_MODULE=${ARROW_MODULE}"
fi

# Build sbatch command
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

# Add dependency if specified
if [ -n "${DEPENDENCY}" ]; then
    SBATCH_CMD+=(--dependency="afterok:${DEPENDENCY}")
    info "Job will wait for job(s): ${DEPENDENCY}"
fi

# Add batch script
SBATCH_CMD+=("${SBATCH_SCRIPT}")

# Submit job
SUBMIT_OUTPUT=$("${SBATCH_CMD[@]}" 2>&1)

# Extract job ID
if echo "${SUBMIT_OUTPUT}" | grep -q "Submitted batch job"; then
    JOB_ID=$(echo "${SUBMIT_OUTPUT}" | grep -oE '[0-9]+' | head -n 1)
    
    # Print submission summary
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
    echo "Split ratios: train=${TRAIN_RATIO}, val=${VAL_RATIO}, test=${TEST_RATIO}"
    echo "Random seed: ${SEED}"
    if [ -n "${DEPENDENCY}" ]; then
        echo "Dependency: afterok:${DEPENDENCY}"
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
    echo "  tail -f ${LOG_DIR}/merge_${JOB_ID}.out"
    echo "  tail -f ${LOG_DIR}/merge_${JOB_ID}.err"
    echo "=========================================="
    
    exit 0
else
    error_exit "Job submission failed: ${SUBMIT_OUTPUT}" 6
fi

