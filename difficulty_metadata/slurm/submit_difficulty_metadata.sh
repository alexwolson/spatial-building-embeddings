#!/bin/bash
# Job submission script for computing global difficulty metadata on Nibi cluster.
#
# This script validates merged parquet inputs, prepares the Python environment,
# and submits a SLURM job that runs difficulty_metadata/compute_neighbors.py.
#
# Usage:
#   ./submit_difficulty_metadata.sh --input-dir /path/to/merged --output-file /scratch/user/difficulty_metadata.parquet --account <ACCOUNT>
#
# Options:
#   --input-dir <DIR>          Directory containing merged parquet files (train/val/test) (default: data/merged)
#   --input-train <FILE>       Override path to train parquet (must reside inside --input-dir)
#   --input-val <FILE>         Override path to val parquet (must reside inside --input-dir)
#   --input-test <FILE>        Override path to test parquet (must reside inside --input-dir)
#   --output-file <FILE>       Destination parquet file (default: difficulty_metadata/difficulty_metadata.parquet)
#   --account <ACCOUNT>        SLURM account name (required, or use SLURM_ACCOUNT env var)
#   --time <TIME>              Time limit (default: 08:00:00)
#   --mem <MEM>                Memory requirement (default: 220G)
#   --cpus <N>                 CPUs to request (default: 32)
#   --dependency <JOB_ID>      Job ID(s) to wait for before starting
#   --venv-path <DIR>          Path to Python virtual environment (default: ~/venv/spatial-building-embeddings)
#   Note: All hyperparameters (neighbors, k0, sample-fraction, batch-size, row-group-size,
#         distance-dtype) come from config.toml only
#   --python-module <MODULE>   Python module to load (default: python/3.12)
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

# Default values
INPUT_DIR=""
INPUT_TRAIN=""
INPUT_VAL=""
INPUT_TEST=""
OUTPUT_FILE=""
ACCOUNT="${SLURM_ACCOUNT:-}"
TIME="00:10:00"
MEM="220G"
CPUS=32
DEPENDENCY=""
VENV_PATH="${HOME}/venv/spatial-building-embeddings"
PYTHON_MODULE="python/3.12"
ARROW_MODULE="arrow/17.0.0"
PROJECT_ROOT=""
NO_VENV=false

# Colours for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Colour

error_exit() {
    echo -e "${RED}Error:${NC} $1" >&2
    exit "${2:-1}"
}

info() {
    echo -e "${GREEN}Info:${NC} $1"
}

warning() {
    echo -e "${YELLOW}Warning:${NC} $1"
}

show_usage() {
    grep "^# " "${0}" | sed 's/^# //' | head -n 60
    exit 0
}

abspath() {
    python3 -c 'import os, sys; print(os.path.abspath(sys.argv[1]))' "$1"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input-dir)
            INPUT_DIR="$2"
            shift 2
            ;;
        --input-train)
            INPUT_TRAIN="$2"
            shift 2
            ;;
        --input-val)
            INPUT_VAL="$2"
            shift 2
            ;;
        --input-test)
            INPUT_TEST="$2"
            shift 2
            ;;
        --output-file)
            OUTPUT_FILE="$2"
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

if [ -z "${ACCOUNT}" ]; then
    error_exit "--account is required (or set SLURM_ACCOUNT environment variable)" 1
fi

# Step 1: Resolve project root
if [ -z "${PROJECT_ROOT}" ]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    CANDIDATE="${SCRIPT_DIR}/../.."
    if [ -f "${CANDIDATE}/pyproject.toml" ]; then
        PROJECT_ROOT="${CANDIDATE}"
    elif [ -f "${PWD}/pyproject.toml" ]; then
        PROJECT_ROOT="${PWD}"
    else
        error_exit "Could not locate project root (pyproject.toml). Use --project-root." 5
    fi
fi

PROJECT_ROOT="$(abspath "${PROJECT_ROOT}")"
if [ ! -f "${PROJECT_ROOT}/pyproject.toml" ]; then
    error_exit "Project root does not contain pyproject.toml: ${PROJECT_ROOT}" 5
fi
info "Project root: ${PROJECT_ROOT}"

# Step 2: Determine input directory
if [ -z "${INPUT_DIR}" ]; then
    if [ -z "${INPUT_TRAIN}" ] || [ -z "${INPUT_VAL}" ] || [ -z "${INPUT_TEST}" ]; then
        INPUT_DIR="${PROJECT_ROOT}/data/merged"
    else
        PARENT_DIR="$(dirname "$(abspath "${INPUT_TRAIN}")")"
        for PATH_CAND in "${INPUT_VAL}" "${INPUT_TEST}"; do
            if [ "$(dirname "$(abspath "${PATH_CAND}")")" != "${PARENT_DIR}" ]; then
                error_exit "train/val/test parquet files must reside in the same directory." 2
            fi
        done
        INPUT_DIR="${PARENT_DIR}"
    fi
fi

INPUT_DIR="$(abspath "${INPUT_DIR}")"
if [ ! -d "${INPUT_DIR}" ]; then
    error_exit "Input directory not found: ${INPUT_DIR}" 2
fi

DEFAULT_TRAIN="${INPUT_DIR}/train.parquet"
DEFAULT_VAL="${INPUT_DIR}/val.parquet"
DEFAULT_TEST="${INPUT_DIR}/test.parquet"

INPUT_TRAIN="${INPUT_TRAIN:-${DEFAULT_TRAIN}}"
INPUT_VAL="${INPUT_VAL:-${DEFAULT_VAL}}"
INPUT_TEST="${INPUT_TEST:-${DEFAULT_TEST}}"

for REQUIRED_FILE in "${INPUT_TRAIN}" "${INPUT_VAL}" "${INPUT_TEST}"; do
    if [ ! -f "${REQUIRED_FILE}" ]; then
        error_exit "Expected parquet file not found: ${REQUIRED_FILE}" 2
    fi
done

PARQUET_COUNT=$(find "${INPUT_DIR}" -maxdepth 1 -type f -name "*.parquet" | wc -l | tr -d ' ')
if [ "${PARQUET_COUNT}" -lt 3 ]; then
    warning "Input directory contains ${PARQUET_COUNT} parquet file(s); ensure merged outputs are complete."
else
    info "Found ${PARQUET_COUNT} parquet file(s) under ${INPUT_DIR}"
fi

# Step 3: Resolve output file
if [ -z "${OUTPUT_FILE}" ]; then
    OUTPUT_FILE="${PROJECT_ROOT}/difficulty_metadata/difficulty_metadata.parquet"
fi
OUTPUT_FILE="$(abspath "${OUTPUT_FILE}")"
OUTPUT_PARENT="$(dirname "${OUTPUT_FILE}")"
mkdir -p "${OUTPUT_PARENT}" || error_exit "Failed to create output directory: ${OUTPUT_PARENT}" 2

# Step 4: Note that parameter validation will be done by the Python script
# when it loads config.toml

# Step 5: Setup Python environment
ORIGINAL_PWD="$(pwd)"

if [ "${NO_VENV}" = false ]; then
    if [ -n "${ARROW_MODULE}" ]; then
        info "Loading Arrow module: ${ARROW_MODULE}"
        module load gcc 2>/dev/null || true
        module load "${ARROW_MODULE}" || warning "Failed to load Arrow module - PyArrow may rely on system install"
    fi

    PROJECT_DEPS=(
        "huggingface-hub>=0.36.0,<1.0.0"
        "pandas>=2.0.0"
        "pillow>=10.0.0"
        "pydantic>=2.0.0"
        "pydantic-settings>=2.0.0"
        "rich>=13.0.0"
        "timm>=1.0.22"
        "torch>=2.9.0"
        "torchvision>=0.24.0"
        "scikit-learn>=1.5.0"
    )

    WHEELHOUSE_DIRS=(
        "/cvmfs/soft.computecanada.ca/custom/python/wheelhouse/gentoo2023/x86-64-v4"
        "/cvmfs/soft.computecanada.ca/custom/python/wheelhouse/gentoo2023/x86-64-v3"
        "/cvmfs/soft.computecanada.ca/custom/python/wheelhouse/gentoo2023/generic"
        "/cvmfs/soft.computecanada.ca/custom/python/wheelhouse/generic"
    )

    if [ ! -d "${VENV_PATH}" ]; then
        info "Creating virtual environment: ${VENV_PATH}"
        if ! module avail "${PYTHON_MODULE}" 2>/dev/null | grep -q "${PYTHON_MODULE}"; then
            error_exit "Python module not available: ${PYTHON_MODULE}" 3
        fi
        module load "${PYTHON_MODULE}"
        if ! command -v uv &> /dev/null; then
            info "Installing uv..."
            pip install uv || error_exit "Failed to install uv" 4
        fi
        uv venv "${VENV_PATH}" || error_exit "Failed to create virtual environment" 4
        source "${VENV_PATH}/bin/activate" || error_exit "Failed to activate virtual environment" 4
        cd "${PROJECT_ROOT}"
        FIND_LINK_FLAGS=()
        for WH in "${WHEELHOUSE_DIRS[@]}"; do
            FIND_LINK_FLAGS+=(--find-links "${WH}")
        done
        uv pip install "${FIND_LINK_FLAGS[@]}" "${PROJECT_DEPS[@]}" || error_exit "Failed to install project dependencies" 4
        uv pip install -e . --no-deps || error_exit "Failed to install project in editable mode" 4
        cd "${ORIGINAL_PWD}"
        deactivate
        info "Virtual environment created and dependencies installed"
    else
        info "Virtual environment exists: ${VENV_PATH}"
        info "Verifying dependencies..."
        source "${VENV_PATH}/bin/activate" || error_exit "Failed to activate virtual environment" 4
        if ! python -c "import numpy, pandas, rich, sklearn" 2>/dev/null; then
            warning "Key packages missing, reinstalling..."
            if ! command -v uv &> /dev/null; then
                info "Installing uv..."
                pip install uv || error_exit "Failed to install uv" 4
            fi
            cd "${PROJECT_ROOT}"
            FIND_LINK_FLAGS=()
            for WH in "${WHEELHOUSE_DIRS[@]}"; do
                FIND_LINK_FLAGS+=(--find-links "${WH}")
            done
            uv pip install "${FIND_LINK_FLAGS[@]}" "${PROJECT_DEPS[@]}" || error_exit "Failed to reinstall dependencies" 4
            uv pip install -e . --no-deps || error_exit "Failed to reinstall project in editable mode" 4
            cd "${ORIGINAL_PWD}"
            info "Dependencies reinstalled"
        else
            info "Dependencies verified"
        fi
        deactivate
    fi
else
    info "Using system Python (--no-venv flag set)"
    if ! module avail "${PYTHON_MODULE}" 2>/dev/null | grep -q "${PYTHON_MODULE}"; then
        error_exit "Python module not available: ${PYTHON_MODULE}" 3
    fi
    VENV_PATH=""
fi

# Step 6: Extract log_dir from config.toml
CONFIG_FILE="${PROJECT_ROOT}/config.toml"
if [ ! -f "${CONFIG_FILE}" ]; then
    error_exit "Config file not found: ${CONFIG_FILE}" 1
fi
# Extract log_dir from config.toml (remove quotes and whitespace)
LOG_DIR=$(awk -F'=' '/^log_dir/ {gsub(/^[[:space:]]*["'\'']|["'\'']$/, "", $2); gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2); print $2}' "${CONFIG_FILE}" | head -1)
if [ -z "${LOG_DIR}" ]; then
    error_exit "log_dir not found in config.toml" 1
fi
# Resolve relative path if needed (relative to PROJECT_ROOT)
if [[ "${LOG_DIR}" != /* ]]; then
    LOG_DIR="${PROJECT_ROOT}/${LOG_DIR}"
fi
mkdir -p "${LOG_DIR}" || error_exit "Failed to create log directory: ${LOG_DIR}" 2

info "Input directory: ${INPUT_DIR}"
info "Output file: ${OUTPUT_FILE}"
info "Log directory: ${LOG_DIR}"

# Step 7: Assemble export variables
SBATCH_SCRIPT="${SCRIPT_DIR}/difficulty_metadata.sbatch"

# Note: All hyperparameters come from config.toml only
EXPORT_VARS="ALL,INPUT_DATASET_PATH=${INPUT_DIR},OUTPUT_FILE=${OUTPUT_FILE},PYTHON_MODULE=${PYTHON_MODULE},PROJECT_ROOT=${PROJECT_ROOT}"

if [ -n "${VENV_PATH:-}" ]; then
    EXPORT_VARS="${EXPORT_VARS},VENV_PATH=${VENV_PATH}"
fi
if [ -n "${ARROW_MODULE:-}" ]; then
    EXPORT_VARS="${EXPORT_VARS},ARROW_MODULE=${ARROW_MODULE}"
fi
EXPORT_VARS="${EXPORT_VARS},LOG_DIR=${LOG_DIR}"

# Step 8: Submit job
SBATCH_CMD=(
    sbatch
    --account="${ACCOUNT}"
    --time="${TIME}"
    --mem="${MEM}"
    --cpus-per-task="${CPUS}"
    --job-name=difficulty_metadata
    --output="${LOG_DIR}/difficulty_metadata_%j.out"
    --error="${LOG_DIR}/difficulty_metadata_%j.err"
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
    echo "Difficulty Metadata Submission Summary"
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
    echo "  tail -f ${LOG_DIR}/difficulty_metadata_${JOB_ID}.out"
    echo "  tail -f ${LOG_DIR}/difficulty_metadata_${JOB_ID}.err"
    echo "=========================================="
    exit 0
else
    error_exit "Job submission failed: ${SUBMIT_OUTPUT}" 6
fi


