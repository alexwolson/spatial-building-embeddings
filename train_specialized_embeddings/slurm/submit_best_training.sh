#!/bin/bash
# Submit a SLURM job to train with best hyperparameters from WandB.
#
# Usage:
#   ./submit_best_training.sh --account <ACCOUNT> [--config <PATH>] [options]
#
# Options:
#   --account <ACCOUNT>         SLURM account name (required or use SLURM_ACCOUNT)
#   --config <PATH>            Path to TOML config file (default: config.toml)
#   --time <HH:MM:SS>           Wall clock limit (default: 48:00:00)
#   --mem <MEM>                 Memory requirement (default: 256G)
#   --cpus <N>                  CPU count (default: 8)
#   --python-module <MODULE>    Python module to load (default: python/3.12)
#   --venv-path <PATH>          Virtual environment path (default: ~/venv/spatial-building-embeddings)
#   --no-venv                   Use system Python instead of managing a venv
#   --project-root <PATH>       Override project root (defaults to repo root)
#   --log-dir <PATH>            Override log directory (default: train_specialized_embeddings/logs/best_training)
#   --help                      Show this message
#
# Examples:
#   ./submit_best_training.sh --account def-someuser --config config.toml
#   ./submit_best_training.sh --account def-someuser --config /path/to/custom_config.toml
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
    grep "^# " "${0}" | sed 's/^# //' | head -n 100
    exit 0
}

resolve_abs_path() {
    python3 -c 'import os, sys; print(os.path.abspath(sys.argv[1]))' "$1"
}

# Defaults
ACCOUNT="${SLURM_ACCOUNT:-}"
CONFIG_PATH=""
TIME_LIMIT="48:00:00"
MEM="256G"
CPUS="8"
PYTHON_MODULE="python/3.12"
ARROW_MODULE="arrow/17.0.0"
VENV_PATH="${HOME}/venv/spatial-building-embeddings"
PROJECT_ROOT=""
NO_VENV=false
LOG_DIR=""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

while [[ $# -gt 0 ]]; do
    case $1 in
        --account)
            ACCOUNT="$2"
            shift 2
            ;;
        --config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        --time)
            TIME_LIMIT="$2"
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
        --python-module)
            PYTHON_MODULE="$2"
            shift 2
            ;;
        --venv-path)
            VENV_PATH="$2"
            shift 2
            ;;
        --no-venv)
            NO_VENV=true
            shift
            ;;
        --project-root)
            PROJECT_ROOT="$2"
            shift 2
            ;;
        --log-dir)
            LOG_DIR="$2"
            shift 2
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
    error_exit "--account is required (or set SLURM_ACCOUNT)" 1
fi

# Resolve project root
if [ -z "${PROJECT_ROOT}" ]; then
    PROJECT_ROOT="${DEFAULT_PROJECT_ROOT}"
fi
PROJECT_ROOT="$(resolve_abs_path "${PROJECT_ROOT}")"

if [ ! -f "${PROJECT_ROOT}/pyproject.toml" ]; then
    error_exit "Project root does not contain pyproject.toml: ${PROJECT_ROOT}" 5
fi

info "Project root: ${PROJECT_ROOT}"

# Determine default paths
if [ -z "${CONFIG_PATH}" ]; then
    CONFIG_PATH="${PROJECT_ROOT}/config.toml"
fi
CONFIG_PATH="$(resolve_abs_path "${CONFIG_PATH}")"

if [ -z "${LOG_DIR}" ]; then
    LOG_DIR="${PROJECT_ROOT}/train_specialized_embeddings/logs/best_training"
fi
LOG_DIR="$(resolve_abs_path "${LOG_DIR}")"

info "Config file: ${CONFIG_PATH}"
info "Log directory: ${LOG_DIR}"

# Validate config file
if [ ! -f "${CONFIG_PATH}" ]; then
    error_exit "Config file not found: ${CONFIG_PATH}" 2
fi

mkdir -p "${LOG_DIR}"

# Environment setup (venv optional)
if [ "${NO_VENV}" = false ]; then
    if ! module avail "${PYTHON_MODULE}" 2>/dev/null | grep -q "${PYTHON_MODULE}"; then
        error_exit "Python module not available: ${PYTHON_MODULE}" 3
    fi

    if [ -n "${ARROW_MODULE}" ]; then
        info "Loading Arrow module: ${ARROW_MODULE}"
        module load gcc 2>/dev/null || true
        module load "${ARROW_MODULE}" || warning "Failed to load Arrow module - PyArrow may be unavailable"
    fi

    if [ -d "${VENV_PATH}" ]; then
        info "Using existing virtual environment: ${VENV_PATH}"
        source "${VENV_PATH}/bin/activate" || error_exit "Failed to activate virtual environment" 4
        if ! python -c "import pandas, torch, wandb" 2>/dev/null; then
            warning "Key packages missing, reinstalling dependencies..."
            if ! command -v uv >/dev/null 2>&1; then
                info "Installing uv..."
                pip install uv || error_exit "Failed to install uv" 4
            fi
            cd "${PROJECT_ROOT}"
            uv pip install -e . || error_exit "Failed to reinstall project dependencies" 4
            info "Dependencies reinstalled"
        else
            info "Dependencies verified"
        fi
        deactivate
    else
        info "Creating virtual environment: ${VENV_PATH}"
        module load "${PYTHON_MODULE}"
        if ! command -v uv >/dev/null 2>&1; then
            info "Installing uv..."
            pip install uv || error_exit "Failed to install uv" 4
        fi
        uv venv "${VENV_PATH}" || error_exit "Failed to create virtual environment" 4
        source "${VENV_PATH}/bin/activate" || error_exit "Failed to activate virtual environment" 4
        cd "${PROJECT_ROOT}"
        uv pip install -e . || error_exit "Failed to install project dependencies" 4
        deactivate
        info "Virtual environment created and dependencies installed"
    fi
else
    info "Using system Python (--no-venv specified)"
    if ! module avail "${PYTHON_MODULE}" 2>/dev/null | grep -q "${PYTHON_MODULE}"; then
        error_exit "Python module not available: ${PYTHON_MODULE}" 3
    fi
    VENV_PATH=""
fi

# Build sbatch parameters
SBATCH_SCRIPT="${SCRIPT_DIR}/train_best.sbatch"

EXPORT_VARS="ALL,PROJECT_ROOT=${PROJECT_ROOT},PYTHON_MODULE=${PYTHON_MODULE},LOG_DIR=${LOG_DIR},CONFIG_PATH=${CONFIG_PATH}"

if [ -n "${VENV_PATH:-}" ]; then
    EXPORT_VARS="${EXPORT_VARS},VENV_PATH=${VENV_PATH}"
fi

if [ -n "${ARROW_MODULE:-}" ]; then
    EXPORT_VARS="${EXPORT_VARS},ARROW_MODULE=${ARROW_MODULE}"
fi

SUBMIT_OUTPUT=$(sbatch \
    --account="${ACCOUNT}" \
    --time="${TIME_LIMIT}" \
    --mem="${MEM}" \
    --cpus-per-task="${CPUS}" \
    --job-name="best_training" \
    --output="${LOG_DIR}/best_training_%j.out" \
    --error="${LOG_DIR}/best_training_%j.err" \
    --export="${EXPORT_VARS}" \
    "${SBATCH_SCRIPT}" 2>&1)

if echo "${SUBMIT_OUTPUT}" | grep -q "Submitted batch job"; then
    JOB_ID=$(echo "${SUBMIT_OUTPUT}" | grep -oE '[0-9]+' | head -n 1)
    echo ""
    echo "=========================================="
    echo "Best Hyperparameters Training Submission Summary"
    echo "=========================================="
    echo "Job ID: ${JOB_ID}"
    echo "Config file: ${CONFIG_PATH}"
    echo "Time limit: ${TIME_LIMIT}"
    echo "Memory: ${MEM}"
    echo "CPUs: ${CPUS}"
    echo "Log directory: ${LOG_DIR}"
    echo ""
    echo "Monitor job status:"
    echo "  squeue -j ${JOB_ID}"
    echo ""
    echo "View logs:"
    echo "  tail -f ${LOG_DIR}/best_training_${JOB_ID}.out"
    echo "  tail -f ${LOG_DIR}/best_training_${JOB_ID}.err"
    echo "=========================================="
else
    error_exit "Job submission failed: ${SUBMIT_OUTPUT}" 6
fi

