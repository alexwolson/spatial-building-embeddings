#!/bin/bash
# Submit a SLURM job array that runs Optuna tuning workers.
#
# Usage:
#   ./submit_optuna_tuning.sh --account <ACCOUNT> [options]
#
# Options:
#   --account <ACCOUNT>         SLURM account name (required or use SLURM_ACCOUNT)
#   --study-name <NAME>         Optuna study name (default: triplet_tuning)
#   --storage-url <URL>         Optuna storage URL. Overrides --storage-path if set.
#   --storage-path <PATH>       Path to SQLite database (default: <project>/train_specialized_embeddings/optuna/optuna.db)
#   --base-config <PATH>        Base training TOML config (default: config.toml)
#   --trial-output-root <PATH>  Directory for per-trial artefacts (default: <project>/train_specialized_embeddings/optuna_trials)
#   --num-workers <N>           Number of workers / job array size (default: 4)
#   --max-concurrent <N>        Maximum concurrent workers (default: unlimited)
#   --trials-per-worker <N>     Number of sequential trials per worker (default: 1)
#   --max-epochs <N>            Override training epochs per trial
#   --disable-wandb             Disable Weights & Biases logging during tuning
#   --wandb-mode <MODE>         Override wandb mode ("online" or "offline")
#   --sqlite-timeout <SECONDS>  SQLite connection timeout (default: 60)
#   --time <HH:MM:SS>           Wall clock limit per worker (default: 24:00:00)
#   --mem <MEM>                 Memory per worker (default: 64G)
#   --cpus <N>                  CPU count per worker (default: 8)
#   --python-module <MODULE>    Python module to load (default: python/3.12)
#   --venv-path <PATH>          Virtual environment path (default: ~/venv/spatial-building-embeddings)
#   --no-venv                   Use system Python instead of managing a venv
#   --project-root <PATH>       Override project root (defaults to repo root)
#   --log-dir <PATH>            Override log directory (default: train_specialized_embeddings/logs/optuna)
#   --verbosity <LEVEL>         Optuna worker logging level (default: 20 / INFO)
#   --help                      Show this message
#
# Examples:
#   ./submit_optuna_tuning.sh --account def-someuser --study-name triplet_sweep
#   ./submit_optuna_tuning.sh --account def-someuser --storage-path /scratch/user/optuna.db --num-workers 16

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
STUDY_NAME="triplet_tuning"
STORAGE_URL=""
STORAGE_PATH=""
BASE_CONFIG_PATH=""
TRIAL_OUTPUT_ROOT=""
NUM_WORKERS=8
MAX_CONCURRENT=""
TRIALS_PER_WORKER=1
MAX_EPOCHS=""
DISABLE_WANDB=false
WANDB_MODE_OVERRIDE=""
SQLITE_TIMEOUT="60"
TIME_LIMIT="01:00:00"
MEM="256G"
CPUS="8"
PYTHON_MODULE="python/3.12"
ARROW_MODULE="arrow/17.0.0"
VENV_PATH="${HOME}/venv/spatial-building-embeddings"
PROJECT_ROOT=""
NO_VENV=false
LOG_DIR=""
OPTUNA_WORKER_VERBOSITY="20"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

while [[ $# -gt 0 ]]; do
    case $1 in
        --account)
            ACCOUNT="$2"
            shift 2
            ;;
        --study-name)
            STUDY_NAME="$2"
            shift 2
            ;;
        --storage-url)
            STORAGE_URL="$2"
            shift 2
            ;;
        --storage-path)
            STORAGE_PATH="$2"
            shift 2
            ;;
        --base-config)
            BASE_CONFIG_PATH="$2"
            shift 2
            ;;
        --trial-output-root)
            TRIAL_OUTPUT_ROOT="$2"
            shift 2
            ;;
        --num-workers)
            NUM_WORKERS="$2"
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
        --trials-per-worker)
            TRIALS_PER_WORKER="$2"
            shift 2
            ;;
        --max-epochs)
            MAX_EPOCHS="$2"
            shift 2
            ;;
        --disable-wandb)
            DISABLE_WANDB=true
            shift
            ;;
        --wandb-mode)
            WANDB_MODE_OVERRIDE="$2"
            shift 2
            ;;
        --sqlite-timeout)
            SQLITE_TIMEOUT="$2"
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
        --verbosity)
            OPTUNA_WORKER_VERBOSITY="$2"
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

if ! [[ "${NUM_WORKERS}" =~ ^[0-9]+$ ]] || [ "${NUM_WORKERS}" -le 0 ]; then
    error_exit "--num-workers must be a positive integer" 1
fi

if ! [[ "${TRIALS_PER_WORKER}" =~ ^[0-9]+$ ]] || [ "${TRIALS_PER_WORKER}" -le 0 ]; then
    error_exit "--trials-per-worker must be a positive integer" 1
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
if [ -z "${BASE_CONFIG_PATH}" ]; then
    BASE_CONFIG_PATH="${PROJECT_ROOT}/config.toml"
fi
BASE_CONFIG_PATH="$(resolve_abs_path "${BASE_CONFIG_PATH}")"

if [ -z "${TRIAL_OUTPUT_ROOT}" ]; then
    TRIAL_OUTPUT_ROOT="${PROJECT_ROOT}/train_specialized_embeddings/optuna_trials"
fi
TRIAL_OUTPUT_ROOT="$(resolve_abs_path "${TRIAL_OUTPUT_ROOT}")"

if [ -z "${LOG_DIR}" ]; then
    LOG_DIR="${PROJECT_ROOT}/train_specialized_embeddings/logs/optuna"
fi
LOG_DIR="$(resolve_abs_path "${LOG_DIR}")"

if [ -z "${STORAGE_URL}" ]; then
    if [ -z "${STORAGE_PATH}" ]; then
        STORAGE_PATH="${PROJECT_ROOT}/train_specialized_embeddings/optuna/optuna.db"
    fi
    STORAGE_PATH="$(resolve_abs_path "${STORAGE_PATH}")"
    if [ -e "${STORAGE_PATH}" ]; then
        warning "Existing Optuna database detected at ${STORAGE_PATH} (new trials will append to this study)"
        warning "If you intended to start fresh, delete or move the file manually before re-running this script."
    fi
    mkdir -p "$(dirname "${STORAGE_PATH}")"
    touch "${STORAGE_PATH}"
    STORAGE_URL="sqlite:///${STORAGE_PATH}"
else
    info "Using provided storage URL (ignoring --storage-path)"
fi

info "Base config: ${BASE_CONFIG_PATH}"
info "Optuna storage URL: ${STORAGE_URL}"
info "Trial output root: ${TRIAL_OUTPUT_ROOT}"
info "Log directory: ${LOG_DIR}"

# Validate base config
if [ ! -f "${BASE_CONFIG_PATH}" ]; then
    error_exit "Base config not found: ${BASE_CONFIG_PATH}" 2
fi

mkdir -p "${TRIAL_OUTPUT_ROOT}"
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
        if ! python -c "import optuna, pandas, torch, wandb" 2>/dev/null; then
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
SBATCH_SCRIPT="${SCRIPT_DIR}/train_optuna_trial.sbatch"
ARRAY_SPEC="1-${NUM_WORKERS}"
if [ -n "${MAX_CONCURRENT}" ]; then
    ARRAY_SPEC="${ARRAY_SPEC}%${MAX_CONCURRENT}"
fi

EXPORT_VARS="ALL,PROJECT_ROOT=${PROJECT_ROOT},PYTHON_MODULE=${PYTHON_MODULE},LOG_DIR=${LOG_DIR},STUDY_NAME=${STUDY_NAME},STORAGE_URL=${STORAGE_URL},BASE_CONFIG_PATH=${BASE_CONFIG_PATH},TRIAL_OUTPUT_ROOT=${TRIAL_OUTPUT_ROOT},TRIALS_PER_WORKER=${TRIALS_PER_WORKER},OPTUNA_WORKER_VERBOSITY=${OPTUNA_WORKER_VERBOSITY},SQLITE_TIMEOUT=${SQLITE_TIMEOUT}"

if [ -n "${VENV_PATH:-}" ]; then
    EXPORT_VARS="${EXPORT_VARS},VENV_PATH=${VENV_PATH}"
fi

if [ -n "${ARROW_MODULE:-}" ]; then
    EXPORT_VARS="${EXPORT_VARS},ARROW_MODULE=${ARROW_MODULE}"
fi

if [ -n "${MAX_EPOCHS:-}" ]; then
    EXPORT_VARS="${EXPORT_VARS},MAX_EPOCHS=${MAX_EPOCHS}"
fi

if [ "${DISABLE_WANDB}" = true ]; then
    EXPORT_VARS="${EXPORT_VARS},DISABLE_WANDB=true"
fi

if [ -n "${WANDB_MODE_OVERRIDE:-}" ]; then
    EXPORT_VARS="${EXPORT_VARS},WANDB_MODE_OVERRIDE=${WANDB_MODE_OVERRIDE}"
fi


SUBMIT_OUTPUT=$(sbatch \
    --account="${ACCOUNT}" \
    --time="${TIME_LIMIT}" \
    --mem="${MEM}" \
    --cpus-per-task="${CPUS}" \
    --job-name="optuna_tuning" \
    --array="${ARRAY_SPEC}" \
    --output="${LOG_DIR}/optuna_worker_%A_%a.out" \
    --error="${LOG_DIR}/optuna_worker_%A_%a.err" \
    --export="${EXPORT_VARS}" \
    "${SBATCH_SCRIPT}" 2>&1)

if echo "${SUBMIT_OUTPUT}" | grep -q "Submitted batch job"; then
    JOB_ID=$(echo "${SUBMIT_OUTPUT}" | grep -oE '[0-9]+' | head -n 1)
    echo ""
    echo "=========================================="
    echo "Optuna Tuning Submission Summary"
    echo "=========================================="
    echo "Job ID: ${JOB_ID}"
    echo "Study name: ${STUDY_NAME}"
    echo "Storage URL: ${STORAGE_URL}"
    echo "Base config: ${BASE_CONFIG_PATH}"
    echo "Trials per worker: ${TRIALS_PER_WORKER}"
    echo "Workers requested: ${NUM_WORKERS}"
    echo "Max concurrent: ${MAX_CONCURRENT:-unlimited}"
    echo "Time limit: ${TIME_LIMIT}"
    echo "Memory: ${MEM}"
    echo "CPUs: ${CPUS}"
    if [ -n "${MAX_EPOCHS:-}" ]; then
        echo "Max epochs per trial: ${MAX_EPOCHS}"
    fi
    echo "WandB enabled: $([ "${DISABLE_WANDB}" = true ] && echo "no" || echo "yes")"
    if [ -n "${WANDB_MODE_OVERRIDE:-}" ]; then
        echo "WandB mode override: ${WANDB_MODE_OVERRIDE}"
    fi
    echo "Trial output directory: ${TRIAL_OUTPUT_ROOT}"
    echo "Log directory: ${LOG_DIR}"
    echo ""
    echo "Monitor job status:"
    echo "  squeue -j ${JOB_ID}"
    echo ""
    echo "View logs:"
    echo "  tail -f ${LOG_DIR}/optuna_worker_${JOB_ID}_*.out"
    echo "  tail -f ${LOG_DIR}/optuna_worker_${JOB_ID}_*.err"
    echo "=========================================="
else
    error_exit "Job submission failed: ${SUBMIT_OUTPUT}" 6
fi

