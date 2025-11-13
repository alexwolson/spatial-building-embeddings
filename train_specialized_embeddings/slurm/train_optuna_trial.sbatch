#!/bin/bash
#SBATCH --account=<ACCOUNT>          # Set by submit script via --account
#SBATCH --time=24:00:00              # Default wall time (override in submit script)
#SBATCH --mem=64G                    # Memory per worker
#SBATCH --cpus-per-task=8            # CPU cores per worker
#SBATCH --gres=gpu:1                 # Require a single GPU
#SBATCH --job-name=optuna_trial
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err

# SLURM batch script that runs an Optuna worker.
# Expected environment variables (set by submit_optuna_tuning.sh):
#   STUDY_NAME, STORAGE_URL, BASE_CONFIG_PATH, TRIAL_OUTPUT_ROOT
#   TRIALS_PER_WORKER, PROJECT_ROOT
# Optional:
#   VENV_PATH, PYTHON_MODULE, ARROW_MODULE, MAX_EPOCHS, DISABLE_WANDB
#   WANDB_MODE_OVERRIDE, SAVE_EMBEDDINGS, SQLITE_TIMEOUT,
#   OPTUNA_WORKER_VERBOSITY, LOG_DIR

set -euo pipefail

echo "=========================================="
echo "Optuna Worker Information"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID:-N/A}"
echo "Node: ${SLURM_NODELIST}"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Study: ${STUDY_NAME}"
echo "Storage: ${STORAGE_URL}"
echo "Base config: ${BASE_CONFIG_PATH}"
echo "Trial output root: ${TRIAL_OUTPUT_ROOT}"
echo "Trials per worker: ${TRIALS_PER_WORKER}"
echo "=========================================="

module purge

if command -v nvidia-smi &> /dev/null; then
    echo "CUDA/nvidia-smi available"
else
    echo "Error: CUDA not available - worker requires GPU access" >&2
    exit 1
fi

if [ -n "${ARROW_MODULE:-}" ]; then
    echo "Loading Arrow module: ${ARROW_MODULE}"
    module load gcc 2>/dev/null || true
    module load "${ARROW_MODULE}" || echo "Warning: Failed to load Arrow module"
fi

if [ -n "${PYTHON_MODULE:-}" ]; then
    echo "Loading Python module: ${PYTHON_MODULE}"
    module load "${PYTHON_MODULE}"
fi

if [ -n "${VENV_PATH:-}" ]; then
    if [ ! -d "${VENV_PATH}" ]; then
        echo "Error: Virtual environment not found at ${VENV_PATH}" >&2
        exit 1
    fi
    echo "Activating virtual environment: ${VENV_PATH}"
    source "${VENV_PATH}/bin/activate"
else
    echo "Using system Python"
fi

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

for required_var in STUDY_NAME STORAGE_URL BASE_CONFIG_PATH TRIAL_OUTPUT_ROOT TRIALS_PER_WORKER; do
    if [ -z "${!required_var:-}" ]; then
        echo "Error: Required environment variable ${required_var} is not set" >&2
        exit 1
    fi
done

if [ ! -f "${BASE_CONFIG_PATH}" ]; then
    echo "Error: Base config not found at ${BASE_CONFIG_PATH}" >&2
    exit 1
fi

VERBOSITY="${OPTUNA_WORKER_VERBOSITY:-20}"
WORKER_ID="${SLURM_JOB_ID}"
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    WORKER_ID="${WORKER_ID}_${SLURM_ARRAY_TASK_ID}"
fi

CMD=(
    python "${PROJECT_ROOT}/train_specialized_embeddings/optuna_worker.py"
    --study-name "${STUDY_NAME}"
    --storage-url "${STORAGE_URL}"
    --base-config "${BASE_CONFIG_PATH}"
    --trial-output-root "${TRIAL_OUTPUT_ROOT}"
    --trials-per-worker "${TRIALS_PER_WORKER}"
    --worker-id "${WORKER_ID}"
    --verbosity "${VERBOSITY}"
)

if [ -n "${MAX_EPOCHS:-}" ]; then
    CMD+=(--max-epochs "${MAX_EPOCHS}")
fi

if [ "${DISABLE_WANDB:-}" = "true" ]; then
    CMD+=(--disable-wandb)
fi

if [ -n "${WANDB_MODE_OVERRIDE:-}" ]; then
    CMD+=(--wandb-mode "${WANDB_MODE_OVERRIDE}")
fi

if [ "${SAVE_EMBEDDINGS:-}" = "true" ]; then
    CMD+=(--save-embeddings)
fi

if [ -n "${SQLITE_TIMEOUT:-}" ]; then
    CMD+=(--sqlite-timeout "${SQLITE_TIMEOUT}")
fi

echo "Executing command:"
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}"
EXIT_CODE=$?

echo "Worker finished with exit code ${EXIT_CODE} at $(date '+%Y-%m-%d %H:%M:%S')"

exit "${EXIT_CODE}"

