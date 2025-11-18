#!/bin/bash
#SBATCH --account=<ACCOUNT>          # Set by submit script via --account
#SBATCH --time=48:00:00              # Default wall time (override in submit script)
#SBATCH --mem=256G                    # Memory requirement
#SBATCH --cpus-per-task=8             # CPU cores
#SBATCH --gres=gpu:1                  # Require a single GPU
#SBATCH --job-name=best_training
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# SLURM batch script that fetches best hyperparameters from WandB and trains final model.
# Expected environment variables (set by submit_best_training.sh):
#   CONFIG_PATH, PROJECT_ROOT
# Optional:
#   VENV_PATH, PYTHON_MODULE, ARROW_MODULE, LOG_DIR

set -euo pipefail

echo "=========================================="
echo "Best Hyperparameters Training Job Information"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Config file: ${CONFIG_PATH}"
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

if [ -z "${CONFIG_PATH:-}" ]; then
    echo "Error: Required environment variable CONFIG_PATH is not set" >&2
    exit 1
fi

if [ ! -f "${CONFIG_PATH}" ]; then
    echo "Error: Config file not found at ${CONFIG_PATH}" >&2
    exit 1
fi

CMD=(
    python "${PROJECT_ROOT}/train_specialized_embeddings/fetch_best_and_train.py"
    --config "${CONFIG_PATH}"
)

echo "Executing command:"
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}"
EXIT_CODE=$?

echo "Job finished with exit code ${EXIT_CODE} at $(date '+%Y-%m-%d %H:%M:%S')"

exit "${EXIT_CODE}"

