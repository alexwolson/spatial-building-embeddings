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

# Load shared functions
source "${PROJECT_ROOT}/slurm/common.sh"

echo "=========================================="
echo "Best Hyperparameters Training Job Information"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Config file: ${CONFIG_PATH}"
echo "=========================================="

# Setup environment
load_python_env "${VENV_PATH:-}" "${PYTHON_MODULE:-}" "${ARROW_MODULE:-}"

# Load CUDA module (required for GPU)
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA/nvidia-smi available"
else
    error_exit "CUDA not available - worker requires GPU access"
fi

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

if [ -z "${CONFIG_PATH:-}" ]; then
    error_exit "Required environment variable CONFIG_PATH is not set"
fi

if [ ! -f "${CONFIG_PATH}" ]; then
    error_exit "Config file not found at ${CONFIG_PATH}"
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
