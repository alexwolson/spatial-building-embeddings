#!/bin/bash
#SBATCH --account=<ACCOUNT>
#SBATCH --time=08:00:00
#SBATCH --mem=220G
#SBATCH --cpus-per-task=32
#SBATCH --job-name=difficulty_metadata
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# SLURM batch script for computing global difficulty metadata.
# Expected environment variables (set by submit_difficulty_metadata.sh):
#   INPUT_DATASET_PATH  Directory containing merged parquet files
#   OUTPUT_FILE         Destination parquet file
#   PROJECT_ROOT        Project root directory (auto-detected if unset)
#   PYTHON_MODULE       Python module to load (e.g., python/3.12)
#   ARROW_MODULE        Optional Arrow module (e.g., arrow/17.0.0)
#   VENV_PATH           Optional virtual environment path
#   NEIGHBORS           Number of neighbours (default 150)
#   K0_FOR_LOCAL_SCALE  Local scale rank (default 50)
#   SAMPLE_FRACTION     Sample fraction for calibration (default 0.03)
#   BATCH_SIZE          BallTree query batch size (default 100000)
#   ROW_GROUP_SIZE      Output parquet row group size (default 50000)
#   DISTANCE_DTYPE      Neighbor distance dtype (float32/float64)
#   LOG_DIR             Directory for log files (optional)

set -euo pipefail

echo "=========================================="
echo "Difficulty Metadata Job Information"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

module purge

if [ -n "${ARROW_MODULE:-}" ]; then
    echo "Loading Arrow module: ${ARROW_MODULE}"
    module load gcc 2>/dev/null || true
    module load "${ARROW_MODULE}" || echo "Warning: Failed to load Arrow module - PyArrow may not be available"
fi

if [ -n "${PYTHON_MODULE:-}" ]; then
    echo "Loading Python module: ${PYTHON_MODULE}"
    module load "${PYTHON_MODULE}"
else
    echo "Warning: PYTHON_MODULE not set, using system Python"
fi

if [ -n "${VENV_PATH:-}" ]; then
    if [ ! -d "${VENV_PATH}" ]; then
        echo "Error: Virtual environment not found at ${VENV_PATH}" >&2
        exit 1
    fi
    echo "Activating virtual environment: ${VENV_PATH}"
    source "${VENV_PATH}/bin/activate"
else
    echo "Using system Python (no virtual environment)"
fi

if [ -z "${INPUT_DATASET_PATH:-}" ]; then
    echo "Error: INPUT_DATASET_PATH not set" >&2
    exit 1
fi

if [ -z "${OUTPUT_FILE:-}" ]; then
    echo "Error: OUTPUT_FILE not set" >&2
    exit 1
fi

if [ ! -d "${INPUT_DATASET_PATH}" ]; then
    if [ -f "${INPUT_DATASET_PATH}" ]; then
        echo "Warning: INPUT_DATASET_PATH points to a single file; expected a directory"
    else
        echo "Error: Input dataset path not found: ${INPUT_DATASET_PATH}" >&2
        exit 1
    fi
fi

mkdir -p "$(dirname "${OUTPUT_FILE}")"

if [ -n "${PROJECT_ROOT:-}" ]; then
    PROJECT_ROOT="$(cd "${PROJECT_ROOT}" && pwd)"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/pyproject.toml" ]; then
    PROJECT_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
    SCRIPT_DIR_SOURCE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(cd "${SCRIPT_DIR_SOURCE}/../.." && pwd)"
fi

if [ ! -f "${PROJECT_ROOT}/difficulty_metadata/compute_neighbors.py" ]; then
    echo "Error: compute_neighbors.py not found at ${PROJECT_ROOT}/difficulty_metadata/compute_neighbors.py" >&2
    exit 1
fi

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
echo "Project root: ${PROJECT_ROOT}"
echo "Input dataset path: ${INPUT_DATASET_PATH}"
echo "Output file: ${OUTPUT_FILE}"

NEIGHBORS="${NEIGHBORS:-150}"
K0_FOR_LOCAL_SCALE="${K0_FOR_LOCAL_SCALE:-50}"
SAMPLE_FRACTION="${SAMPLE_FRACTION:-0.03}"
BATCH_SIZE="${BATCH_SIZE:-100000}"
ROW_GROUP_SIZE="${ROW_GROUP_SIZE:-50000}"
DISTANCE_DTYPE="${DISTANCE_DTYPE:-float32}"

case "${DISTANCE_DTYPE}" in
    float32|float64) ;;
    *)
        echo "Error: DISTANCE_DTYPE must be float32 or float64" >&2
        exit 1
        ;;
esac

CONFIG_DIR="${TMPDIR:-/tmp}"
CONFIG_FILE="${CONFIG_DIR}/difficulty_metadata_${SLURM_JOB_ID}.toml"

cleanup() {
    rm -f "${CONFIG_FILE}"
}
trap cleanup EXIT

cat > "${CONFIG_FILE}" << EOF
[difficulty_metadata]
input_parquet_path = "${INPUT_DATASET_PATH}"
output_parquet_path = "${OUTPUT_FILE}"
neighbors = ${NEIGHBORS}
k0_for_local_scale = ${K0_FOR_LOCAL_SCALE}
sample_fraction_for_bands = ${SAMPLE_FRACTION}
batch_size = ${BATCH_SIZE}
row_group_size = ${ROW_GROUP_SIZE}
distance_dtype = "${DISTANCE_DTYPE}"
EOF

LOG_FILE="${LOG_DIR:-.}/difficulty_metadata_${SLURM_JOB_ID}.log"

echo "Running difficulty metadata computation at $(date '+%Y-%m-%d %H:%M:%S')"
python "${PROJECT_ROOT}/difficulty_metadata/compute_neighbors.py" --config "${CONFIG_FILE}" | tee "${LOG_FILE}"
EXIT_CODE=${PIPESTATUS[0]}

if [ ${EXIT_CODE} -ne 0 ]; then
    echo "Error: difficulty metadata computation failed with exit code ${EXIT_CODE}" >&2
    echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
    exit ${EXIT_CODE}
fi

echo "Difficulty metadata computation completed successfully at $(date '+%Y-%m-%d %H:%M:%S')"
echo "Output written to: ${OUTPUT_FILE}"
echo "Log file: ${LOG_FILE}"

exit 0


