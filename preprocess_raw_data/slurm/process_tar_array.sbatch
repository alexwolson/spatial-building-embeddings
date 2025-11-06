#!/bin/bash
#SBATCH --account=<ACCOUNT>          # Required: Resource Allocation Project (set via submit script)
#SBATCH --time=2:00:00                    # Default: 2 hours (adjustable)
#SBATCH --mem-per-cpu=16G                 # Default for large tar files (up to 18GB) - can be overridden via sbatch --mem-per-cpu
#SBATCH --cpus-per-task=1                 # Serial processing, no parallelization
#SBATCH --array=1-1%10                    # Default: 1 task (overridden by submit script)
#SBATCH --job-name=process_tar_array
#SBATCH --output=%x_%A_%a.out  # %x = job name, %A = array job ID, %a = task ID
#SBATCH --error=%x_%A_%a.err

# SLURM batch script for processing a single tar file as part of a job array.
# This script is called by submit_tar_jobs.sh and processes one tar file per task.
#
# Environment variables expected:
#   TAR_LIST_FILE: Path to file containing list of tar files (one per line)
#   OUTPUT_DIR: Directory for output Parquet files
#   PROJECT_ROOT: Path to project root directory (optional, will auto-detect if not set)
#   VENV_PATH: Path to Python virtual environment (optional, if using venv)
#   PYTHON_MODULE: Python module to load (e.g., python/3.12)
#   ARROW_MODULE: Arrow module to load for PyArrow (optional, e.g., arrow/17.0.0)

set -euo pipefail

# Log job information
echo "=========================================="
echo "Job Array Task Information"
echo "=========================================="
echo "Job ID: ${SLURM_ARRAY_JOB_ID}"
echo "Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

# Step 1: Clean environment (best practice)
module purge

# Step 2: Load Arrow module (required for PyArrow, must be loaded before Python/venv)
if [ -n "${ARROW_MODULE:-}" ]; then
    echo "Loading Arrow module: ${ARROW_MODULE}"
    module load gcc 2>/dev/null || true
    module load "${ARROW_MODULE}" || echo "Warning: Failed to load Arrow module - PyArrow may not be available"
fi

# Step 3: Load Python module
if [ -n "${PYTHON_MODULE:-}" ]; then
    echo "Loading Python module: ${PYTHON_MODULE}"
    module load "${PYTHON_MODULE}"
else
    echo "Warning: PYTHON_MODULE not set, using system Python"
fi

# Step 4: Activate Python virtual environment (if using venv)
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

# Step 4: Get tar file path from list file using SLURM_ARRAY_TASK_ID
if [ ! -f "${TAR_LIST_FILE}" ]; then
    echo "Error: Tar list file not found: ${TAR_LIST_FILE}" >&2
    exit 1
fi

TAR_FILE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${TAR_LIST_FILE}")

if [ -z "${TAR_FILE}" ]; then
    echo "Error: No tar file found at task ID ${SLURM_ARRAY_TASK_ID}" >&2
    exit 1
fi

echo "Processing tar file: ${TAR_FILE}"

# Step 5: Check if output already exists (resume capability)
TAR_BASENAME=$(basename "${TAR_FILE}")
OUTPUT_FILE="${OUTPUT_DIR}/${TAR_BASENAME}.parquet"

if [ -f "${OUTPUT_FILE}" ]; then
    echo "Skipping ${TAR_FILE} - output already exists: ${OUTPUT_FILE}"
    exit 0
fi

# Step 6: Create config file and run processing script
echo "Starting processing at $(date '+%Y-%m-%d %H:%M:%S')"

# Determine project root - use environment variable if set, otherwise auto-detect
if [ -z "${PROJECT_ROOT:-}" ]; then
    # Get the directory where this script is located
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
else
    PROJECT_ROOT="$(cd "${PROJECT_ROOT}" && pwd)"
fi

if [ ! -f "${PROJECT_ROOT}/preprocess_raw_data/process_tar.py" ]; then
    echo "Error: process_tar.py not found at ${PROJECT_ROOT}/preprocess_raw_data/process_tar.py" >&2
    exit 1
fi

# Add project root to PYTHONPATH so imports work
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# Use SLURM_TMPDIR (node-local storage) for extraction if available
# This reduces memory pressure and improves I/O performance for large tar files
EXTRACT_DIR="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/extract_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${EXTRACT_DIR}"

# Create temporary config file in temp directory
CONFIG_DIR="${TMPDIR:-/tmp}"
CONFIG_FILE="${CONFIG_DIR}/process_tar_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.toml"

# Ensure cleanup on exit
cleanup() {
    rm -f "${CONFIG_FILE}"
    # Clean up extraction directory if it exists
    if [ -d "${EXTRACT_DIR}" ]; then
        rm -rf "${EXTRACT_DIR}"
    fi
}
trap cleanup EXIT

cat > "${CONFIG_FILE}" << EOF
[process_tar]
tar_file = "${TAR_FILE}"
output_dir = "${OUTPUT_DIR}"
temp_dir = "${EXTRACT_DIR}"
EOF

python "${PROJECT_ROOT}/preprocess_raw_data/process_tar.py" \
    --config "${CONFIG_FILE}"

EXIT_CODE=$?

# Step 7: Error handling
if [ ${EXIT_CODE} -ne 0 ]; then
    echo "Error: Processing failed with exit code ${EXIT_CODE}" >&2
    echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
    exit ${EXIT_CODE}
fi

echo "Processing completed successfully at $(date '+%Y-%m-%d %H:%M:%S')"
echo "Output file: ${OUTPUT_FILE}"

exit 0

