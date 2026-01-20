#!/bin/bash
#SBATCH --account=<ACCOUNT>          # Required: Resource Allocation Project (set via submit script)
#SBATCH --time=24:00:00                    # Default: 24 hours (model inference takes time)
#SBATCH --mem-per-cpu=21G                   # CPU memory: 124GB total / 6 cores (recommended for H100-3g.40gb on Nibi)
#SBATCH --cpus-per-task=6                  # Recommended for H100-3g.40gb MIG instance on Nibi
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1  # MIG instance: 40GB GPU memory, 6.1 RGU (50% cost vs full GPU)
#SBATCH --job-name=generate_embeddings_from_parquet
#SBATCH --output=%x_%j.out  # %x = job name, %j = job ID
#SBATCH --error=%x_%j.err
#
# GPU Configuration for DINOv3 ViT-7B:
#   - Using H100-3g.40gb MIG instance (40GB GPU memory, sufficient for 7B model)
#   - RGU cost: 6.1 RGU (vs 12.2 RGU for full H100-80gb = 50% cost reduction)
#   - Recommended resources per Alliance docs: 6 cores, 124 GB system memory
#   - See: https://docs.alliancecan.ca/wiki/Multi-Instance_GPU/en
#   - See: https://docs.alliancecan.ca/wiki/Allocations_and_compute_scheduling/en

# SLURM batch script for generating embeddings from packaged_images.parquet using
# the published Spatial Building Embeddings model from HuggingFace Hub.
# This script is called by submit_embeddings_from_parquet.sh.
#
# Environment variables expected:
#   INPUT_PARQUET: Path to input parquet file
#   OUTPUT_FILE: Path to output file
#   MODEL_ID: HuggingFace model ID (optional, defaults to alexwaolson/spatial-building-embeddings)
#   BATCH_SIZE: Batch size for inference (optional, default: 32)
#   LIMIT: Maximum rows to process (optional, for testing)
#   PROJECT_ROOT: Path to project root directory
#   VENV_PATH: Path to Python virtual environment (optional, if using venv)
#   PYTHON_MODULE: Python module to load (e.g., python/3.11.5)
#   ARROW_MODULE: Arrow module to load for PyArrow (optional, e.g., arrow/17.0.0)
#   HF_TOKEN: HuggingFace authentication token (required for gated models)
#   LOG_DIR: Directory for log files (outputs go here if not redirected)

set -euo pipefail

# Load shared functions
source "${PROJECT_ROOT}/slurm/common.sh"

# Log job information
echo "=========================================="
echo "Job Information"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Input file: ${INPUT_PARQUET}"
echo "Output file: ${OUTPUT_FILE}"
echo "Model ID: ${MODEL_ID:-alexwaolson/spatial-building-embeddings}"
echo "Batch size: ${BATCH_SIZE:-32}"
if [ -n "${LIMIT:-}" ]; then
    echo "Row limit: ${LIMIT} (testing mode)"
fi
echo "=========================================="

# Step 1: Setup environment
load_python_env "${VENV_PATH:-}" "${PYTHON_MODULE:-}" "${ARROW_MODULE:-}"

# Step 1.5: Set HuggingFace cache to scratch (to avoid filling ~/.cache on home with 50GB quota)
# HF_HOME controls where all HuggingFace caches are stored (hub, transformers, etc.)
SCRATCH_BASE="${SCRATCH:-${HOME}/scratch}"
HF_CACHE_DIR="${SCRATCH_BASE}/spatial-building-embeddings/huggingface_cache"
mkdir -p "${HF_CACHE_DIR}"
export HF_HOME="${HF_CACHE_DIR}"
export TRANSFORMERS_CACHE="${HF_CACHE_DIR}/transformers"
export HF_HUB_CACHE="${HF_CACHE_DIR}/hub"
info "HuggingFace cache directory: ${HF_HOME}"

# Step 2: Load CUDA module (required for GPU)
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA/nvidia-smi available"
else
    error_exit "CUDA not available - GPU is required"
fi

# Step 3: Verify GPU is available
echo "Verifying GPU availability..."
if ! python -c "import torch; assert torch.cuda.is_available(), 'GPU not available'; print(f'GPU available: {torch.cuda.get_device_name(0)}')" 2>/dev/null; then
    error_exit "GPU is not available - this job requires a GPU"
fi

# Step 4: Validate input file
if [ ! -f "${INPUT_PARQUET}" ]; then
    error_exit "Input parquet file not found: ${INPUT_PARQUET}"
fi

if [ ! -r "${INPUT_PARQUET}" ]; then
    error_exit "Input parquet file is not readable: ${INPUT_PARQUET}"
fi

echo "Processing parquet file: ${INPUT_PARQUET}"

# Step 5: Validate script exists
if [ ! -f "${PROJECT_ROOT}/embedding_pipeline/publish/generate_embeddings_from_parquet.py" ]; then
    error_exit "generate_embeddings_from_parquet.py not found at ${PROJECT_ROOT}/embedding_pipeline/publish/generate_embeddings_from_parquet.py"
fi

# Add project root to PYTHONPATH so imports work
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# Step 6: Validate output directory is writable
OUTPUT_DIR=$(dirname "${OUTPUT_FILE}")
if [ "${OUTPUT_DIR}" != "." ] && [ "${OUTPUT_DIR}" != "$(basename "${OUTPUT_FILE}")" ]; then
    if [ ! -d "${OUTPUT_DIR}" ]; then
        echo "Creating output directory: ${OUTPUT_DIR}"
        mkdir -p "${OUTPUT_DIR}" || error_exit "Failed to create output directory: ${OUTPUT_DIR}"
    fi
    
    if [ ! -w "${OUTPUT_DIR}" ]; then
        error_exit "Output directory is not writable: ${OUTPUT_DIR}"
    fi
fi

# Step 7: Run processing script
echo "Starting processing at $(date '+%Y-%m-%d %H:%M:%S')"

# Build command arguments
CMD_ARGS=(
    "--input" "${INPUT_PARQUET}"
    "--output" "${OUTPUT_FILE}"
    "--model-id" "${MODEL_ID:-alexwaolson/spatial-building-embeddings}"
    "--batch-size" "${BATCH_SIZE:-32}"
    "--device" "cuda"
)

if [ -n "${LIMIT:-}" ]; then
    CMD_ARGS+=("--limit" "${LIMIT}")
fi

# Run the script (unbuffered for real-time log streaming)
python -u "${PROJECT_ROOT}/embedding_pipeline/publish/generate_embeddings_from_parquet.py" "${CMD_ARGS[@]}"

EXIT_CODE=$?

# Step 8: Error handling
if [ ${EXIT_CODE} -ne 0 ]; then
    echo "Error: Processing failed with exit code ${EXIT_CODE}" >&2
    echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
    exit ${EXIT_CODE}
fi

echo "Processing completed successfully at $(date '+%Y-%m-%d %H:%M:%S')"
echo "Output file: ${OUTPUT_FILE}"

# Step 9: Verify output was created
# The embedding script now writes a parquet *dataset directory* by default.
# If OUTPUT_FILE exists as a file (e.g. legacy output), the script will write to a sidecar
# directory named "<OUTPUT_FILE>.parts".
if [ -d "${OUTPUT_FILE}.parts" ]; then
    OUTPUT_CHECK_PATH="${OUTPUT_FILE}.parts"
else
    OUTPUT_CHECK_PATH="${OUTPUT_FILE}"
fi

if [ ! -e "${OUTPUT_CHECK_PATH}" ]; then
    error_exit "Output path was not created: ${OUTPUT_CHECK_PATH}"
fi

OUTPUT_SIZE=$(du -sh "${OUTPUT_CHECK_PATH}" | cut -f1)
echo "Output size: ${OUTPUT_SIZE}"

exit 0
