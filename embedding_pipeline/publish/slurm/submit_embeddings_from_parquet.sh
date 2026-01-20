#!/bin/bash
# Job submission script for generating embeddings from packaged_images.parquet on Nibi cluster.
#
# This script sets up Python environment, validates inputs, and submits a SLURM job
# to generate embeddings using the published Spatial Building Embeddings model from HuggingFace Hub.
#
# Usage:
#   ./submit_embeddings_from_parquet.sh --account <ACCOUNT> --input <INPUT> --output <OUTPUT>
#
# Options:
#   --account <ACCOUNT>        SLURM account name (required, or use SLURM_ACCOUNT env var)
#   --input <PATH>             Input parquet file path (required)
#   --output <PATH>            Output parquet/csv file path (required)
#   --model-id <ID>            HuggingFace model ID (optional, defaults to alexwaolson/spatial-building-embeddings or MODEL_ID env var)
#   --batch-size <N>           Model inference batch size (default: 32)
#   --limit <N>                Maximum rows to process (optional, for testing)
#   --time <TIME>              Time limit per job (default: 24:00:00)
#   --mem-per-cpu <MEM>        Memory per CPU (default: 21G)
#   --venv-path <PATH>         Path to Python virtual environment (default: ~/venv/spatial-building-embeddings)
#   --python-module <MODULE>   Python module to load (default: python/3.11.5)
#   --project-root <DIR>       Path to project root directory (default: auto-detect)
#   --no-venv                  Use system Python instead of creating/using venv
#   --help                     Show this help message
#
# GPU Configuration:
#   - Configured for DINOv3 ViT-7B using H100-3g.40gb MIG instance (40GB GPU memory)
#   - Batch size can be adjusted via --batch-size (default: 32)
#   - Using MIG instance: 6.1 RGU cost (vs 12.2 RGU for full GPU = 50% cost reduction)
#   - Resources: 6 CPU cores, 124 GB system memory (recommended per Alliance docs)
#   - See: https://docs.alliancecan.ca/wiki/Multi-Instance_GPU/en
#   - See: https://docs.alliancecan.ca/wiki/Allocations_and_compute_scheduling/en
#
# Environment Variables:
#   HF_TOKEN                   Hugging Face authentication token (required for gated models like DINOv3)
#                              Set this before running the script:
#                                export HF_TOKEN="hf_..."
#                              Or get it from: https://huggingface.co/settings/tokens
#                              The token will be automatically passed to the SLURM job.
#   MODEL_ID                   HuggingFace model ID (optional, can be overridden with --model-id)
#
# Exit codes:
#   0: Success
#   1: Invalid arguments or missing required options
#   2: Input file not found or output directory not writable
#   3: Python module not available
#   4: Virtual environment creation failed
#   5: Project root not found
#   6: Job submission failed

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../../slurm/common.sh"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default values
ACCOUNT="${SLURM_ACCOUNT:-}"
INPUT_PARQUET=""
OUTPUT_FILE=""
MODEL_ID="${MODEL_ID:-alexwaolson/spatial-building-embeddings}"
BATCH_SIZE="32"
LIMIT=""
TIME="24:00:00"
MEM_PER_CPU="21G"
VENV_PATH="${HOME}/venv/spatial-building-embeddings"
PYTHON_MODULE="python/3.11.5"
ARROW_MODULE="arrow/17.0.0"
PROJECT_ROOT=""
NO_VENV=false

show_usage() {
    grep "^# " "${0}" | sed 's/^# //' | head -n 40
    exit 0
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --account) ACCOUNT="$2"; shift 2 ;;
        --input) INPUT_PARQUET="$2"; shift 2 ;;
        --output) OUTPUT_FILE="$2"; shift 2 ;;
        --model-id) MODEL_ID="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --limit) LIMIT="$2"; shift 2 ;;
        --time) TIME="$2"; shift 2 ;;
        --mem-per-cpu) MEM_PER_CPU="$2"; shift 2 ;;
        --venv-path) VENV_PATH="$2"; shift 2 ;;
        --python-module) PYTHON_MODULE="$2"; shift 2 ;;
        --project-root) PROJECT_ROOT="$2"; shift 2 ;;
        --no-venv) NO_VENV=true; shift ;;
        --help) show_usage ;;
        *) error_exit "Unknown option: $1" 1 ;;
    esac
done

# Validate required arguments
if [ -z "${ACCOUNT}" ]; then
    error_exit "--account is required (or set SLURM_ACCOUNT environment variable)" 1
fi

if [ -z "${INPUT_PARQUET}" ]; then
    error_exit "--input is required" 1
fi

if [ -z "${OUTPUT_FILE}" ]; then
    error_exit "--output is required" 1
fi

# Step 1: Validate project root
PROJECT_ROOT=$(resolve_project_root "${PROJECT_ROOT}" "${SCRIPT_DIR}")
if [ -z "${PROJECT_ROOT}" ] || [ ! -f "${PROJECT_ROOT}/pyproject.toml" ]; then
    error_exit "Project root not found (pyproject.toml). Use --project-root." 5
fi
info "Project root: ${PROJECT_ROOT}"

# Step 2: Validate input file
if [ ! -f "${INPUT_PARQUET}" ]; then
    error_exit "Input parquet file not found: ${INPUT_PARQUET}" 2
fi

if [ ! -r "${INPUT_PARQUET}" ]; then
    error_exit "Input parquet file is not readable: ${INPUT_PARQUET}" 2
fi

info "Input parquet file: ${INPUT_PARQUET}"

# Step 3: Validate output directory is writable
OUTPUT_DIR=$(dirname "${OUTPUT_FILE}")
if [ "${OUTPUT_DIR}" != "." ] && [ "${OUTPUT_DIR}" != "$(basename "${OUTPUT_FILE}")" ]; then
    if [ ! -d "${OUTPUT_DIR}" ]; then
        info "Creating output directory: ${OUTPUT_DIR}"
        mkdir -p "${OUTPUT_DIR}" || error_exit "Failed to create output directory: ${OUTPUT_DIR}" 2
    fi
    
    if [ ! -w "${OUTPUT_DIR}" ]; then
        error_exit "Output directory is not writable: ${OUTPUT_DIR}" 2
    fi
fi

info "Output file: ${OUTPUT_FILE}"

# Step 4: Validate HF_TOKEN (warn if not set, as gated models require it)
if [ -z "${HF_TOKEN:-}" ]; then
    warning "HF_TOKEN not set - this may be required for gated models (e.g., DINOv3)"
    warning "Set it with: export HF_TOKEN='hf_...'"
else
    info "HF_TOKEN detected - will be passed to job for Hugging Face authentication"
fi

# Step 5: Setup Python environment
setup_python_env "${PROJECT_ROOT}" "${VENV_PATH}" "${PYTHON_MODULE}" "${ARROW_MODULE}" "${NO_VENV}" "pandas,pyarrow,torch,transformers,PIL"

info "Environment setup complete, proceeding to job submission..."

# Step 6: Get log directory
LOG_DIR=$(get_log_dir "${PROJECT_ROOT}" "")
mkdir -p "${LOG_DIR}" || error_exit "Failed to create log directory: ${LOG_DIR}" 2

info "Log directory: ${LOG_DIR}"

# Step 7: Submit job
SBATCH_SCRIPT="${SCRIPT_DIR}/generate_embeddings_from_parquet.sbatch"

# Build export string
EXPORT_VARS="ALL,INPUT_PARQUET=${INPUT_PARQUET},OUTPUT_FILE=${OUTPUT_FILE},MODEL_ID=${MODEL_ID},BATCH_SIZE=${BATCH_SIZE},PROJECT_ROOT=${PROJECT_ROOT},PYTHON_MODULE=${PYTHON_MODULE},LOG_DIR=${LOG_DIR}"
if [ -n "${LIMIT}" ]; then
    EXPORT_VARS="${EXPORT_VARS},LIMIT=${LIMIT}"
fi
if [ "${NO_VENV}" = false ] && [ -n "${VENV_PATH:-}" ]; then
    EXPORT_VARS="${EXPORT_VARS},VENV_PATH=${VENV_PATH}"
fi
if [ -n "${ARROW_MODULE:-}" ]; then
    EXPORT_VARS="${EXPORT_VARS},ARROW_MODULE=${ARROW_MODULE}"
fi
# Pass through HF_TOKEN if set
if [ -n "${HF_TOKEN:-}" ]; then
    EXPORT_VARS="${EXPORT_VARS},HF_TOKEN=${HF_TOKEN}"
fi

SUBMIT_OUTPUT=$(sbatch \
    --account="${ACCOUNT}" \
    --time="${TIME}" \
    --mem-per-cpu="${MEM_PER_CPU}" \
    --output="${LOG_DIR}/generate_embeddings_from_parquet_%j.out" \
    --error="${LOG_DIR}/generate_embeddings_from_parquet_%j.err" \
    --export="${EXPORT_VARS}" \
    "${SBATCH_SCRIPT}" 2>&1)

if echo "${SUBMIT_OUTPUT}" | grep -q "Submitted batch job"; then
    # Extract the job ID from sbatch output
    JOB_ID=$(echo "${SUBMIT_OUTPUT}" | sed -n 's/.*Submitted batch job \([0-9]\+\).*/\1/p')
    
    if [ -z "${JOB_ID}" ]; then
        # Fallback to old method if sed fails
        JOB_ID=$(echo "${SUBMIT_OUTPUT}" | grep -oE '[0-9]+' | head -n 1)
    fi
    
    echo ""
    echo "=========================================="
    echo "Job Submission Summary"
    echo "=========================================="
    echo "Job ID: ${JOB_ID}"
    echo "Time limit: ${TIME}"
    echo "Input file: ${INPUT_PARQUET}"
    echo "Output file: ${OUTPUT_FILE}"
    echo "Model ID: ${MODEL_ID}"
    echo "Batch size: ${BATCH_SIZE}"
    if [ -n "${LIMIT}" ]; then
        echo "Row limit: ${LIMIT} (testing mode)"
    fi
    if [ "${NO_VENV}" = false ] && [ -n "${VENV_PATH}" ]; then
        echo "Virtual environment: ${VENV_PATH}"
    else
        echo "Python: System Python (${PYTHON_MODULE})"
    fi
    echo ""
    echo "Monitor job status:"
    echo "  squeue -j ${JOB_ID}"
    echo "  squeue -u \$(whoami)"
    echo ""
    echo "View logs:"
    echo "  tail -f ${LOG_DIR}/generate_embeddings_from_parquet_${JOB_ID}.out"
    echo "  tail -f ${LOG_DIR}/generate_embeddings_from_parquet_${JOB_ID}.err"
    echo "=========================================="
    
    exit 0
else
    error_exit "Job submission failed: ${SUBMIT_OUTPUT}" 6
fi
