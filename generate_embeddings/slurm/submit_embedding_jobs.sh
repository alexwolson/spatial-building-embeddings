#!/bin/bash
# Job submission script for generating embeddings on Nibi cluster.
#
# This script discovers intermediate parquet files, sets up Python environment if needed,
# and submits a SLURM job array to generate embeddings in parallel.
#
# Usage:
#   ./submit_embedding_jobs.sh --account <ACCOUNT>
#
# Options:
#   --account <ACCOUNT>        SLURM account name (required, or use SLURM_ACCOUNT env var)
#   --time <TIME>              Time limit per job (default: 4:00:00)
#   --max-concurrent <N>       Maximum concurrent jobs (default: unlimited, let SLURM decide)
#   --mem-per-cpu <MEM>        Memory per CPU (default: 8G)
#   --resume                    Skip already-processed parquet files
#   Note: model_name and batch_size come from config.toml only
#   --venv-path <PATH>         Path to Python virtual environment (default: ~/venv/spatial-building-embeddings)
#   --python-module <MODULE>   Python module to load (default: python/3.11.5)
#   --project-root <DIR>       Path to project root directory (default: auto-detect)
#   --no-venv                  Use system Python instead of creating/using venv
#   --test                     Submit only 0061.parquet for testing (smallest file)
#   --help                     Show this help message
#
# GPU Configuration:
#   - Configured for DINOv3 ViT-7B using H100-3g.40gb MIG instance (40GB GPU memory)
#   - Batch size is automatically reduced in config.toml for 7B model (default: 16)
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
#                              The token will be automatically passed to the SLURM jobs.
#
# Exit codes:
#   0: Success
#   1: Invalid arguments or missing required options
#   2: Input directory not found or contains no parquet files
#   3: Python module not available
#   4: Virtual environment creation failed
#   5: Project root not found
#   6: Job submission failed

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../../slurm/common.sh"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default values
ACCOUNT="${SLURM_ACCOUNT:-}"
TIME="24:00:00"
MAX_CONCURRENT=""
MEM_PER_CPU="48G"
RESUME=false
VENV_PATH="${HOME}/venv/spatial-building-embeddings"
PYTHON_MODULE="python/3.11.5"
ARROW_MODULE="arrow/17.0.0"
PROJECT_ROOT=""
NO_VENV=false
TEST_MODE=false

show_usage() {
    grep "^# " "${0}" | sed 's/^# //' | head -n 30
    exit 0
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --account) ACCOUNT="$2"; shift 2 ;;
        --time) TIME="$2"; shift 2 ;;
        --max-concurrent)
            if [ -z "$2" ] || [ "$2" = "unlimited" ]; then
                MAX_CONCURRENT=""
            else
                MAX_CONCURRENT="$2"
            fi
            shift 2
            ;;
        --mem-per-cpu) MEM_PER_CPU="$2"; shift 2 ;;
        --resume) RESUME=true; shift ;;
        --venv-path) VENV_PATH="$2"; shift 2 ;;
        --python-module) PYTHON_MODULE="$2"; shift 2 ;;
        --project-root) PROJECT_ROOT="$2"; shift 2 ;;
        --no-venv) NO_VENV=true; shift ;;
        --test) TEST_MODE=true; shift ;;
        --help) show_usage ;;
        *) error_exit "Unknown option: $1" 1 ;;
    esac
done

if [ -z "${ACCOUNT}" ]; then
    error_exit " --account is required (or set SLURM_ACCOUNT environment variable)" 1
fi

# Step 1: Validate project root
PROJECT_ROOT=$(resolve_project_root "${PROJECT_ROOT}" "${SCRIPT_DIR}")
if [ -z "${PROJECT_ROOT}" ] || [ ! -f "${PROJECT_ROOT}/pyproject.toml" ]; then
    error_exit "Project root not found (pyproject.toml). Use --project-root." 5
fi
info "Project root: ${PROJECT_ROOT}"

# Step 2: Read paths from config.toml
CONFIG_FILE="${PROJECT_ROOT}/config.toml"
INTERMEDIATES_DIR=$(read_toml_value "${CONFIG_FILE}" "paths" "intermediates_dir")
OUTPUT_DIR=$(read_toml_value "${CONFIG_FILE}" "paths" "embeddings_dir")

if [ -z "${INTERMEDIATES_DIR}" ]; then error_exit "intermediates_dir not found in [paths] section of config.toml" 1; fi
if [ -z "${OUTPUT_DIR}" ]; then error_exit "embeddings_dir not found in [paths] section of config.toml" 1; fi

# Step 3: Validate input directories
if [ ! -d "${INTERMEDIATES_DIR}" ]; then
    error_exit "Intermediates directory does not exist: ${INTERMEDIATES_DIR}" 2
fi

PARQUET_COUNT=$(find "${INTERMEDIATES_DIR}" -name "*.parquet" -type f | wc -l)
if [ "${PARQUET_COUNT}" -eq 0 ]; then
    error_exit "No .parquet files found in intermediates directory: ${INTERMEDIATES_DIR}" 2
fi

info "Found ${PARQUET_COUNT} parquet file(s) in ${INTERMEDIATES_DIR}"

# Step 4: Setup Python environment
setup_python_env "${PROJECT_ROOT}" "${VENV_PATH}" "${PYTHON_MODULE}" "${ARROW_MODULE}" "${NO_VENV}" "pandas,rich,pydantic,PIL,torch,timm,einops"

info "Environment setup complete, proceeding to job submission..."

# Step 5: Create output and log directories
mkdir -p "${OUTPUT_DIR}" || error_exit "Failed to create output directory: ${OUTPUT_DIR}" 2

LOG_DIR=$(get_log_dir "${PROJECT_ROOT}" "")
mkdir -p "${LOG_DIR}" || error_exit "Failed to create log directory: ${LOG_DIR}" 2

info "Intermediates directory: ${INTERMEDIATES_DIR}"
info "Output directory: ${OUTPUT_DIR}"
info "Log directory: ${LOG_DIR}"

# Step 6: Generate parquet file list
PARQUET_LIST_FILE="${SCRIPT_DIR}/.parquet_list_$$.txt"
find "${INTERMEDIATES_DIR}" -name "*.parquet" -type f | sort > "${PARQUET_LIST_FILE}"

# Step 7: Count parquet files
NUM_PARQUETS=$(wc -l < "${PARQUET_LIST_FILE}")

# Step 8: If --resume flag, filter out already-processed files
if [ "${RESUME}" = true ]; then
    info "Resume mode: checking for already-processed files..."
    TEMP_LIST="${PARQUET_LIST_FILE}.temp"
    > "${TEMP_LIST}"
    
    PROCESSED=0
    while IFS= read -r PARQUET_FILE; do
        PARQUET_BASENAME=$(basename "${PARQUET_FILE}" .parquet)
        OUTPUT_FILE="${OUTPUT_DIR}/${PARQUET_BASENAME}_embeddings.parquet"
        if [ ! -f "${OUTPUT_FILE}" ]; then
            echo "${PARQUET_FILE}" >> "${TEMP_LIST}"
        else
            ((PROCESSED++)) || true
        fi
    done < "${PARQUET_LIST_FILE}"
    
    mv "${TEMP_LIST}" "${PARQUET_LIST_FILE}"
    NUM_PARQUETS=$(wc -l < "${PARQUET_LIST_FILE}")
    
    if [ "${PROCESSED}" -gt 0 ]; then
        info "Skipping ${PROCESSED} already-processed file(s)"
    fi
fi

if [ "${NUM_PARQUETS}" -eq 0 ]; then
    rm -f "${PARQUET_LIST_FILE}"
    info "No files to process (all already processed or no parquet files found)"
    exit 0
fi

# Step 8.5: If --test mode, use 0061.parquet (smallest file) for testing
if [ "${TEST_MODE}" = true ]; then
    info "Test mode: looking for 0061.parquet (smallest file for testing)"
    TEST_FILE=""
    while IFS= read -r PARQUET_FILE; do
        PARQUET_BASENAME=$(basename "${PARQUET_FILE}" .parquet)
        if [ "${PARQUET_BASENAME}" = "0061" ]; then
            TEST_FILE="${PARQUET_FILE}"
            break
        fi
    done < "${PARQUET_LIST_FILE}"
    
    if [ -z "${TEST_FILE}" ]; then
        warning "0061.parquet not found in list, using first file instead"
        TEST_FILE=$(head -n 1 "${PARQUET_LIST_FILE}")
    fi
    
    TEMP_LIST="${PARQUET_LIST_FILE}.test"
    echo "${TEST_FILE}" > "${TEMP_LIST}"
    mv "${TEMP_LIST}" "${PARQUET_LIST_FILE}"
    NUM_PARQUETS=1
    info "Test mode: using ${TEST_FILE}"
fi

info "Submitting job array for ${NUM_PARQUETS} parquet file(s)"

# Step 9: Submit job array
SBATCH_SCRIPT="${SCRIPT_DIR}/generate_embeddings_array.sbatch"

# Build export string - include PROJECT_ROOT and other variables
# RAW_DIR is no longer needed if paths are absolute/managed by config
EXPORT_VARS="ALL,PARQUET_LIST_FILE=${PARQUET_LIST_FILE},OUTPUT_DIR=${OUTPUT_DIR},PYTHON_MODULE=${PYTHON_MODULE},LOG_DIR=${LOG_DIR},PROJECT_ROOT=${PROJECT_ROOT}"
if [ "${NO_VENV}" = false ] && [ -n "${VENV_PATH:-}" ]; then
    EXPORT_VARS="${EXPORT_VARS},VENV_PATH=${VENV_PATH}"
fi
if [ -n "${ARROW_MODULE:-}" ]; then
    EXPORT_VARS="${EXPORT_VARS},ARROW_MODULE=${ARROW_MODULE}"
fi
# Pass through HF_TOKEN if set (required for gated models like DINOv3)
if [ -n "${HF_TOKEN:-}" ]; then
    EXPORT_VARS="${EXPORT_VARS},HF_TOKEN=${HF_TOKEN}"
    info "HF_TOKEN detected - will be passed to job for Hugging Face authentication"
fi

if [ -n "${MAX_CONCURRENT}" ]; then
    ARRAY_SPEC="1-${NUM_PARQUETS}%${MAX_CONCURRENT}"
else
    ARRAY_SPEC="1-${NUM_PARQUETS}"
fi

SUBMIT_OUTPUT=$(sbatch \
    --account="${ACCOUNT}" \
    --time="${TIME}" \
    --mem-per-cpu="${MEM_PER_CPU}" \
    --array="${ARRAY_SPEC}" \
    --output="${LOG_DIR}/generate_embeddings_%A_%a.out" \
    --error="${LOG_DIR}/generate_embeddings_%A_%a.err" \
    --export="${EXPORT_VARS}" \
    "${SBATCH_SCRIPT}" 2>&1)

if echo "${SUBMIT_OUTPUT}" | grep -q "Submitted batch job"; then
    JOB_ID=$(echo "${SUBMIT_OUTPUT}" | grep -oE '[0-9]+' | head -n 1)
    
    echo ""
    echo "=========================================="
    echo "Job Submission Summary"
    echo "=========================================="
    echo "Job ID: ${JOB_ID}"
    echo "Number of tasks: ${NUM_PARQUETS}"
    echo "Max concurrent: ${MAX_CONCURRENT:-unlimited}"
    echo "Time limit: ${TIME}"
    echo "Intermediates directory: ${INTERMEDIATES_DIR}"
    echo "Output directory: ${OUTPUT_DIR}"
    if [ "${NO_VENV}" = false ] && [ -n "${VENV_PATH}" ]; then
        echo "Virtual environment: ${VENV_PATH}"
    else
        echo "Python: System Python (${PYTHON_MODULE})"
    fi
    echo "Parquet list file: ${PARQUET_LIST_FILE}"
    echo ""
    echo "Monitor job status:"
    echo "  squeue -j ${JOB_ID}"
    echo ""
    echo "View logs:"
    echo "  tail -f ${LOG_DIR}/generate_embeddings_${JOB_ID}_*.out"
    echo "  tail -f ${LOG_DIR}/generate_embeddings_${JOB_ID}_*.err"
    echo "=========================================="
    
    exit 0
else
    error_exit "Job submission failed: ${SUBMIT_OUTPUT}" 6
fi
