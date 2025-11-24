#!/bin/bash

# Shared configuration and functions for SLURM submission scripts.

# Colours for output
export RED='\033[0;31m'
export GREEN='\033[0;32m'
export YELLOW='\033[1;33m'
export NC='\033[0m' # No Colour

# -----------------------------------------------------------------------------
# Logging Helpers
# -----------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------
# Path Utilities
# -----------------------------------------------------------------------------

resolve_abs_path() {
    python3 -c 'import os, sys; print(os.path.abspath(sys.argv[1]))' "$1"
}

resolve_project_root() {
    local provided_root="$1"
    local script_dir="$2"
    
    if [ -n "${provided_root}" ]; then
        resolve_abs_path "${provided_root}"
        return
    fi

    # Auto-detect
    local candidate="${script_dir}/../.."
    if [ -f "${candidate}/pyproject.toml" ]; then
        resolve_abs_path "${candidate}"
    elif [ -f "${PWD}/pyproject.toml" ]; then
        resolve_abs_path "${PWD}"
    else
        echo "" 
    fi
}

# -----------------------------------------------------------------------------
# Config Utilities
# -----------------------------------------------------------------------------

read_toml_value() {
    local toml_file="$1"
    local section="$2"
    local key="$3"
    
    if [ ! -f "${toml_file}" ]; then
        echo "Config file not found: ${toml_file}" >&2
        return 1
    fi
    
    # Use python for robust TOML parsing (requires Python 3.11+)
    python3 -c "
import tomllib
import sys
try:
    with open('${toml_file}', 'rb') as f:
        data = tomllib.load(f)
    if '${section}' not in data:
        print(f\"Section [${section}] not found in config file\", file=sys.stderr)
        sys.exit(1)
    val = data.get('${section}', {}).get('${key}')
    if val is None:
        print(f\"Key '${key}' not found in section [${section}] of config file\", file=sys.stderr)
        sys.exit(1)
    print(val)
except FileNotFoundError:
    print(f\"Config file not found: ${toml_file}\", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f\"Error reading config: {e}\", file=sys.stderr)
    sys.exit(1)
" || return 1
}

get_log_dir() {
    local project_root="$1"
    local config_file="${project_root}/config.toml"
    local default_subdir="$2" # e.g., "logs"

    if [ ! -f "${config_file}" ]; then
        error_exit "Config file not found: ${config_file}" 1
    fi

    # Try to read from [global] section
    local log_dir
    log_dir=$(read_toml_value "${config_file}" "global" "log_dir")

    if [ -z "${log_dir}" ]; then
        if [ -n "${default_subdir}" ]; then
            log_dir="${default_subdir}"
        else
            error_exit "log_dir not found in [global] section of config.toml" 1
        fi
    fi

    # Assuming absolute paths in config.toml as per user requirement
    # If relative, it remains relative (caller responsibility or absolute assumption)
    
    echo "${log_dir}"
}

# -----------------------------------------------------------------------------
# Environment Setup
# -----------------------------------------------------------------------------

setup_python_env() {
    local project_root="$1"
    local venv_path="$2"
    local python_module="$3"
    local arrow_module="$4"
    local no_venv="$5"
    local check_packages="$6" # Comma-separated list of packages to import-check

    # Wheelhouse directories for Alliance clusters
    local wheelhouse_dirs=(
        "/cvmfs/soft.computecanada.ca/custom/python/wheelhouse/gentoo2023/x86-64-v4"
        "/cvmfs/soft.computecanada.ca/custom/python/wheelhouse/gentoo2023/x86-64-v3"
        "/cvmfs/soft.computecanada.ca/custom/python/wheelhouse/gentoo2023/generic"
        "/cvmfs/soft.computecanada.ca/custom/python/wheelhouse/generic"
    )
    local find_link_flags=()
    for wh in "${wheelhouse_dirs[@]}"; do
        find_link_flags+=(--find-links "${wh}")
    done

    if [ "${no_venv}" = false ]; then
        # Check Python module
        if ! module avail "${python_module}" 2>/dev/null | grep -q "${python_module}"; then
             error_exit "Python module not available: ${python_module}" 3
        fi

        # Load Arrow module (gcc dependency usually handled by module system or pre-load)
        if [ -n "${arrow_module}" ]; then
            info "Loading Arrow module: ${arrow_module}"
            module load gcc 2>/dev/null || true
            module load "${arrow_module}" || warning "Failed to load Arrow module - PyArrow may be unavailable"
        fi

        if [ -d "${venv_path}" ]; then
            info "Using existing virtual environment: ${venv_path}"
            source "${venv_path}/bin/activate" || error_exit "Failed to activate virtual environment" 4
            
            # Verification
            local verify_cmd="import sys"
            if [ -n "${check_packages}" ]; then
                verify_cmd="import ${check_packages}"
            fi
            
            if ! python -c "${verify_cmd}" 2>/dev/null; then
                warning "Key packages (${check_packages}) missing or broken, reinstalling dependencies..."
                
                if ! command -v uv >/dev/null 2>&1; then
                    info "Installing uv..."
                    pip install uv || error_exit "Failed to install uv" 4
                fi
                
                pushd "${project_root}" > /dev/null
                uv pip install "${find_link_flags[@]}" -e . || error_exit "Failed to reinstall project dependencies" 4
                popd > /dev/null
                info "Dependencies reinstalled"
            else
                info "Dependencies verified"
            fi
            deactivate
        else
            info "Creating virtual environment: ${venv_path}"
            module load "${python_module}"
            
            if ! command -v uv >/dev/null 2>&1; then
                info "Installing uv..."
                pip install uv || error_exit "Failed to install uv" 4
            fi
            
            uv venv "${venv_path}" || error_exit "Failed to create virtual environment" 4
            source "${venv_path}/bin/activate" || error_exit "Failed to activate virtual environment" 4
            
            pushd "${project_root}" > /dev/null
            uv pip install "${find_link_flags[@]}" -e . || error_exit "Failed to install project dependencies" 4
            popd > /dev/null
            
            deactivate
            info "Virtual environment created and dependencies installed"
        fi
    else
        info "Using system Python (--no-venv specified)"
        if ! module avail "${python_module}" 2>/dev/null | grep -q "${python_module}"; then
            error_exit "Python module not available: ${python_module}" 3
        fi
        # No venv path to return/export
    fi
}

load_python_env() {
    local venv_path="$1"
    local python_module="$2"
    local arrow_module="$3"

    module purge

    if [ -n "${arrow_module}" ]; then
        info "Loading Arrow module: ${arrow_module}"
        module load gcc 2>/dev/null || true
        module load "${arrow_module}" || warning "Failed to load Arrow module - PyArrow may be unavailable"
    fi

    if [ -n "${python_module}" ]; then
        info "Loading Python module: ${python_module}"
        module load "${python_module}"
    fi

    if [ -n "${venv_path}" ]; then
        if [ -d "${venv_path}" ]; then
            info "Activating virtual environment: ${venv_path}"
            source "${venv_path}/bin/activate"
        else
            # Only warn if we were given a path but it's missing (unexpected on compute node if submit script did its job)
            warning "Virtual environment not found at ${venv_path}, using system Python"
        fi
    else
        info "Using system Python (no virtual environment)"
    fi
}
