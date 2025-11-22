#!/bin/bash

# Script to download raw dataset tar files using aria2c
# Usage: ./download.sh [--include-problematic] <output_directory>

set -e  # Exit on error

# Problematic tar files that error out when extracted
PROBLEMATIC_FILES=("0004.tar" "0030.tar" "0070.tar" "0072.tar" "0075.tar" "0080.tar" "0081.tar" "0084.tar" "0097.tar")

# Parse arguments
INCLUDE_PROBLEMATIC=false
INSTALL_ARIA2=false
OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --include-problematic|-i)
            INCLUDE_PROBLEMATIC=true
            shift
            ;;
        --install-aria2)
            INSTALL_ARIA2=true
            shift
            ;;
        *)
            if [ -z "$OUTPUT_DIR" ]; then
                OUTPUT_DIR="$1"
            else
                echo "Error: Unexpected argument: $1"
                echo "Usage: $0 [--include-problematic] [--install-aria2] <output_directory>"
                exit 1
            fi
            shift
            ;;
    esac
done

# Check if output directory argument is provided
if [ -z "$OUTPUT_DIR" ]; then
    echo "Error: Output directory argument is required"
    echo "Usage: $0 [--include-problematic] [--install-aria2] <output_directory>"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TXT_FILE="$SCRIPT_DIR/dataset_unaligned_aria2c.txt"
TXT_URL="https://raw.githubusercontent.com/amir32002/3D_Street_View/master/links/dataset_unaligned_aria2c.txt"
FILTERED_TXT_FILE="$SCRIPT_DIR/dataset_unaligned_aria2c_filtered.txt"

# Function to install aria2c locally
install_aria2_locally() {
    local install_dir="$SCRIPT_DIR/bin"
    # URL for a static build of aria2c 1.36.0
    local aria2_url="https://github.com/q3aql/aria2-static-builds/releases/download/v1.36.0/aria2-1.36.0-linux-gnu-64bit-build1.tar.bz2"
    
    echo "Attempting to install aria2c locally to $install_dir..."
    mkdir -p "$install_dir"
    
    local temp_archive="$install_dir/aria2.tar.bz2"
    
    # Download
    if command -v curl &> /dev/null; then
        curl -L -o "$temp_archive" "$aria2_url"
    elif command -v wget &> /dev/null; then
        wget -O "$temp_archive" "$aria2_url"
    else
        echo "Error: Neither curl nor wget found for downloading aria2c."
        return 1
    fi
    
    if [ $? -ne 0 ]; then
        echo "Error: Failed to download aria2c static binary."
        rm -f "$temp_archive"
        return 1
    fi
    
    # Extract
    echo "Extracting aria2c..."
    if ! tar -xjf "$temp_archive" -C "$install_dir"; then
        echo "Error: Failed to extract archive."
        rm -f "$temp_archive"
        return 1
    fi
    
    # Find the binary (it might be in a subdir)
    local extracted_bin=$(find "$install_dir" -name "aria2c" -type f | head -n 1)
    
    if [ -n "$extracted_bin" ]; then
        mv "$extracted_bin" "$install_dir/aria2c"
        chmod +x "$install_dir/aria2c"
        echo "aria2c installed successfully to $install_dir/aria2c"
        
        # Cleanup
        rm -f "$temp_archive"
        # Remove any extracted subdirectories (assumes they start with aria2)
        find "$install_dir" -mindepth 1 -maxdepth 1 -type d -name "aria2*" -exec rm -rf {} +
        return 0
    else
        echo "Error: Could not find aria2c binary in downloaded archive."
        return 1
    fi
}

# Function for sequential fallback download
download_sequential() {
    local input_file="$1"
    local output_dir="$2"
    
    echo "Using sequential download (curl/wget) as fallback..."
    
    local state=0
    local current_url=""
    
    while IFS= read -r line; do
        # Skip empty lines
        [[ -z "$line" ]] && continue
        
        if [ $state -eq 0 ]; then
            if [[ "$line" == http* ]]; then
                current_url="$line"
                state=1
            fi
        elif [ $state -eq 1 ]; then
            if [[ "$line" == *out=* ]]; then
                local rel_path=$(echo "$line" | sed 's/.*out=//')
                local full_path="$output_dir/$rel_path"
                local dir_path=$(dirname "$full_path")
                
                # Create directory if it doesn't exist
                mkdir -p "$dir_path"
                
                if [ -f "$full_path" ]; then
                    echo "File $rel_path exists. Resuming..."
                else
                    echo "Downloading $rel_path..."
                fi
                
                if command -v curl &> /dev/null; then
                    curl -L -C - -o "$full_path" "$current_url" --retry 3 --fail
                elif command -v wget &> /dev/null; then
                    wget -c -O "$full_path" "$current_url"
                else
                    echo "Error: Neither curl nor wget found."
                    return 1
                fi
                
                state=0
            else
                # If we expected out= but got something else, maybe it's a new URL or garbage?
                # Current file format implies strictly paired lines.
                if [[ "$line" == http* ]]; then
                    # Reset state and treat as new URL (prev one had no out=?)
                    current_url="$line"
                    state=1
                else
                    state=0
                fi
            fi
        fi
    done < <(cat "$input_file"; echo)
}

# Check for aria2c availability
USE_ARIA2=true

# 1. Try loading module (common on clusters)
module load aria2 2>/dev/null || true

# 2. Check for local binary
if [ -f "$SCRIPT_DIR/bin/aria2c" ]; then
    export PATH="$SCRIPT_DIR/bin:$PATH"
fi

# 3. Check if installed
if ! command -v aria2c &> /dev/null; then
    if [ "$INSTALL_ARIA2" = true ]; then
        install_aria2_locally
        if [ $? -eq 0 ]; then
            export PATH="$SCRIPT_DIR/bin:$PATH"
        else
            echo "Warning: Failed to install aria2c. Falling back to sequential download."
            USE_ARIA2=false
        fi
    else
        echo "aria2c not found."
        echo "Tip: Pass --install-aria2 to automatically install a static binary for faster downloads."
        echo "Falling back to sequential download (slower)..."
        USE_ARIA2=false
    fi
else
    echo "Found aria2c: $(command -v aria2c)"
fi


# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Download the txt file if it doesn't exist locally
if [ ! -f "$TXT_FILE" ]; then
    echo "Downloading dataset_unaligned_aria2c.txt from GitHub..."
    curl -L -o "$TXT_FILE" "$TXT_URL"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to download dataset_unaligned_aria2c.txt"
        exit 1
    fi
    echo "Downloaded dataset_unaligned_aria2c.txt"
else
    echo "Using existing dataset_unaligned_aria2c.txt"
fi

# Create filtered version of the txt file if excluding problematic files
if [ "$INCLUDE_PROBLEMATIC" = false ]; then
    echo "Filtering out problematic tar files..."
    # Create a temporary file for filtering
    TEMP_FILE=$(mktemp)
    
    # Read the file line by line and filter out problematic files
    SKIP_NEXT=false
    while IFS= read -r line || [ -n "$line" ]; do
        if [ "$SKIP_NEXT" = true ]; then
            SKIP_NEXT=false
            continue
        fi
        
        # Check if this line contains a problematic filename
        IS_PROBLEMATIC=false
        for problematic in "${PROBLEMATIC_FILES[@]}"; do
            if [[ "$line" == *"$problematic"* ]]; then
                IS_PROBLEMATIC=true
                break
            fi
        done
        
        if [ "$IS_PROBLEMATIC" = true ]; then
            # Skip this line and the next line (the out= line)
            SKIP_NEXT=true
            continue
        fi
        
        echo "$line" >> "$TEMP_FILE"
    done < "$TXT_FILE"
    
    mv "$TEMP_FILE" "$FILTERED_TXT_FILE"
    INPUT_FILE="$FILTERED_TXT_FILE"
    echo "Created filtered download list (excluding ${#PROBLEMATIC_FILES[@]} problematic files)"
else
    INPUT_FILE="$TXT_FILE"
    echo "Including all files (including problematic ones)"
fi

# Run download
echo "Starting download to $OUTPUT_DIR..."

if [ "$USE_ARIA2" = true ]; then
    aria2c \
        --auto-file-renaming=false \
        --continue \
        --split=5 \
        --max-connection-per-server=5 \
        -d "$OUTPUT_DIR" \
        -i "$INPUT_FILE"
        
    EXIT_CODE=$?
else
    download_sequential "$INPUT_FILE" "$OUTPUT_DIR"
    EXIT_CODE=$?
fi

if [ $EXIT_CODE -eq 0 ]; then
    echo "Download completed successfully!"
else
    echo "Error: Download failed"
    exit 1
fi

