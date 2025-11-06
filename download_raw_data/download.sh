#!/bin/bash

# Script to download raw dataset tar files using aria2c
# Usage: ./download.sh [--include-problematic] <output_directory>

set -e  # Exit on error

# Problematic tar files that error out when extracted
PROBLEMATIC_FILES=("0004.tar" "0030.tar" "0070.tar" "0072.tar" "0075.tar" "0080.tar" "0081.tar" "0084.tar" "0097.tar")

# Parse arguments
INCLUDE_PROBLEMATIC=false
OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --include-problematic|-i)
            INCLUDE_PROBLEMATIC=true
            shift
            ;;
        *)
            if [ -z "$OUTPUT_DIR" ]; then
                OUTPUT_DIR="$1"
            else
                echo "Error: Unexpected argument: $1"
                echo "Usage: $0 [--include-problematic] <output_directory>"
                exit 1
            fi
            shift
            ;;
    esac
done

# Check if output directory argument is provided
if [ -z "$OUTPUT_DIR" ]; then
    echo "Error: Output directory argument is required"
    echo "Usage: $0 [--include-problematic] <output_directory>"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TXT_FILE="$SCRIPT_DIR/dataset_unaligned_aria2c.txt"
TXT_URL="https://raw.githubusercontent.com/amir32002/3D_Street_View/master/links/dataset_unaligned_aria2c.txt"
FILTERED_TXT_FILE="$SCRIPT_DIR/dataset_unaligned_aria2c_filtered.txt"

# Check if aria2c is installed
if ! command -v aria2c &> /dev/null; then
    echo "Error: aria2c is not installed"
    echo "Please install aria2c:"
    echo "  macOS: brew install aria2"
    echo "  Linux: sudo apt-get install aria2"
    exit 1
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

# Run aria2c with standard defaults
echo "Starting download to $OUTPUT_DIR..."
aria2c \
    --auto-file-renaming=false \
    --continue \
    --split=5 \
    --max-connection-per-server=5 \
    -d "$OUTPUT_DIR" \
    -i "$INPUT_FILE"

if [ $? -eq 0 ]; then
    echo "Download completed successfully!"
else
    echo "Error: Download failed"
    exit 1
fi

