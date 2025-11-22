#!/bin/bash

# Script to download the aria2c input file and display the download command
# Usage: ./download.sh <output_directory>

set -e  # Exit on error

# Check if output directory argument is provided
if [ -z "$1" ]; then
    echo "Error: Output directory argument is required"
    echo "Usage: $0 <output_directory>"
    exit 1
fi

OUTPUT_DIR="$1"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TXT_FILE="$SCRIPT_DIR/dataset_unaligned_aria2c.txt"
TXT_URL="https://raw.githubusercontent.com/amir32002/3D_Street_View/master/links/dataset_unaligned_aria2c.txt"

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

# Check if local aria2c exists and add to PATH
if [ -f "$SCRIPT_DIR/bin/aria2c" ]; then
    export PATH="$SCRIPT_DIR/bin:$PATH"
fi

echo ""
echo "Run this command to download the dataset:"
echo ""
echo "aria2c --auto-file-renaming=false --continue --split=5 --max-connection-per-server=5 -d $OUTPUT_DIR -i $TXT_FILE"
