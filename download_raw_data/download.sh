#!/bin/bash

# Script to download the dataset using aria2c (building it from source if necessary).
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

# 1. Load environment
if command -v module &> /dev/null; then
    echo "Loading StdEnv/2023..."
    module load StdEnv/2023 || echo "Warning: Failed to load StdEnv/2023, proceeding anyway..."
fi

# 2. Check for aria2c or build it
PREFIX="$HOME/.local"
ARIA2_BIN="$PREFIX/bin/aria2c"

if [ -f "$ARIA2_BIN" ]; then
    echo "Found aria2c at $ARIA2_BIN"
elif command -v aria2c &> /dev/null; then
    ARIA2_BIN=$(command -v aria2c)
    echo "Found aria2c in PATH at $ARIA2_BIN"
else
    echo "aria2c not found. Building from source..."
    
    # Create temporary directory for build
    BUILD_DIR=$(mktemp -d)
    echo "Building in $BUILD_DIR..."
    
    pushd "$BUILD_DIR" > /dev/null
    
    # Clone
    echo "Cloning aria2..."
    git clone https://github.com/aria2/aria2.git
    cd aria2
    
    # Configure and Build
    echo "Configuring..."
    mkdir -p "$PREFIX"
    autoreconf -i
    ./configure --prefix="$PREFIX"
    
    echo "Compiling (using $(nproc) cores)..."
    make -j$(nproc)
    
    echo "Installing..."
    make install
    
    popd > /dev/null
    
    # Cleanup
    echo "Cleaning up build artifacts..."
    rm -rf "$BUILD_DIR"
    
    if [ -f "$ARIA2_BIN" ]; then
        echo "Successfully installed aria2c to $ARIA2_BIN"
    else
        echo "Error: Build completed but binary not found at $ARIA2_BIN"
        exit 1
    fi
fi

# 3. Ensure dataset list exists
if [ ! -f "$TXT_FILE" ]; then
    echo "Downloading dataset_unaligned_aria2c.txt from GitHub..."
    if command -v curl &> /dev/null; then
        curl -L -o "$TXT_FILE" "$TXT_URL"
    elif command -v wget &> /dev/null; then
        wget -O "$TXT_FILE" "$TXT_URL"
    else
        echo "Error: Neither curl nor wget found."
        exit 1
    fi
    
    if [ $? -ne 0 ]; then
        echo "Error: Failed to download dataset_unaligned_aria2c.txt"
        exit 1
    fi
    echo "Downloaded dataset_unaligned_aria2c.txt"
else
    echo "Using existing dataset_unaligned_aria2c.txt"
fi

# 4. Run download
echo ""
echo "Starting download to $OUTPUT_DIR..."
echo "Command: $ARIA2_BIN --continue -d $OUTPUT_DIR -i $TXT_FILE"
echo ""

mkdir -p "$OUTPUT_DIR"
"$ARIA2_BIN" --auto-file-renaming=false --continue --split=5 --max-connection-per-server=5 -d "$OUTPUT_DIR" -i "$TXT_FILE"
