#!/bin/bash

# Helper script to install aria2c locally for use with download.sh
# Usage: ./install_aria2.sh

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="$SCRIPT_DIR/bin"
ARIA2_URL="https://github.com/aria2/aria2/releases/download/release-1.37.0/aria2-1.37.0.tar.bz2"

echo "Installing aria2c to $INSTALL_DIR/aria2c..."
mkdir -p "$INSTALL_DIR"

TEMP_ARCHIVE="$INSTALL_DIR/aria2.tar.bz2"

# Download
if command -v curl &> /dev/null; then
    curl -L -o "$TEMP_ARCHIVE" "$ARIA2_URL"
elif command -v wget &> /dev/null; then
    wget -O "$TEMP_ARCHIVE" "$ARIA2_URL"
else
    echo "Error: Neither curl nor wget found for downloading aria2c."
    exit 1
fi

if [ $? -ne 0 ]; then
    echo "Error: Failed to download aria2c static binary."
    rm -f "$TEMP_ARCHIVE"
    exit 1
fi

# Extract
echo "Extracting aria2c..."
if ! tar -xjf "$TEMP_ARCHIVE" -C "$INSTALL_DIR"; then
    echo "Error: Failed to extract archive."
    rm -f "$TEMP_ARCHIVE"
    exit 1
fi

# Find the binary (it might be in a subdir)
EXTRACTED_BIN=$(find "$INSTALL_DIR" -name "aria2c" -type f | head -n 1)

if [ -n "$EXTRACTED_BIN" ]; then
    mv "$EXTRACTED_BIN" "$INSTALL_DIR/aria2c"
    chmod +x "$INSTALL_DIR/aria2c"
    echo "aria2c installed successfully to $INSTALL_DIR/aria2c"
    
    # Cleanup
    rm -f "$TEMP_ARCHIVE"
    # Remove any extracted subdirectories (assumes they start with aria2)
    find "$INSTALL_DIR" -mindepth 1 -maxdepth 1 -type d -name "aria2*" -exec rm -rf {} +
    exit 0
else
    echo "Error: Could not find aria2c binary in downloaded archive."
    rm -f "$TEMP_ARCHIVE"
    exit 1
fi

