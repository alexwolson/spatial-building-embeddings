# Download Raw Data

This directory contains scripts for downloading the raw dataset tar files.

## Prerequisites

- `aria2c` must be installed on your system
  - macOS: `brew install aria2`
  - Linux: `sudo apt-get install aria2` or `sudo yum install aria2`
  - Windows: Download from [aria2 official website](https://aria2.github.io/)

## Usage

Run the download script with an output directory:

```bash
./download.sh [--include-problematic] <output_directory>
```

For example:

```bash
./download.sh /path/to/data
```

### Options

- `--include-problematic` or `-i`: Include problematic tar files that error out when extracted. By default, these files are excluded from the download:
  - 0004.tar
  - 0030.tar
  - 0070.tar
  - 0072.tar
  - 0075.tar
  - 0080.tar
  - 0081.tar
  - 0084.tar
  - 0097.tar

## What the Script Does

1. **Downloads the URL list**: If `dataset_unaligned_aria2c.txt` doesn't exist locally, the script automatically downloads it from the [3D Street View repository](https://github.com/amir32002/3D_Street_View/blob/master/links/dataset_unaligned_aria2c.txt).

2. **Filters problematic files**: By default, the script excludes 9 problematic tar files that error out when extracted. Use the `--include-problematic` flag to download them anyway.

3. **Runs aria2c**: The script uses `aria2c` with the following standard options:
   - `--auto-file-renaming=false`: Preserves original filenames
   - `--continue`: Resumes interrupted downloads
   - `--split=5`: Uses 5 connections per file for faster downloads
   - `--max-connection-per-server=5`: Limits connections per server
   - `-d <output_directory>`: Sets the base download directory

4. **Preserves directory structure**: The downloaded files will maintain the `dataset_unaligned/` directory structure as specified in the URL list file.

## Notes

- The script will create the output directory if it doesn't exist
- Downloads can be resumed if interrupted
- The URL list file is cached locally after the first download

