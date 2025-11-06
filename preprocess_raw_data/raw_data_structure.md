# Raw Data Structure in Tar Archives

## Overview

The preprocessing pipeline extracts tar archives containing building street view images and their associated metadata. This document describes the structure of the raw data within these tar files and what information is to be retained during processing.

## Tar Archive Contents

Each tar archive (always `.tar`) contains a directory structure with the following layout:

```
{dataset_id:04d}/
└── {dataset_id:04d}/
    ├── {stem}.jpg    # Street view image
    ├── {stem}.txt    # Metadata file
    ├── {stem}.jpg    # Another image
    ├── {stem}.txt    # Another metadata file
    └── ...
```

Where:
- `{dataset_id:04d}` is a zero-padded 4-digit dataset identifier (e.g., `0088`)
- `{stem}` is a filename stem that may encode additional identifiers
- Each image file (`.jpg` or `.jpeg`) has a corresponding `.txt` metadata file with the same stem

### Example Structure

```
0088/
└── 0088/
    ├── image_001.jpg
    ├── image_001.txt
    ├── image_002.jpg
    ├── image_002.txt
    └── ...
```

## Metadata File Format (`.txt` files)

Each `.txt` metadata file contains one or more lines, but the critical line starts with the character `d`. This line encodes comprehensive information about the image and its capture parameters.

### Full "d" Line Format

The `d` line contains **17 space-separated values** in the following order:

```
d {DatasetID} {TargetID} {PatchID} {StreetViewID} {target_lat} {target_lon} {target_height} {normal_x} {normal_y} {normal_z} {street_lat} {street_lon} {street_height} {distance} {heading} {pitch} {roll}
```

**Field Breakdown:**

1. **Identifier Fields (4 values):**
   - `DatasetID`: Integer identifier for the dataset collection
   - `TargetID`: Integer identifier for the target building/location (used for triplet grouping)
   - `PatchID`: Integer identifier for the specific image patch/view
   - `StreetViewID`: Integer identifier for the street view source

2. **Target Point (3 values):**
   - `target_lat`: Latitude of the target building point (degrees)
   - `target_lon`: Longitude of the target building point (degrees)
   - `target_height`: Height/elevation of the target point (meters)

3. **Surface Normal (3 values):**
   - `normal_x`, `normal_y`, `normal_z`: 3D surface normal vector components

4. **Street View Location (3 values):**
   - `street_lat`: Latitude of the camera/street view location (degrees)
   - `street_lon`: Longitude of the camera/street view location (degrees)
   - `street_height`: Height/elevation of the camera location (meters)

5. **Camera Parameters (4 values):**
   - `distance`: Distance from camera to target (meters)
   - `heading`: Camera heading/yaw angle (degrees)
   - `pitch`: Camera pitch angle (degrees)
   - `roll`: Camera roll angle (degrees)

### Example Metadata Line

```
d 88 1234 567 890 45.5088 -73.5878 50.2 0.0 0.0 1.0 45.5090 -73.5880 2.0 150.5 45.0 10.0 0.0
```

## Image Files (`.jpg` or `.jpeg`)

Each metadata file is paired with a corresponding image file:
- Same filename stem as the `.txt` file
- Format: JPEG (`.jpg` or `.jpeg`)
- Content: Street view image of a building facade
- Images may be of varying sizes and aspect ratios

## What We Actually Retain

During preprocessing, the pipeline selectively extracts and transforms the raw data. Here's what is **retained** and what is **discarded**:

### ✅ Retained Information

#### 1. **Core Identifiers** (from metadata)
- `TargetID`: **Critical** - used for grouping images into triplets (same target = positive pairs)
- `PatchID`: Preserved for identifying specific views
- `StreetViewID`: Preserved for source tracking

#### 2. **Geographic Coordinates** (from metadata)
- `Target Point Latitude`: **Critical** - used for computing geographic embeddings
- `Target Point Longitude`: **Critical** - used for computing geographic embeddings
- **Note:** Target height is **discarded** (only lat/lon are used)

#### 3. **Image Data**

- For now, leave this data where it is
- However, we should attempt to validate that:
    - Each image is valid (i.e. not corrupted, or otherwise empty)
    - There are no orphan images or metadata files (where one exists without the other)
- In cases where an image/metadata pair is invalid, it should be skipped entirely.

#### 6. **Data Splits**
- Train/validation/test splits based on `TargetID` (ensures no target leakage)
- Default split: 70% train, 15% val, 15% test

### ❌ Discarded Information

The following fields from the metadata are **parsed but not retained**:

1. **Target Point Height**: The elevation/height component of the target point
2. **Surface Normal**: All 3 components (normal_x, normal_y, normal_z)
3. **Street View Location**: All 3 components (street_lat, street_lon, street_height)
4. **Camera Parameters**: Distance, heading, pitch, and roll angles
6. **Image Metadata**: EXIF data, original dimensions, etc. are not preserved

### Data Size

There are on the order of **25 million** image - metadata entries in this dataset.