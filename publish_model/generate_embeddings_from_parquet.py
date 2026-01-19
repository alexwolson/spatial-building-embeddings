#!/usr/bin/env python3
"""
Generate embeddings for images in packaged_images.parquet using the published 
Spatial Building Embeddings model from HuggingFace Hub.

This script reads a parquet file containing building images and metadata,
generates embeddings for each image, and writes the embeddings back to a new
parquet file or CSV.

Usage:
    # Generate embeddings and save to new parquet file
    uv run python publish_model/generate_embeddings_from_parquet.py \\
        --input /Volumes/Data/north_america_building_sampler/artifacts/packaged_images.parquet \\
        --output embeddings.parquet

    # Generate embeddings for first 100 images only
    uv run python publish_model/generate_embeddings_from_parquet.py \\
        --input packaged_images.parquet \\
        --output embeddings.parquet \\
        --limit 100

    # Use custom model ID
    MODEL_ID="custom/org/model-name" uv run python publish_model/generate_embeddings_from_parquet.py \\
        --input packaged_images.parquet \\
        --output embeddings.parquet
"""

import argparse
import io
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


def load_model_from_hub(model_id: str, device: Optional[str] = None) -> tuple[AutoModel, AutoImageProcessor]:
    """
    Load the Spatial Building Embeddings model and processor from HuggingFace Hub.

    Args:
        model_id: HuggingFace model ID (e.g., 'alexwaolson/spatial-building-embeddings')
        device: Device to load model on ('cuda', 'cpu', or None for auto-detect)

    Returns:
        Tuple of (model, processor)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model from HuggingFace Hub: {model_id}")
    print(f"Using device: {device}")

    # Load model (trust_remote_code=True needed for custom model classes)
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
    model.eval()
    model.to(device)

    # Load image processor from the backbone model name
    backbone_name = model.config.backbone_model_name
    print(f"Loading image processor for backbone: {backbone_name}")
    processor = AutoImageProcessor.from_pretrained(backbone_name)

    print("Model loaded successfully!")
    return model, processor


def image_bytes_to_pil(image_bytes: bytes) -> Image.Image:
    """
    Convert image bytes to PIL Image.

    Args:
        image_bytes: Binary image data (JPEG bytes)

    Returns:
        PIL Image object
    """
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def generate_embeddings_batch(
    images: list[bytes],
    model: AutoModel,
    processor: AutoImageProcessor,
    device: str,
) -> np.ndarray:
    """
    Generate embeddings for a batch of images.

    Args:
        images: List of image byte arrays
        model: Loaded SpatialEmbeddingsModel
        processor: Image processor
        device: Device to run inference on

    Returns:
        Numpy array of shape (batch_size, 256) containing embeddings
    """
    device_obj = torch.device(device)

    # Convert bytes to PIL Images
    pil_images = [image_bytes_to_pil(img_bytes) for img_bytes in images]

    # Preprocess batch
    inputs = processor(images=pil_images, return_tensors="pt")
    inputs = {k: v.to(device_obj) for k, v in inputs.items()}

    # Generate embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        if isinstance(outputs, dict):
            embeddings = outputs["embeddings"]
        else:
            embeddings = outputs[0]

        # Move to CPU and convert to numpy
        embeddings_np = embeddings.cpu().numpy().astype(np.float32)

    return embeddings_np


def _atomic_write_parquet_part(
    part_path: Path,
    osmids: list[object],
    embeddings: np.ndarray,
) -> None:
    """
    Atomically write a parquet part file with columns: OSMID, embedding.
    Embedding is stored as a PyArrow list array with float32 elements per row.
    """
    part_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = part_path.with_name(part_path.name + ".tmp")

    if embeddings.ndim != 2:
        raise ValueError(f"Expected embeddings to be 2D array, got shape {embeddings.shape}")

    if len(osmids) != embeddings.shape[0]:
        raise ValueError(f"OSMID count ({len(osmids)}) != embeddings rows ({embeddings.shape[0]})")

    # Ensure float32 for consistent parquet typing.
    embeddings = embeddings.astype(np.float32, copy=False)
    embedding_lists: list[list[float]] = embeddings.tolist()

    table = pa.Table.from_arrays(
        [
            pa.array(osmids),
            pa.array(embedding_lists, type=pa.list_(pa.float32())),
        ],
        names=["OSMID", "embedding"],
    )

    try:
        pq.write_table(table, tmp_path, compression="snappy")
        os.replace(tmp_path, part_path)
    except Exception:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        raise


def _atomic_write_empty_parquet_part(part_path: Path) -> None:
    """
    Write an empty parquet part file to mark a chunk as processed.
    This prevents re-processing empty chunks on resume.
    """
    part_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = part_path.with_name(part_path.name + ".tmp")
    table = pa.Table.from_arrays(
        [
            pa.array([], type=pa.int64()),
            pa.array([], type=pa.list_(pa.float32())),
        ],
        names=["OSMID", "embedding"],
    )
    try:
        pq.write_table(table, tmp_path, compression="snappy")
        os.replace(tmp_path, part_path)
    except Exception:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        raise


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp_path, path)
    except Exception:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        raise


def _determine_output_dataset_dir(output_arg: Path) -> Path:
    """
    Determine where to write the parquet dataset directory.

    - If output_arg is a directory (or does not yet exist), we write there.
    - If a sidecar directory (<output>.parts) exists, we resume there.
    - If output_arg exists as a file, we write/resume into the sidecar directory (<output>.parts)
      to avoid clobbering the existing file.
    """
    sidecar_dir = Path(str(output_arg) + ".parts")

    if output_arg.exists():
        if output_arg.is_dir():
            return output_arg
        # Exists as a file: don't overwrite; use sidecar dir.
        print(
            f"Warning: output path exists as a file ({output_arg}). "
            f"Writing/resuming parquet dataset in sidecar directory: {sidecar_dir}",
            file=sys.stderr,
        )
        return sidecar_dir

    if sidecar_dir.exists():
        print(f"Found existing sidecar dataset directory, resuming: {sidecar_dir}", file=sys.stderr)
        return sidecar_dir

    return output_arg


def _sum_existing_part_rows(parts_dir: Path) -> int:
    if not parts_dir.exists():
        return 0
    total = 0
    for part in sorted(parts_dir.glob("rg*_rb*.parquet")):
        try:
            total += pq.ParquetFile(part).metadata.num_rows
        except Exception as e:
            print(f"Warning: failed reading parquet metadata for {part}: {e}", file=sys.stderr)
    return total


def _infer_model_id_for_state(model: AutoModel, model_id_arg: str) -> str:
    """
    Prefer the explicit CLI/HF id for stability across sessions.
    Some transformers models set name_or_path, but that may be a local cache path.
    """
    # CLI arg is the best user intent signal.
    if model_id_arg:
        return model_id_arg
    return getattr(model, "name_or_path", "") or ""


def process_parquet_file(
    input_path: Path,
    output_path: Path,
    model: AutoModel,
    processor: AutoImageProcessor,
    device: str,
    model_id: str,
    batch_size: int = 32,
    limit: Optional[int] = None,
    filter_valid: bool = True,
    read_batch_size: int = 1000,
) -> None:
    """
    Process parquet file, generate embeddings, and write to output.

    Args:
        input_path: Path to input parquet file
        output_path: Path to output parquet file
        model: Loaded model
        processor: Image processor
        device: Device to use
        batch_size: Batch size for processing images through model
        limit: Maximum number of rows to process (None for all)
        filter_valid: If True, only process rows where is_valid=True
        read_batch_size: Number of rows to read from parquet at a time
    """
    print(f"\nReading parquet file: {input_path}")

    if output_path.suffix == ".csv":
        raise ValueError(
            "CSV output is not supported for resumable incremental writes. "
            "Please specify a parquet dataset directory path (e.g., --output embeddings.parquet). "
            "The script will create a directory structure with multiple parquet part files."
        )

    # Determine output dataset directory and internal layout.
    output_dataset_dir = _determine_output_dataset_dir(output_path)
    parts_dir = output_dataset_dir / "parts"
    state_path = output_dataset_dir / "state.json"

    # Resume / compatibility checks.
    filter_valid_effective = filter_valid
    state: Optional[dict] = None
    model_id_for_state = _infer_model_id_for_state(model, model_id)
    if state_path.exists():
        state = _read_json(state_path)
        # Basic safety to avoid mixing incompatible outputs.
        # Only check critical parameters that affect output correctness.
        # Batch sizes can vary between runs without affecting correctness.
        critical_params = {
            "input_path": str(input_path),
            "model_id": model_id_for_state,
            "filter_valid": bool(filter_valid_effective),
        }
        for k, v in critical_params.items():
            existing_value = state.get(k)
            if existing_value != v:
                raise ValueError(
                    f"Existing state.json is incompatible for key '{k}'.\n"
                    f"Existing: {existing_value}\n"
                    f"Current:  {v}\n"
                    f"Refusing to mix outputs in {output_dataset_dir}."
                )

    # Check total rows first
    parquet_file = pq.ParquetFile(input_path)
    total_rows = parquet_file.metadata.num_rows
    num_row_groups = parquet_file.metadata.num_row_groups

    num_rows_to_process = limit if limit else total_rows
    num_rows_to_process = min(num_rows_to_process, total_rows)

    # Estimate already-produced embeddings by summing existing part row counts.
    produced = _sum_existing_part_rows(parts_dir)

    print(f"Total rows in file: {total_rows}")
    print(f"Rows to process (upper bound): {num_rows_to_process}")
    print(f"Output dataset directory: {output_dataset_dir}")
    print(f"Parts directory: {parts_dir}")
    if produced > 0:
        print(f"Resuming: detected {produced} embeddings already written")

    if limit is not None and produced >= limit:
        print(f"Limit {limit} already satisfied by existing output ({produced} embeddings). Exiting.")
        return

    # Initialize / update state.
    base_state = {
        "input_path": str(input_path),
        "output_dataset_dir": str(output_dataset_dir),
        "model_id": model_id_for_state,
        "device": device,
        "batch_size": int(batch_size),
        "read_batch_size": int(read_batch_size),
        "filter_valid": bool(filter_valid_effective),
        "total_rows_in_input": int(total_rows),
        "num_row_groups": int(num_row_groups),
    }
    if state is None:
        state = {**base_state, "embeddings_written": int(produced)}
    else:
        state.update(base_state)
        state["embeddings_written"] = int(produced)
    _atomic_write_json(state_path, state)

    # Process row-group-by-row-group for deterministic part naming.
    # NOTE: Some pyarrow versions (e.g. older cluster modules) do not support
    # ParquetFile.iter_batches(use_pandas=True). To stay compatible, we always
    # iterate over Arrow RecordBatches and convert to pandas explicitly.
    stop = False
    for rg in range(num_row_groups):
        if stop:
            break

        rb_idx = 0
        for batch in parquet_file.iter_batches(batch_size=read_batch_size, row_groups=[rg]):
            part_path = parts_dir / f"rg{rg:05d}_rb{rb_idx:05d}.parquet"
            rb_idx += 1

            # Skip deterministically if already written.
            if part_path.exists():
                continue

            if limit is not None and produced >= limit:
                stop = True
                break

            batch_df = batch.to_pandas()

            # Filter valid images if requested
            if filter_valid_effective and "is_valid" in batch_df.columns:
                batch_df = batch_df[batch_df["is_valid"]]

            if len(batch_df) == 0:
                _atomic_write_empty_parquet_part(part_path)
                print(f"Wrote empty part (no rows after filtering): {part_path}")
                continue

            # Extract image bytes and OSMIDs
            image_bytes_list: list[bytes] = []
            osmids_list: list[object] = []

            # Access columns directly instead of using iterrows() for better performance.
            image_bytes_col = batch_df.get("image_bytes")
            osmids_col = batch_df.get("OSMID")

            if image_bytes_col is not None:
                image_bytes_values = image_bytes_col.tolist()
                osmids_values = osmids_col.tolist() if osmids_col is not None else [None] * len(image_bytes_values)

                for img_bytes, osmid in zip(image_bytes_values, osmids_values):
                    if img_bytes is not None and len(img_bytes) > 0:
                        image_bytes_list.append(img_bytes)
                        osmids_list.append(osmid)

            if len(image_bytes_list) == 0:
                _atomic_write_empty_parquet_part(part_path)
                print(f"Wrote empty part (no usable image_bytes): {part_path}")
                continue

            # Respect --limit in terms of embeddings to produce.
            # Track if the batch was truncated to avoid writing a part file that would
            # prevent processing the remainder on subsequent runs without the limit.
            original_batch_size = len(image_bytes_list)
            batch_truncated_by_limit = False
            if limit is not None:
                remaining = limit - produced
                if remaining <= 0:
                    stop = True
                    break
                if len(image_bytes_list) > remaining:
                    image_bytes_list = image_bytes_list[:remaining]
                    osmids_list = osmids_list[:remaining]
                    batch_truncated_by_limit = True

            # Run inference for this chunk in model batch sizes.
            chunk_embeddings: list[np.ndarray] = []
            chunk_osmids: list[object] = []
            chunk_failures = 0

            for i in range(0, len(image_bytes_list), batch_size):
                batch_images = image_bytes_list[i : i + batch_size]
                batch_osmids = osmids_list[i : i + batch_size]
                try:
                    embs = generate_embeddings_batch(batch_images, model, processor, device)
                    chunk_embeddings.append(embs)
                    chunk_osmids.extend(batch_osmids)
                    produced += len(batch_images)
                except Exception as e:
                    print(f"Error processing model batch in chunk {part_path}: {e}", file=sys.stderr)
                    import traceback

                    traceback.print_exc()
                    chunk_failures += 1
                    continue

            # Don't write a part file if:
            # 1. All batches failed (write empty marker to prevent infinite retries)
            # 2. Some batches failed (skip part to retry on resume)
            # 3. Batch was truncated by --limit (skip part to process remainder later)
            if len(chunk_embeddings) == 0:
                # Write an empty marker to prevent infinite retries of consistently failing chunks.
                _atomic_write_empty_parquet_part(part_path)
                print(
                    f"Warning: all {chunk_failures} model batches failed for chunk. "
                    f"Wrote empty marker to prevent infinite retries: {part_path}",
                    file=sys.stderr,
                )
                continue

            if chunk_failures > 0:
                # Some (but not all) batches failed. Don't write the part file so
                # the entire chunk can be retried on resume to avoid data loss.
                print(
                    f"Warning: {chunk_failures} model batch(es) failed for chunk {part_path}. "
                    f"Skipping part write to allow retry on resume.",
                    file=sys.stderr,
                )
                continue

            if batch_truncated_by_limit:
                # Batch was truncated by --limit. Don't write the part file so
                # the remainder can be processed on subsequent runs without the limit.
                print(
                    f"Batch truncated by --limit (processed {len(image_bytes_list)} of {original_batch_size} total). "
                    f"Skipping part write to allow full batch processing on next run without limit."
                )
                continue

            embeddings_array = np.concatenate(chunk_embeddings, axis=0)
            _atomic_write_parquet_part(part_path, chunk_osmids, embeddings_array)

            # Update state after each successful part write.
            assert state is not None
            state["embeddings_written"] = int(produced)
            state["last_written_part"] = str(part_path)
            _atomic_write_json(state_path, state)

            if produced % 100 == 0 or (limit is not None and produced >= limit):
                print(f"Embeddings written: {produced}/{num_rows_to_process} (upper bound)")
            print(f"Wrote part: {part_path} ({embeddings_array.shape[0]} rows)")

            if limit is not None and produced >= limit:
                stop = True
                break

    if produced == 0:
        print("No embeddings generated!", file=sys.stderr)
        sys.exit(1)

    print(f"\nDone! Embeddings written so far: {produced}")
    print(f"Output dataset directory: {output_dataset_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate embeddings for images in packaged_images.parquet using Spatial Building Embeddings model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Path to input parquet file (packaged_images.parquet)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Path to output parquet dataset directory (e.g. embeddings.parquet). "
        "If a file already exists at that path, output will be written to a sidecar "
        "directory named '<output>.parts'.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=os.environ.get("MODEL_ID", "alexwaolson/spatial-building-embeddings"),
        help="HuggingFace model ID (default: alexwaolson/spatial-building-embeddings or MODEL_ID env var)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for model inference (default: 32)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of rows to process (default: all)",
    )
    parser.add_argument(
        "--no-filter-valid",
        action="store_true",
        help="Process all images, not just those with is_valid=True",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu", "auto"],
        default="auto",
        help="Device to use for inference (default: auto-detect)",
    )

    args = parser.parse_args()

    # Validate input file
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # Load model
    try:
        model, processor = load_model_from_hub(args.model_id, device=device)
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        print("\nTip: Make sure you have internet connection and the model ID is correct.", file=sys.stderr)
        print("For gated models, ensure HF_TOKEN environment variable is set.", file=sys.stderr)
        sys.exit(1)

    # Process parquet file
    try:
        process_parquet_file(
            input_path=args.input,
            output_path=args.output,
            model=model,
            processor=processor,
            device=device,
            model_id=args.model_id,
            batch_size=args.batch_size,
            limit=args.limit,
            filter_valid=not args.no_filter_valid,
        )
    except Exception as e:
        print(f"Error processing parquet file: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
