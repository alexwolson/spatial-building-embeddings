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
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
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


def process_parquet_file(
    input_path: Path,
    output_path: Path,
    model: AutoModel,
    processor: AutoImageProcessor,
    device: str,
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
    
    # Check total rows first
    parquet_file = pq.ParquetFile(input_path)
    total_rows = parquet_file.metadata.num_rows
    num_rows_to_process = limit if limit else total_rows
    num_rows_to_process = min(num_rows_to_process, total_rows)
    
    print(f"Total rows in file: {total_rows}")
    print(f"Rows to process: {num_rows_to_process}")

    # Process in batches
    all_embeddings = []
    all_osmids = []
    processed = 0

    # Read batches from parquet using iter_batches.
    # NOTE: Some pyarrow versions (e.g. older cluster modules) do not support
    # ParquetFile.iter_batches(use_pandas=True). To stay compatible, we always
    # iterate over Arrow RecordBatches and convert to pandas explicitly.
    batch_iter = parquet_file.iter_batches(batch_size=read_batch_size)

    for batch in batch_iter:
        # `batch` is typically a pyarrow.RecordBatch
        batch_df = batch.to_pandas()
        if processed >= num_rows_to_process:
            break
        
        # Filter valid images if requested
        if filter_valid and "is_valid" in batch_df.columns:
            batch_df = batch_df[batch_df["is_valid"] == True]
        
        if len(batch_df) == 0:
            continue
        
        # Limit rows if specified
        remaining = num_rows_to_process - processed
        if len(batch_df) > remaining:
            batch_df = batch_df.head(remaining)
        
        # Extract image bytes and OSMIDs
        image_batches = []
        osmids_batch = []
        
        for idx, row in batch_df.iterrows():
            image_bytes = row.get("image_bytes")
            osmid = row.get("OSMID")
            
            if image_bytes is not None and len(image_bytes) > 0:
                image_batches.append(image_bytes)
                osmids_batch.append(osmid)
        
        # Process images in model batch size
        for i in range(0, len(image_batches), batch_size):
            batch_images = image_batches[i : i + batch_size]
            batch_osmids = osmids_batch[i : i + batch_size]
            
            try:
                embeddings = generate_embeddings_batch(batch_images, model, processor, device)
                all_embeddings.append(embeddings)
                all_osmids.extend(batch_osmids)
                processed += len(batch_images)
                
                if processed % 100 == 0 or processed >= num_rows_to_process:
                    print(f"Processed {processed}/{num_rows_to_process} images...")
            except Exception as e:
                print(f"Error processing batch: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc()
                # Continue with next batch
                continue
        
        if processed >= num_rows_to_process:
            break

    # Concatenate all embeddings
    if len(all_embeddings) == 0:
        print("No embeddings generated!", file=sys.stderr)
        sys.exit(1)
    
    embeddings_array = np.concatenate(all_embeddings, axis=0)
    print(f"\nGenerated {len(embeddings_array)} embeddings")
    print(f"Embedding shape: {embeddings_array.shape}")
    print(f"Mean embedding norm: {np.linalg.norm(embeddings_array, axis=1).mean():.4f}")

    # Create output DataFrame
    output_df = pd.DataFrame(
        {
            "OSMID": all_osmids,
            "embedding": [emb for emb in embeddings_array],
        }
    )

    # If embeddings should be stored as arrays in parquet, we need to convert
    # Parquet supports arrays, so we can store as list
    output_df["embedding"] = output_df["embedding"].apply(lambda x: x.tolist())

    # Write output
    print(f"\nWriting embeddings to: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.suffix == ".parquet":
        output_df.to_parquet(output_path, index=False, compression="snappy")
    elif output_path.suffix == ".csv":
        # For CSV, we might want to flatten or store differently
        # Option 1: Store as space-separated values
        output_df["embedding"] = output_df["embedding"].apply(lambda x: " ".join(map(str, x)))
        output_df.to_csv(output_path, index=False)
    else:
        # Default to parquet
        output_path = output_path.with_suffix(".parquet")
        output_df.to_parquet(output_path, index=False, compression="snappy")
        print(f"Changed output path to: {output_path}")

    print(f"Done! Wrote {len(output_df)} embeddings to {output_path}")


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
        help="Path to output file (.parquet or .csv)",
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
