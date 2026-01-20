#!/usr/bin/env python3
"""
Generate embeddings for images using the published Spatial Building Embeddings model from HuggingFace Hub.

This script demonstrates how to load the model from HuggingFace Hub and generate embeddings for one or more images.

Usage:
    # Single image
    uv run python -m embedding_pipeline.publish.generate_embeddings_from_hf path/to/image.jpg

    # Multiple images
    uv run python -m embedding_pipeline.publish.generate_embeddings_from_hf image1.jpg image2.jpg image3.jpg

    # Using custom model ID
    MODEL_ID="custom/org/model-name" uv run python -m embedding_pipeline.publish.generate_embeddings_from_hf image.jpg

    # Save embeddings to file (numpy .npy format)
    uv run python -m embedding_pipeline.publish.generate_embeddings_from_hf image.jpg --output embeddings.npy
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


def load_model_from_hub(
    model_id: str, device: Optional[str] = None
) -> tuple[AutoModel, AutoImageProcessor]:
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


def generate_embedding(
    image_path: Path,
    model: AutoModel,
    processor: AutoImageProcessor,
    device: str,
) -> np.ndarray:
    """
    Generate embedding for a single image.

    Args:
        image_path: Path to the image file
        model: Loaded SpatialEmbeddingsModel
        processor: Image processor
        device: Device to run inference on

    Returns:
        Numpy array of shape (256,) containing the embedding
    """
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate embedding
    with torch.no_grad():
        outputs = model(**inputs)
        # The model returns a dict with 'embeddings' key or a tuple
        if isinstance(outputs, dict):
            embeddings = outputs["embeddings"]
        else:
            embeddings = outputs[0]

        # Convert to numpy (remove batch dimension if present)
        if embeddings.dim() > 1:
            embeddings = embeddings.squeeze(0)
        embeddings_np = embeddings.cpu().numpy()

    return embeddings_np


def generate_embeddings_batch(
    image_paths: list[Path],
    model: AutoModel,
    processor: AutoImageProcessor,
    device: str,
    batch_size: int = 8,
) -> np.ndarray:
    """
    Generate embeddings for multiple images in batches.

    Args:
        image_paths: List of paths to image files
        model: Loaded SpatialEmbeddingsModel
        processor: Image processor
        device: Device to run inference on
        batch_size: Number of images to process per batch

    Returns:
        Numpy array of shape (num_images, 256) containing embeddings
    """
    all_embeddings = []
    device_obj = torch.device(device)

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        batch_images = []

        # Load and preprocess batch
        for path in batch_paths:
            image = Image.open(path).convert("RGB")
            batch_images.append(image)

        # Process batch
        inputs = processor(images=batch_images, return_tensors="pt")
        inputs = {k: v.to(device_obj) for k, v in inputs.items()}

        # Generate embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            if isinstance(outputs, dict):
                embeddings = outputs["embeddings"]
            else:
                embeddings = outputs[0]

            # Move to CPU and convert to numpy
            embeddings_np = embeddings.cpu().numpy()
            all_embeddings.append(embeddings_np)

        print(
            f"Processed {min(i + batch_size, len(image_paths))}/{len(image_paths)} images..."
        )

    # Concatenate all batches
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    return all_embeddings


def main():
    parser = argparse.ArgumentParser(
        description="Generate embeddings for images using Spatial Building Embeddings model from HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "images",
        nargs="+",
        type=Path,
        help="Path(s) to image file(s)",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="alexwaolson/spatial-building-embeddings",
        help="HuggingFace model ID (default: alexwaolson/spatial-building-embeddings)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Optional: Save embeddings to .npy file (for batch processing)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for processing multiple images (default: 8)",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu", "auto"],
        default="auto",
        help="Device to use for inference (default: auto-detect)",
    )

    args = parser.parse_args()

    # Validate image files exist
    for image_path in args.images:
        if not image_path.exists():
            print(f"Error: Image file not found: {image_path}", file=sys.stderr)
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
        print(
            "\nTip: Make sure you have internet connection and the model ID is correct.",
            file=sys.stderr,
        )
        print(
            "For gated models, ensure HF_TOKEN environment variable is set.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Generate embeddings
    if len(args.images) == 1:
        # Single image
        print(f"\nGenerating embedding for: {args.images[0]}")
        embedding = generate_embedding(args.images[0], model, processor, device)
        print(f"Embedding shape: {embedding.shape}")
        print(f"Embedding norm: {np.linalg.norm(embedding):.4f}")
        print(f"First 5 values: {embedding[:5]}")
    else:
        # Multiple images
        print(f"\nGenerating embeddings for {len(args.images)} images...")
        embeddings = generate_embeddings_batch(
            args.images, model, processor, device, args.batch_size
        )
        print(f"Embeddings shape: {embeddings.shape}")
        print(f"Mean embedding norm: {np.linalg.norm(embeddings, axis=1).mean():.4f}")

    # Save to file if requested
    if args.output:
        if len(args.images) == 1:
            np.save(args.output, embedding)
        else:
            np.save(args.output, embeddings)
        print(f"\nSaved embeddings to: {args.output}")

    print("\nDone!")


if __name__ == "__main__":
    main()
