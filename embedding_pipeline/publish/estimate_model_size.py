#!/usr/bin/env python3
"""
Estimate the memory size of the Spatial Embeddings Model when loaded into memory.

This script calculates the size based on:
1. DINOv3 ViT-7B backbone (~7B parameters)
2. Custom projector head (~3M parameters)
"""


# Model architecture (from checkpoint inference)
PROJECTOR_CONFIG = {
    "input_dim": 4096,
    "hidden_dim": 384,
    "output_dim": 256,
    "num_hidden_layers": 3,
    "hidden_dim_multiplier": 1.0,
    "use_residual": True,
    "use_layer_norm": False,
}

# DINOv3 ViT-7B approximate parameter count
# Source: Based on Vision Transformer architecture and HuggingFace model info
DINOV3_VIT7B_PARAMS = 7_000_000_000  # ~7 billion parameters


def estimate_projector_params(config: dict) -> int:
    """Estimate number of parameters in the projector head."""
    input_dim = config["input_dim"]
    hidden_dim = config["hidden_dim"]
    output_dim = config["output_dim"]
    num_hidden_layers = config["num_hidden_layers"]
    use_residual = config["use_residual"]
    use_layer_norm = config["use_layer_norm"]
    multiplier = config.get("hidden_dim_multiplier", 1.0)

    # Compute hidden dimensions
    hidden_dims = [hidden_dim]
    current_dim = hidden_dim
    for i in range(1, num_hidden_layers):
        current_dim = max(16, int(round(current_dim * multiplier)))
        hidden_dims.append(current_dim)

    # Count parameters
    params = 0

    # Input layer: input_dim -> hidden_dims[0]
    params += input_dim * hidden_dims[0] + hidden_dims[0]  # weights + bias

    # Hidden layers (if any)
    for i in range(1, len(hidden_dims)):
        params += hidden_dims[i - 1] * hidden_dims[i] + hidden_dims[i]  # weights + bias

    # Output layer: last_hidden_dim -> output_dim
    params += hidden_dims[-1] * output_dim + output_dim  # weights + bias

    # Residual projection (if used)
    if use_residual:
        params += input_dim * output_dim + output_dim  # weights + bias

    # LayerNorm parameters (if used) - 2 params per layer (weight + bias)
    if use_layer_norm:
        # Input norm
        params += 2 * hidden_dims[0]
        # Hidden norms
        for dim in hidden_dims[1:]:
            params += 2 * dim
        # Output norm
        params += 2 * output_dim

    return params


def format_bytes(bytes: int) -> str:
    """Format bytes into human-readable format."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes < 1024.0:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.2f} PB"


def main():
    print("Spatial Embeddings Model Memory Estimate")
    print("=" * 60)

    # Estimate projector parameters
    projector_params = estimate_projector_params(PROJECTOR_CONFIG)

    print("\nProjector Head:")
    print(f"  Parameters: {projector_params:,}")

    # Total parameters
    total_params = DINOV3_VIT7B_PARAMS + projector_params

    print("\nBackbone (DINOv3 ViT-7B):")
    print(f"  Parameters: {DINOV3_VIT7B_PARAMS:,}")

    print("\nTotal Model Parameters:")
    print(f"  {total_params:,}")

    # Memory estimates for different precision
    print(f"\n{'Precision':<15} {'Memory (RAM/VRAM)':<20} {'Notes'}")
    print("-" * 60)

    # Float32 (FP32) - 4 bytes per parameter
    fp32_bytes = total_params * 4
    print(f"{'FP32 (float32)':<15} {format_bytes(fp32_bytes):<20} Standard precision")

    # Float16 (FP16) - 2 bytes per parameter
    fp16_bytes = total_params * 2
    print(f"{'FP16 (float16)':<15} {format_bytes(fp16_bytes):<20} Half precision")

    # BFloat16 (BF16) - 2 bytes per parameter
    bf16_bytes = total_params * 2
    print(
        f"{'BF16 (bfloat16)':<15} {format_bytes(bf16_bytes):<20} Brain float (similar to FP16)"
    )

    # Int8 quantization - 1 byte per parameter
    int8_bytes = total_params * 1
    print(
        f"{'Int8 (quantized)':<15} {format_bytes(int8_bytes):<20} Quantized (lower quality)"
    )

    # Memory overhead (optimizer states, activations, etc.)
    print("\nAdditional Memory Considerations:")
    print("  - Optimizer states: ~2-3x model size (for training)")
    print("  - Activations: Varies with batch size and sequence length")
    print("  - Gradient memory: ~1x model size (for training)")

    # Inference memory estimate (model + small overhead)
    inference_overhead = 0.1  # 10% overhead for activations, buffers, etc.
    fp16_inference_bytes = fp16_bytes * (1 + inference_overhead)

    print("\nInference Memory Estimate (FP16, minimal batch):")
    print(f"  Model weights: {format_bytes(fp16_bytes)}")
    print(f"  With overhead (~10%): {format_bytes(fp16_inference_bytes)}")

    # Batch processing estimate
    batch_size = 16  # From config.toml
    image_size = 224  # Typical DINO input size

    # Activation memory per batch (rough estimate)
    # Backbone activations (simplified): batch_size * num_patches * hidden_dim * 4 bytes (FP32)
    # DINOv3 ViT-7B: patch_size=16, image_size=224 -> 14x14 patches = 196 patches
    num_patches = (image_size // 16) ** 2  # 196 patches
    hidden_dim = 4096  # DINOv3 ViT-7B hidden dimension

    # Rough activation memory (excluding intermediate layers)
    activation_memory_per_image = num_patches * hidden_dim * 4  # FP32
    batch_activation_memory = activation_memory_per_image * batch_size

    fp16_batch_memory = fp16_bytes + batch_activation_memory
    print(f"\nInference with Batch Size {batch_size} (FP16):")
    print(f"  Model weights: {format_bytes(fp16_bytes)}")
    print(f"  Activations: ~{format_bytes(batch_activation_memory)}")
    print(f"  Total: ~{format_bytes(fp16_batch_memory)}")

    print("\nRecommendations:")
    print("  - GPU with >= 40GB VRAM (FP16, batch_size=16)")
    print("  - GPU with >= 80GB VRAM (FP16, larger batches)")
    print("  - CPU inference possible but much slower")
    print("  - Consider quantization for edge devices")


if __name__ == "__main__":
    main()
