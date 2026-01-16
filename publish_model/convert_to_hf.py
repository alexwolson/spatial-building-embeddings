#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import torch
import logging

# Add project root to sys.path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config import (
    load_config_from_file,
    TripletTrainingConfig,
    GenerateEmbeddingsConfig,
)
from publish_model.configuration_spatial_embeddings import SpatialEmbeddingsConfig
from publish_model.modeling_spatial_embeddings import SpatialEmbeddingsModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def infer_architecture_from_checkpoint(
    state_dict: dict[str, torch.Tensor], logger: logging.Logger
) -> dict:
    """
    Infer architecture parameters from checkpoint state_dict.
    
    Returns a dict with: input_dim, hidden_dim, output_dim, num_hidden_layers,
    use_residual, use_layer_norm, hidden_dim_multiplier
    """
    # Infer input_dim from input_layer or residual_proj
    if "input_layer.weight" in state_dict:
        input_dim = state_dict["input_layer.weight"].shape[1]
        first_hidden_dim = state_dict["input_layer.weight"].shape[0]
    elif "residual_proj.weight" in state_dict:
        input_dim = state_dict["residual_proj.weight"].shape[1]
        # Can't infer first_hidden_dim from residual, need to check hidden_layers
        first_hidden_dim = None
    else:
        raise ValueError("Cannot infer input_dim: no input_layer or residual_proj in checkpoint")
    
    # Infer output_dim from output_layer or residual_proj
    if "output_layer.weight" in state_dict:
        output_dim = state_dict["output_layer.weight"].shape[0]
    elif "residual_proj.weight" in state_dict:
        output_dim = state_dict["residual_proj.weight"].shape[0]
    else:
        raise ValueError("Cannot infer output_dim: no output_layer or residual_proj in checkpoint")
    
    # Infer first hidden dim if not already known
    if first_hidden_dim is None:
        # Check if there are hidden layers
        hidden_layer_keys = [k for k in state_dict.keys() if k.startswith("hidden_layers.") and k.endswith(".weight")]
        if hidden_layer_keys:
            # When input_layer is absent, use shape[1] (input dimension) of first hidden layer
            # Linear layer weights have shape (out_features, in_features)
            first_hidden_key = sorted(hidden_layer_keys)[0]
            first_hidden_dim = state_dict[first_hidden_key].shape[1]
        else:
            # No hidden layers, use output_dim as hidden_dim (single layer)
            first_hidden_dim = output_dim
    
    # Count hidden layers
    # Note: when num_hidden_layers=1, there are no hidden_layers.* entries
    # (only input_layer and output_layer exist)
    # When num_hidden_layers=N (N>1), there are N-1 hidden_layers.* entries
    hidden_layer_keys = [k for k in state_dict.keys() if k.startswith("hidden_layers.") and ".weight" in k]
    num_hidden_layers = len(hidden_layer_keys) + 1  # +1 because input_layer counts as one
    
    # Infer hidden_dim_multiplier from hidden layer dimensions
    hidden_dim_multiplier = 1.0
    if num_hidden_layers > 1:
        # Collect dimensions: input_layer output and hidden_layers outputs
        # input_layer outputs first_hidden_dim
        # hidden_layers[i] outputs the (i+1)th hidden dimension
        hidden_dims = [first_hidden_dim]  # Start with input_layer output
        
        # There are num_hidden_layers - 1 entries in hidden_layers (indexed 0 to num_hidden_layers-2)
        for i in range(num_hidden_layers - 1):
            key = f"hidden_layers.{i}.weight"
            if key in state_dict:
                # The output dimension of hidden_layers[i] is shape[0]
                hidden_dims.append(state_dict[key].shape[0])
        
        if len(hidden_dims) >= 2:
            # Calculate multiplier from first two dimensions
            multiplier = hidden_dims[1] / hidden_dims[0] if hidden_dims[0] > 0 else 1.0
            hidden_dim_multiplier = round(multiplier, 2)
    
    # Infer use_residual
    use_residual = "residual_proj.weight" in state_dict
    
    # Infer use_layer_norm
    use_layer_norm = "input_norm.weight" in state_dict or "output_norm.weight" in state_dict
    
    logger.info(f"Inferred architecture from checkpoint:")
    logger.info(f"  input_dim: {input_dim}")
    logger.info(f"  hidden_dim: {first_hidden_dim}")
    logger.info(f"  output_dim: {output_dim}")
    logger.info(f"  num_hidden_layers: {num_hidden_layers}")
    logger.info(f"  hidden_dim_multiplier: {hidden_dim_multiplier}")
    logger.info(f"  use_residual: {use_residual}")
    logger.info(f"  use_layer_norm: {use_layer_norm}")
    
    return {
        "input_dim": input_dim,
        "hidden_dim": first_hidden_dim,
        "output_dim": output_dim,
        "num_hidden_layers": num_hidden_layers,
        "hidden_dim_multiplier": hidden_dim_multiplier,
        "use_residual": use_residual,
        "use_layer_norm": use_layer_norm,
    }


def main():
    # 1. Load configuration
    config_path = project_root / "config.toml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found at {config_path}")

    # We need training config for architecture params and embedding config for backbone name
    train_config = load_config_from_file(config_path, "triplet_training")
    embed_config = load_config_from_file(config_path, "generate_embeddings")

    logger.info("Loaded configuration from config.toml")
    logger.info(f"Backbone: {embed_config.model_name}")

    # 2. Load Checkpoint to Infer Architecture
    checkpoint_path = project_root / "data/checkpoints/checkpoint_best.pt"
    if not checkpoint_path.exists():
        # Fallback to config path
        checkpoint_path = train_config.checkpoint_dir / "checkpoint_best.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path} or {train_config.checkpoint_dir}"
        )

    logger.info(f"Loading checkpoint from {checkpoint_path} to infer architecture...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"]
    
    # Infer architecture from checkpoint (this is the source of truth)
    inferred_arch = infer_architecture_from_checkpoint(state_dict, logger)
    
    # Use inferred architecture, but fall back to config for things we can't infer
    # (dropout, activation) or things that might be configurable (dropout)
    hf_backbone = embed_config.model_name
    logger.info(f"Using backbone: {hf_backbone}")

    hf_config = SpatialEmbeddingsConfig(
        backbone_model_name=hf_backbone,
        input_dim=inferred_arch["input_dim"],
        hidden_dim=inferred_arch["hidden_dim"],
        output_dim=inferred_arch["output_dim"],
        dropout=train_config.dropout,  # Can't infer from weights, use config
        num_hidden_layers=inferred_arch["num_hidden_layers"],
        hidden_dim_multiplier=inferred_arch["hidden_dim_multiplier"],
        activation=train_config.activation,  # Can't infer from weights, use config
        use_residual=inferred_arch["use_residual"],
        use_layer_norm=inferred_arch["use_layer_norm"],
    )

    # 3. Initialize Model
    logger.info("Initializing SpatialEmbeddingsModel...")
    model = SpatialEmbeddingsModel(hf_config)

    # 4. Load Checkpoint Weights
    # Prefix keys with 'projector.' since the checkpoint is just the projector
    projector_state_dict = {f"projector.{k}": v for k, v in state_dict.items()}

    # Load projector weights
    missing, unexpected = model.load_state_dict(projector_state_dict, strict=False)

    # We expect missing keys for the backbone (since we initialized it from pretrained,
    # but load_state_dict with strict=False and partial dict won't load backbone weights from checkpoint
    # - which is correct because checkpoint ONLY has projector).
    # However, we want to ensure the backbone is loaded.
    # The `SpatialEmbeddingsModel.__init__` calls `AutoModel.from_pretrained`, so backbone is already loaded with pretrained weights.
    # We just need to ensure we didn't have unexpected keys.
    if unexpected:
        raise RuntimeError(f"Unexpected keys in checkpoint: {unexpected}")

    logger.info("Successfully loaded projector weights.")

    # 5. Save Pretrained Model
    # Use scratch directory to avoid disk quota issues on project filesystem
    # Follow Alliance standard: use $SCRATCH env var, fallback to ~/scratch pattern
    scratch_base = os.environ.get("SCRATCH")
    if scratch_base is None:
        # Fallback to HOME/scratch pattern (as used in config.toml)
        home = os.environ.get("HOME", "/home/awolson")
        scratch_base = os.path.join(home, "scratch")
    
    output_dir = Path(scratch_base) / "spatial-building-embeddings" / "published_model"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving model to {output_dir}")
    model.save_pretrained(output_dir)
    # Also save the config separately to be sure (save_pretrained does this)
    # And copy the python files for custom code
    import shutil

    shutil.copy(
        project_root / "publish_model/configuration_spatial_embeddings.py", output_dir
    )
    shutil.copy(
        project_root / "publish_model/modeling_spatial_embeddings.py", output_dir
    )

    logger.info(f"Done! Model is ready in {output_dir}")


if __name__ == "__main__":
    main()
