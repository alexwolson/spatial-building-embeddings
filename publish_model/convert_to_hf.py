#!/usr/bin/env python3
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
    logger.info(
        f"Architecture: {train_config.input_dim} -> {train_config.hidden_dim} -> {train_config.output_dim}"
    )

    # 2. Create HF Config
    hf_backbone = embed_config.model_name
    logger.info(f"Using backbone: {hf_backbone}")

    hf_config = SpatialEmbeddingsConfig(
        backbone_model_name=hf_backbone,
        input_dim=train_config.input_dim,
        hidden_dim=train_config.hidden_dim,
        output_dim=train_config.output_dim,
        dropout=train_config.dropout,
        num_hidden_layers=train_config.num_hidden_layers,
        hidden_dim_multiplier=train_config.hidden_dim_multiplier,
        activation=train_config.activation,
        use_residual=train_config.use_residual,
        use_layer_norm=train_config.use_layer_norm,
    )

    # 3. Initialize Model
    logger.info("Initializing SpatialEmbeddingsModel...")
    model = SpatialEmbeddingsModel(hf_config)

    # 4. Load Checkpoint Weights
    checkpoint_path = project_root / "data/checkpoints/checkpoint_best.pt"
    if not checkpoint_path.exists():
        # Fallback to config path
        checkpoint_path = train_config.checkpoint_dir / "checkpoint_best.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path} or {train_config.checkpoint_dir}"
        )

    logger.info(f"Loading weights from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"]

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

    # 5. Save Pretrained
    output_dir = project_root / "published_model"
    output_dir.mkdir(exist_ok=True)

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

    logger.info("Done! Model is ready in 'published_model/'")


if __name__ == "__main__":
    main()
