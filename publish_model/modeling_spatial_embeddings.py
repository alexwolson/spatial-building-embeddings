import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, AutoModel
from .configuration_spatial_embeddings import SpatialEmbeddingsConfig
from typing import Optional, Tuple, Union, Literal


class EmbeddingProjector(nn.Module):
    """
    Configurable MLP projection head for embedding transformation.
    (Copied from train_specialized_embeddings/model.py for self-contained publishing)
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 512,
        output_dim: int = 256,
        dropout: float = 0.1,
        num_hidden_layers: int = 1,
        hidden_dim_multiplier: float = 1.0,
        activation: Literal["gelu", "relu", "silu"] = "gelu",
        use_residual: bool = True,
        use_layer_norm: bool = True,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_residual = use_residual
        self.use_layer_norm = use_layer_norm
        self.num_hidden_layers = num_hidden_layers
        self.hidden_dim_multiplier = hidden_dim_multiplier
        self.activation_name = activation

        self.hidden_dims = self._compute_hidden_dims(
            hidden_dim, num_hidden_layers, hidden_dim_multiplier
        )

        self.activation = self._resolve_activation(activation)

        # First hidden block
        self.input_layer = nn.Linear(input_dim, self.hidden_dims[0])
        if use_layer_norm:
            self.input_norm = nn.LayerNorm(self.hidden_dims[0])
        self.input_dropout = nn.Dropout(dropout)

        # Additional hidden blocks (if any)
        self.hidden_layers = nn.ModuleList()
        if use_layer_norm:
            self.hidden_norms = nn.ModuleList()
        else:
            self.hidden_norms = None
        self.hidden_dropouts = nn.ModuleList()

        for idx in range(1, len(self.hidden_dims)):
            layer = nn.Linear(self.hidden_dims[idx - 1], self.hidden_dims[idx])
            self.hidden_layers.append(layer)
            if use_layer_norm:
                self.hidden_norms.append(nn.LayerNorm(self.hidden_dims[idx]))
            self.hidden_dropouts.append(nn.Dropout(dropout))

        # Output block
        self.output_layer = nn.Linear(self.hidden_dims[-1], output_dim)
        if use_layer_norm:
            self.output_norm = nn.LayerNorm(output_dim)
        self.output_dropout = nn.Dropout(dropout)

        # Residual shortcut (projects input directly to output)
        if use_residual:
            self.residual_proj = nn.Linear(input_dim, output_dim)

    @staticmethod
    def _compute_hidden_dims(
        base_hidden_dim: int, num_layers: int, multiplier: float
    ) -> list[int]:
        dims: list[int] = []
        current_dim = base_hidden_dim
        for layer_idx in range(num_layers):
            if layer_idx == 0:
                dims.append(base_hidden_dim)
            else:
                current_dim = max(16, int(round(current_dim * multiplier)))
                dims.append(current_dim)
        return dims

    @staticmethod
    def _resolve_activation(name: str) -> nn.Module:
        if name == "gelu":
            return nn.GELU()
        if name == "relu":
            return nn.ReLU()
        if name == "silu":
            return nn.SiLU()
        raise ValueError(f"Unsupported activation: {name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First hidden block
        out = self.input_layer(x)
        if self.use_layer_norm:
            out = self.input_norm(out)
        out = self.activation(out)
        out = self.input_dropout(out)

        # Additional hidden blocks
        for idx, layer in enumerate(self.hidden_layers):
            out = layer(out)
            if self.use_layer_norm and self.hidden_norms is not None:
                out = self.hidden_norms[idx](out)
            out = self.activation(out)
            out = self.hidden_dropouts[idx](out)

        # Output block
        out = self.output_layer(out)
        if self.use_layer_norm:
            out = self.output_norm(out)
        out = self.output_dropout(out)

        # Residual connection
        if self.use_residual:
            residual = self.residual_proj(x)
            out = out + residual

        # L2 normalization
        out = F.normalize(out, p=2, dim=1)

        return out


class SpatialEmbeddingsModel(PreTrainedModel):
    config_class = SpatialEmbeddingsConfig

    def __init__(self, config: SpatialEmbeddingsConfig):
        super().__init__(config)
        self.config = config
        
        # Initialize backbone
        self.backbone = AutoModel.from_pretrained(config.backbone_model_name, trust_remote_code=True, safe_serialization=True)
        
        # Initialize projector
        self.projector = EmbeddingProjector(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.output_dim,
            dropout=config.dropout,
            num_hidden_layers=config.num_hidden_layers,
            hidden_dim_multiplier=config.hidden_dim_multiplier,
            activation=config.activation,
            use_residual=config.use_residual,
            use_layer_norm=config.use_layer_norm,
        )

    def forward(
        self, 
        pixel_values: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, torch.Tensor]:
        """
        Args:
            pixel_values: Tensor of shape (batch_size, channels, height, width)
            return_dict: Whether to return a dictionary or tuple
        
        Returns:
            If return_dict is True (default for HF), returns object with 'embeddings'.
            Otherwise returns (embeddings,).
        """
        # Pass through backbone
        outputs = self.backbone(pixel_values=pixel_values, return_dict=True, **kwargs)
        
        # Extract pooled output (CLS token or similar)
        # DINOv2 outputs pooler_output in some versions, or last_hidden_state
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            backbone_emb = outputs.pooler_output
        else:
            # Fallback: Use CLS token from last hidden state
            backbone_emb = outputs.last_hidden_state[:, 0]

        # Project to specialized embedding
        specialized_emb = self.projector(backbone_emb)

        if return_dict:
            return {"embeddings": specialized_emb, "backbone_outputs": outputs}
        return (specialized_emb,)

