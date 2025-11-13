"""
Model architecture for specialized embedding projection.

Provides a two-layer MLP projection head that maps precomputed embeddings
to a lower-dimensional specialized embedding space.
"""

import logging
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class EmbeddingProjector(nn.Module):
    """
    Configurable MLP projection head for embedding transformation.

    Architecture:
    - Input: 768-dim (DINOv2 base)
    - Hidden stack: configurable depth/width (default single 512-dim layer)
    - Output: configurable (default 256)
    - Activation: selectable (default GELU)
    - Normalization: LayerNorm (optional)
    - Regularization: Dropout
    - Shortcut: Residual connection (optional)
    - Output: L2-normalized
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
        """
        Initialize embedding projector.

        Args:
            input_dim: Input embedding dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output embedding dimension
            dropout: Dropout probability
            num_hidden_layers: Number of hidden layers before the output projection
            hidden_dim_multiplier: Factor applied to hidden dims for deeper layers
            activation: Activation function applied after each hidden layer
            use_residual: Whether to use residual shortcut
            use_layer_norm: Whether to use LayerNorm
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_residual = use_residual
        self.use_layer_norm = use_layer_norm
        self.num_hidden_layers = num_hidden_layers
        self.hidden_dim_multiplier = hidden_dim_multiplier
        self.activation_name = activation

        if num_hidden_layers < 1:
            raise ValueError("num_hidden_layers must be >= 1")
        if hidden_dim_multiplier <= 0:
            raise ValueError("hidden_dim_multiplier must be > 0")

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

        logger.info(
            "Initialized EmbeddingProjector: %s -> %s -> %s (layers=%d, multiplier=%.2f, "
            "activation=%s, residual=%s, layer_norm=%s, dropout=%.2f)",
            input_dim,
            self.hidden_dims,
            output_dim,
            num_hidden_layers,
            hidden_dim_multiplier,
            activation,
            use_residual,
            use_layer_norm,
            dropout,
        )

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
    def _resolve_activation(name: Literal["gelu", "relu", "silu"]) -> nn.Module:
        if name == "gelu":
            return nn.GELU()
        if name == "relu":
            return nn.ReLU()
        if name == "silu":
            return nn.SiLU()
        raise ValueError(f"Unsupported activation: {name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input embeddings [batch_size, input_dim]

        Returns:
            Projected embeddings [batch_size, output_dim], L2-normalized
        """
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


class TripletLossWrapper:
    """
    Wrapper around PyTorch triplet loss with reward tracking for UCB sampler.
    """

    def __init__(
        self,
        margin: float = 0.5,
        distance: Literal["euclidean", "cosine"] = "euclidean",
    ):
        """
        Initialize triplet loss wrapper.

        Args:
            margin: Triplet loss margin
            distance: Distance metric ("euclidean" or "cosine")
        """
        self.margin = margin
        self.distance = distance

        if distance == "euclidean":
            self.loss_fn = nn.TripletMarginLoss(margin=margin, p=2)
        elif distance == "cosine":
            # Use cosine distance variant
            self.loss_fn = nn.TripletMarginWithDistanceLoss(
                margin=margin,
                distance_function=lambda x, y: 1 - F.cosine_similarity(x, y),
            )
        else:
            raise ValueError(f"Unknown distance metric: {distance}")

        logger.info(
            f"Initialized TripletLossWrapper: margin={margin}, distance={distance}"
        )

    def __call__(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute triplet loss.

        Args:
            anchor: Anchor embeddings [batch_size, embedding_dim]
            positive: Positive embeddings [batch_size, embedding_dim]
            negative: Negative embeddings [batch_size, embedding_dim]

        Returns:
            Loss value
        """
        return self.loss_fn(anchor, positive, negative)

    def compute_reward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> float:
        """
        Compute reward for UCB sampler (negative loss, higher is better).

        Args:
            anchor: Anchor embeddings [batch_size, embedding_dim]
            positive: Positive embeddings [batch_size, embedding_dim]
            negative: Negative embeddings [batch_size, embedding_dim]

        Returns:
            Reward value (negative loss)
        """
        loss = self(anchor, positive, negative)
        # Reward is negative loss (higher loss = lower reward)
        return -loss.item()

