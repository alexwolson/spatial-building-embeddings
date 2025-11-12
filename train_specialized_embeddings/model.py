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
    Two-layer MLP projection head for embedding transformation.

    Architecture:
    - Input: 768-dim (DINOv2 base)
    - Hidden: configurable (default 512)
    - Output: configurable (default 256)
    - Activation: GELU
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
            use_residual: Whether to use residual shortcut
            use_layer_norm: Whether to use LayerNorm
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_residual = use_residual
        self.use_layer_norm = use_layer_norm

        # First layer
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        if use_layer_norm:
            self.ln1 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)

        # Second layer
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        if use_layer_norm:
            self.ln2 = nn.LayerNorm(output_dim)
        self.dropout2 = nn.Dropout(dropout)

        # Residual shortcut (projects input directly to output)
        if use_residual:
            self.residual_proj = nn.Linear(input_dim, output_dim)

        logger.info(
            f"Initialized EmbeddingProjector: {input_dim} -> {hidden_dim} -> {output_dim} "
            f"(residual={use_residual}, layer_norm={use_layer_norm}, dropout={dropout})"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input embeddings [batch_size, input_dim]

        Returns:
            Projected embeddings [batch_size, output_dim], L2-normalized
        """
        # First layer
        out = self.fc1(x)
        if self.use_layer_norm:
            out = self.ln1(out)
        out = F.gelu(out)
        out = self.dropout1(out)

        # Second layer
        out = self.fc2(out)
        if self.use_layer_norm:
            out = self.ln2(out)
        out = self.dropout2(out)

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
