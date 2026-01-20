import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal
import logging

logger = logging.getLogger(__name__)


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
