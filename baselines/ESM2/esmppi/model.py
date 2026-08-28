from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


PAIR_FEATURE_NAMES = ("sum", "absdiff", "product")


def symmetric_pair_features(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Create an order-invariant representation for a protein pair."""
    if first.shape != second.shape:
        raise ValueError(f"Partner embedding shape mismatch: {first.shape} vs {second.shape}")
    return torch.cat((first + second, torch.abs(first - second), first * second), dim=-1)


class RegressionHead(nn.Module):
    """Small regression head trained on frozen pair embeddings."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (512, 128),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

        dimensions = [input_dim, *[int(value) for value in hidden_dims]]
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dimensions[:-1], dimensions[1:]):
            if out_dim <= 0:
                raise ValueError("Every hidden dimension must be positive")
            layers.extend(
                [
                    nn.Linear(in_dim, out_dim),
                    nn.LayerNorm(out_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
        layers.append(nn.Linear(dimensions[-1], 1))
        self.network = nn.Sequential(*layers)

    def forward(self, pair_feature: torch.Tensor) -> torch.Tensor:
        return self.network(pair_feature).squeeze(-1)

