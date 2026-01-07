from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import torch
import torch.nn as nn


@dataclass
class MLPDecoderConfig:
    d_alpha: int = 64
    hidden: List[int] = None  # e.g., [256, 256]
    activation: str = "gelu"  # gelu | relu | silu
    layer_norm: bool = False


def _act(name: str) -> nn.Module:
    name = name.lower()
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    if name in ("silu", "swish"):
        return nn.SiLU()
    raise ValueError(f"Unknown activation: {name}")


class MLPDecoder(nn.Module):
    """Shared decoder: latent code -> flattened weight chunk."""

    def __init__(self, chunk_size: int, cfg: MLPDecoderConfig):
        super().__init__()
        hidden = cfg.hidden or [256, 256]

        layers: List[nn.Module] = []
        in_dim = cfg.d_alpha
        for h in hidden:
            layers.append(nn.Linear(in_dim, h))
            if cfg.layer_norm:
                layers.append(nn.LayerNorm(h))
            layers.append(_act(cfg.activation))
            in_dim = h
        layers.append(nn.Linear(in_dim, chunk_size))
        self.net = nn.Sequential(*layers)

    def forward(self, alpha: torch.Tensor) -> torch.Tensor:
        return self.net(alpha)
