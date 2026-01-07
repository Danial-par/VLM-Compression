from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from vlmcompress.utils.module import get_module, set_module
from vlmcompress.compression.chunk_index import ChunkedParamSpec
from vlmcompress.compression.compressor import ManifoldCompressor


@dataclass
class ManifoldLinearConfig:
    cache_weight: bool = False           # cache full weight after first decode
    max_decode_batch: int = 2048         # decode chunks in micro-batches to limit memory


class ManifoldLinear(nn.Module):
    """An `nn.Linear` whose weight is generated from a manifold parameterization.

    Weight is stored implicitly as:
      - global codebook: alpha[chunk_id] in R^{d_alpha}
      - shared decoder: psi(alpha) -> chunk in R^{chunk_size}

    The module references the *shared* codebook and decoder (not owned by this module).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: Optional[torch.Tensor],
        *,
        weight_shape,
        weight_numel: int,
        chunk_size: int,
        start_chunk: int,
        num_chunks: int,
        codebook: nn.Embedding,
        decoder: nn.Module,
        cfg: ManifoldLinearConfig,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)

        # Keep bias as a trainable parameter (or None). Bias is *not* compressed in this prototype.
        if bias is None:
            self.bias = None
        else:
            self.bias = nn.Parameter(bias.detach().clone())

        self.weight_shape = tuple(weight_shape)
        self.weight_numel = int(weight_numel)
        self.chunk_size = int(chunk_size)
        self.start_chunk = int(start_chunk)
        self.num_chunks = int(num_chunks)

        self.cfg = cfg

        # Store references without registering them as submodules of this layer.
        object.__setattr__(self, "_codebook", codebook)
        object.__setattr__(self, "_decoder", decoder)

        self.register_buffer("_weight_cache", None, persistent=False)

    @property
    def codebook(self) -> nn.Embedding:
        return getattr(self, "_codebook")

    @property
    def decoder(self) -> nn.Module:
        return getattr(self, "_decoder")

    def materialize_weight(self) -> torch.Tensor:
        """Decode and assemble the full weight matrix."""
        # alpha: [num_chunks, d_alpha]
        alpha = self.codebook.weight[self.start_chunk : self.start_chunk + self.num_chunks]
        # Decode in micro-batches to limit peak memory
        outs = []
        bs = int(self.cfg.max_decode_batch)
        for i in range(0, self.num_chunks, bs):
            outs.append(self.decoder(alpha[i : i + bs]))
        chunks = torch.cat(outs, dim=0)  # [num_chunks, chunk_size]
        flat = chunks.reshape(-1)[: self.weight_numel]
        w = flat.view(*self.weight_shape)
        return w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.cfg.cache_weight and self._weight_cache is not None:
            w = self._weight_cache
        else:
            w = self.materialize_weight()
            if self.cfg.cache_weight:
                self._weight_cache = w

        # Ensure dtype/device match the activation dtype/device
        if w.dtype != x.dtype:
            w = w.to(dtype=x.dtype)
        if w.device != x.device:
            w = w.to(device=x.device)

        b = self.bias
        if b is not None:
            if b.device != x.device:
                b = b.to(device=x.device)
            if b.dtype != x.dtype:
                b = b.to(dtype=x.dtype)

        return F.linear(x, w, b)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"chunks={self.num_chunks}, chunk_size={self.chunk_size}, cache={self.cfg.cache_weight}"
        )


def patch_model_with_manifold_linears(
    model: nn.Module,
    compressor: ManifoldCompressor,
    *,
    cache_weight: bool = False,
    max_decode_batch: int = 2048,
    strict: bool = True,
) -> nn.Module:
    """Replace selected `nn.Linear` modules in-place using compressor specs.

    Adds `model.manifold_compressor` so `state_dict` captures the codebook+decoder.
    """
    # Register compressor once on the root model (so it appears in state_dict).
    model.manifold_compressor = compressor

    # Patch modules
    cfg = ManifoldLinearConfig(cache_weight=cache_weight, max_decode_batch=max_decode_batch)

    missing = []
    patched = 0
    for spec in compressor.index.specs:
        try:
            lin = get_module(model, spec.module_name)
        except Exception:
            missing.append(spec.module_name)
            continue

        if not isinstance(lin, nn.Linear):
            if strict:
                raise TypeError(f"Expected nn.Linear at {spec.module_name}, got {type(lin)}")
            else:
                continue

        new_lin = ManifoldLinear(
            in_features=lin.in_features,
            out_features=lin.out_features,
            bias=lin.bias,
            weight_shape=spec.shape,
            weight_numel=spec.numel,
            chunk_size=spec.chunk_size,
            start_chunk=spec.start_chunk,
            num_chunks=spec.num_chunks,
            codebook=model.manifold_compressor.codebook,
            decoder=model.manifold_compressor.decoder,
            cfg=cfg,
        )
        set_module(model, spec.module_name, new_lin)
        patched += 1

    if strict and missing:
        raise KeyError(f"{len(missing)} modules from ckpt were not found in the model. Example: {missing[0]}")
    return model
