from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from vlmcompress.utils.module import iter_named_linears


@dataclass
class ChunkedParamSpec:
    """Metadata for a single weight tensor that is chunked into 1D segments."""

    module_name: str               # e.g., 'language_model.model.layers.0.self_attn.q_proj'
    param_name: str                # typically 'weight'
    shape: Tuple[int, ...]         # original tensor shape
    numel: int                     # total elements
    chunk_size: int
    start_chunk: int               # global chunk start index for this param
    num_chunks: int                # number of chunks for this param

    # Storage for reconstruction loss:
    # We keep a contiguous flattened CPU tensor for slicing.
    flat_cpu: torch.Tensor         # shape [numel] on CPU, no grad

    def to_jsonable(self) -> Dict:
        d = asdict(self)
        d.pop("flat_cpu", None)
        return d


class WeightChunkIndex:
    """Index over many weight tensors, providing global chunk IDs.

    Each weight tensor is flattened and split into contiguous 1D chunks of length `chunk_size`.
    The last chunk is padded with zeros; a mask indicates valid elements.
    """

    def __init__(self, specs: List[ChunkedParamSpec], chunk_size: int):
        self.specs = specs
        self.chunk_size = int(chunk_size)

        # Global arrays: for each global chunk id -> (spec_id, start_offset, valid_len)
        total_chunks = sum(s.num_chunks for s in specs)
        self.num_chunks_total = total_chunks

        chunk_to_spec = torch.empty((total_chunks,), dtype=torch.int32)
        chunk_start = torch.empty((total_chunks,), dtype=torch.int64)
        chunk_len = torch.empty((total_chunks,), dtype=torch.int32)

        g = 0
        for spec_id, s in enumerate(specs):
            for j in range(s.num_chunks):
                start = j * self.chunk_size
                end = min(start + self.chunk_size, s.numel)
                ln = end - start
                chunk_to_spec[g] = spec_id
                chunk_start[g] = start
                chunk_len[g] = ln
                g += 1

        self.chunk_to_spec = chunk_to_spec
        self.chunk_start = chunk_start
        self.chunk_len = chunk_len

    @staticmethod
    def from_model(
        model: nn.Module,
        chunk_size: int,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        param_name: str = "weight",
        dtype: torch.dtype = torch.float16,
        pin_memory: bool = True,
    ) -> "WeightChunkIndex":
        """Collect `nn.Linear` weights from a model and build a chunk index."""
        specs: List[ChunkedParamSpec] = []
        start_chunk = 0

        for module_name, lin in iter_named_linears(model, include=include, exclude=exclude):
            w = getattr(lin, param_name, None)
            if w is None:
                continue
            w_detached = w.detach().to("cpu", dtype=dtype).contiguous()
            flat = w_detached.view(-1)
            if pin_memory:
                try:
                    flat = flat.pin_memory()
                except Exception:
                    pass

            numel = flat.numel()
            n_chunks = int(math.ceil(numel / chunk_size))
            spec = ChunkedParamSpec(
                module_name=module_name,
                param_name=param_name,
                shape=tuple(w.shape),
                numel=numel,
                chunk_size=int(chunk_size),
                start_chunk=start_chunk,
                num_chunks=n_chunks,
                flat_cpu=flat,
            )
            specs.append(spec)
            start_chunk += n_chunks

        return WeightChunkIndex(specs=specs, chunk_size=chunk_size)

    def json_metadata(self) -> Dict:
        return {
            "chunk_size": self.chunk_size,
            "num_chunks_total": self.num_chunks_total,
            "specs": [s.to_jsonable() for s in self.specs],
        }

    def get_chunks_cpu(self, global_chunk_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (targets, mask) on CPU for a batch of global chunk IDs.

        targets: [B, chunk_size] float16
        mask:    [B, chunk_size] float16 (1 for valid elements, 0 for padding)
        """
        ids = global_chunk_ids.detach().to("cpu", non_blocking=False).long()
        bsz = ids.numel()

        targets = torch.zeros((bsz, self.chunk_size), dtype=torch.float16)
        mask = torch.zeros((bsz, self.chunk_size), dtype=torch.float16)

        spec_ids = self.chunk_to_spec[ids].long()
        starts = self.chunk_start[ids].long()
        lens = self.chunk_len[ids].long()

        for i in range(bsz):
            sid = int(spec_ids[i].item())
            st = int(starts[i].item())
            ln = int(lens[i].item())
            flat = self.specs[sid].flat_cpu
            targets[i, :ln] = flat[st : st + ln]
            mask[i, :ln] = 1.0

        return targets, mask

    def get_chunks(self, global_chunk_ids: torch.Tensor, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (targets, mask) on `device` with dtype `dtype`."""
        t_cpu, m_cpu = self.get_chunks_cpu(global_chunk_ids)
        return t_cpu.to(device=device, dtype=dtype, non_blocking=True), m_cpu.to(device=device, dtype=dtype, non_blocking=True)

    def spec_by_module(self) -> Dict[str, ChunkedParamSpec]:
        return {s.module_name: s for s in self.specs}
