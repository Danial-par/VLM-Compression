from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

from vlmcompress.compression.compressor import ManifoldCompressor, ReconTrainConfig
from vlmcompress.compression.chunk_index import ChunkedParamSpec


@dataclass
class MinimalChunkIndex:
    """A lightweight stand-in for WeightChunkIndex for *inference-time* patching.

    We intentionally avoid storing the original weight values here.
    """

    chunk_size: int
    num_chunks_total: int
    specs: List[ChunkedParamSpec]


def _find_chunk_index_json(ckpt_dir: Path) -> Path:
    # chunk_index.json is written to cfg.out_dir (parent of ckpt_dir).
    # Allow both layouts:
    #  - out_dir/chunk_index.json
    #  - ckpt_dir/chunk_index.json
    candidates = [
        ckpt_dir / "chunk_index.json",
        ckpt_dir.parent / "chunk_index.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"Could not find chunk_index.json near {ckpt_dir}")


def load_compression_checkpoint(ckpt_dir: str, map_location: str | torch.device = "cpu") -> Tuple[ReconTrainConfig, ManifoldCompressor, Dict[str, Any]]:
    """Load (train_cfg, compressor_module, chunk_index_metadata)."""
    ckpt_dir_p = Path(ckpt_dir)
    ckpt_path = ckpt_dir_p / "compressor.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint file: {ckpt_path}")

    blob = torch.load(ckpt_path, map_location=map_location)
    cfg = ReconTrainConfig(**blob["cfg"])

    meta_path = _find_chunk_index_json(ckpt_dir_p)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    # Build minimal specs list
    specs: List[ChunkedParamSpec] = []
    for s in meta["specs"]:
        specs.append(
            ChunkedParamSpec(
                module_name=s["module_name"],
                param_name=s["param_name"],
                shape=tuple(s["shape"]),
                numel=int(s["numel"]),
                chunk_size=int(s["chunk_size"]),
                start_chunk=int(s["start_chunk"]),
                num_chunks=int(s["num_chunks"]),
                flat_cpu=torch.empty((0,), dtype=torch.float16),  # unused
            )
        )

    index = MinimalChunkIndex(
        chunk_size=int(meta["chunk_size"]),
        num_chunks_total=int(meta["num_chunks_total"]),
        specs=specs,
    )

    compressor = ManifoldCompressor(index=index, cfg=cfg)
    compressor.load_state_dict(blob["state_dict"], strict=True)
    compressor.eval()

    return cfg, compressor, meta
