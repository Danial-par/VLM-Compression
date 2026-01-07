from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm

from vlmcompress.compression.chunk_index import WeightChunkIndex
from vlmcompress.compression.generator import MLPDecoder, MLPDecoderConfig
from vlmcompress.utils.logging import setup_logging


@dataclass
class ReconTrainConfig:
    # Chunking / parameterization
    chunk_size: int = 4096
    d_alpha: int = 64
    decoder_hidden: Tuple[int, ...] = (256, 256)
    decoder_activation: str = "gelu"
    decoder_layer_norm: bool = False

    # Optimization
    batch_size: int = 256
    steps: int = 5000
    lr: float = 2e-3
    weight_decay: float = 0.0
    code_l2: float = 0.0  # optional regularizer on codes

    # Runtime
    seed: int = 42
    device: str = "cuda"
    dtype: str = "float16"  # float16|bfloat16|float32
    use_amp: bool = True
    grad_clip: float = 1.0

    # Logging / saving
    log_every: int = 50
    save_every: int = 500
    out_dir: str = "outputs/llava15_7b_recon"


def _dtype(name: str) -> torch.dtype:
    name = name.lower()
    if name in ("float16", "fp16", "half"):
        return torch.float16
    if name in ("bfloat16", "bf16"):
        return torch.bfloat16
    if name in ("float32", "fp32"):
        return torch.float32
    raise ValueError(f"Unknown dtype: {name}")


class ManifoldCompressor(nn.Module):
    """Global codebook + shared decoder for weight-chunk reconstruction."""

    def __init__(self, index: WeightChunkIndex, cfg: ReconTrainConfig):
        super().__init__()
        self.index = index
        self.cfg = cfg

        self.codebook = nn.Embedding(index.num_chunks_total, cfg.d_alpha)
        nn.init.normal_(self.codebook.weight, mean=0.0, std=0.02)

        dec_cfg = MLPDecoderConfig(
            d_alpha=cfg.d_alpha,
            hidden=list(cfg.decoder_hidden),
            activation=cfg.decoder_activation,
            layer_norm=cfg.decoder_layer_norm,
        )
        self.decoder = MLPDecoder(chunk_size=index.chunk_size, cfg=dec_cfg)

    @torch.no_grad()
    def compression_stats(self) -> Dict[str, float]:
        """Compute rough parameter counts / compression ratio (storage, not runtime)."""
        # Original params: sum of all chunked tensors
        orig = sum(s.numel for s in self.index.specs)

        # Compressed storage: codebook + decoder parameters
        comp = sum(p.numel() for p in self.parameters())

        return {
            "original_numel": float(orig),
            "compressed_numel": float(comp),
            "compression_ratio": float(orig) / float(comp) if comp > 0 else float("inf"),
        }

    def reconstruct_chunks(self, chunk_ids: torch.Tensor) -> torch.Tensor:
        """Decode latent codes for a batch of chunk IDs -> [B, chunk_size]."""
        alpha = self.codebook(chunk_ids)
        return self.decoder(alpha)


def train_weight_reconstruction(
    compressor: ManifoldCompressor,
    index: WeightChunkIndex,
    cfg: ReconTrainConfig,
    save_hook=None,
) -> str:
    """Train codebook+decoder to reconstruct pretrained weights (data-free).

    Returns: path to the final checkpoint directory.
    """
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(str(out_dir))

    device = torch.device(cfg.device)
    compute_dtype = _dtype(cfg.dtype)

    # Use AMP scaler only for CUDA + fp16 compute
    amp_enabled = bool(cfg.use_amp and device.type == "cuda" and compute_dtype == torch.float16)

    # IMPORTANT: keep params in fp32 when using GradScaler, otherwise unscale_ fails
    param_dtype = torch.float32 if amp_enabled else compute_dtype

    compressor.to(device=device, dtype=param_dtype)
    compressor.train()

    opt = torch.optim.AdamW(compressor.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # New AMP API (works on recent PyTorch); safe on older too if torch.amp exists
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled) if device.type == "cuda" else torch.amp.GradScaler(enabled=False)

    # Save metadata
    meta_path = out_dir / "chunk_index.json"
    meta_path.write_text(json.dumps(index.json_metadata(), indent=2), encoding="utf-8")
    logger.info(f"Wrote chunk index metadata to: {meta_path}")

    stats = compressor.compression_stats()
    logger.info(f"Compression stats: {stats}")

    global_chunks = index.num_chunks_total
    logger.info(f"Training on {global_chunks:,} total chunks")

    pbar = tqdm(range(1, cfg.steps + 1), desc="train", dynamic_ncols=True)
    running = 0.0
    for step in pbar:
        chunk_ids = torch.randint(0, global_chunks, (cfg.batch_size,), device=device)

        # Targets pulled from CPU (weights are stored on CPU in the index).
        target, mask = index.get_chunks(chunk_ids, device=device, dtype=compute_dtype)

        opt.zero_grad(set_to_none=True)
        # Use autocast only when amp_enabled (CUDA fp16 compute)
        if amp_enabled:
            autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float16)
        else:
            from contextlib import nullcontext
            autocast_ctx = nullcontext()

        with autocast_ctx:
            pred = compressor.reconstruct_chunks(chunk_ids)
            mse = (pred - target) ** 2
            mse = (mse * mask).sum() / (mask.sum() + 1e-8)

            if cfg.code_l2 > 0:
                alpha = compressor.codebook(chunk_ids)
                mse = mse + cfg.code_l2 * (alpha ** 2).mean()

        if amp_enabled:
            scaler.scale(mse).backward()
            if cfg.grad_clip and cfg.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(compressor.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
        else:
            mse.backward()
            if cfg.grad_clip and cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(compressor.parameters(), cfg.grad_clip)
            opt.step()

        running = 0.95 * running + 0.05 * float(mse.item()) if step > 1 else float(mse.item())
        if step % cfg.log_every == 0:
            pbar.set_postfix({"mse": f"{running:.6f}"})

        if step % cfg.save_every == 0 or step == cfg.steps:
            ckpt_dir = out_dir / f"ckpt_step{step:06d}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = ckpt_dir / "compressor.pt"
            torch.save(
                {
                    "cfg": cfg.__dict__,
                    "state_dict": compressor.state_dict(),
                    "stats": stats,
                },
                ckpt_path,
            )
            logger.info(f"Saved checkpoint: {ckpt_path}")
            if save_hook is not None:
                save_hook(str(ckpt_dir), step)

    final_dir = str(out_dir / f"ckpt_step{cfg.steps:06d}")
    logger.info(f"Done. Final checkpoint: {final_dir}")
    return final_dir
