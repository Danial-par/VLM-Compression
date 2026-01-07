from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml

from vlmcompress.compression.chunk_index import WeightChunkIndex
from vlmcompress.compression.compressor import ManifoldCompressor, ReconTrainConfig, train_weight_reconstruction
from vlmcompress.utils.seed import seed_everything
from vlmcompress.utils.hf import HFLoadConfig, load_llava_hf
from vlmcompress.utils.logging import setup_logging


def _parse_device_map(dm: Any):
    if dm is None:
        return None
    if isinstance(dm, str):
        if dm.lower() == "cpu":
            return None
        if dm.lower() == "auto":
            return "auto"
        return dm
    return dm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    model_cfg = cfg.get("model", {})
    comp_cfg = cfg.get("compress", {})
    train_cfg = cfg.get("train", {})

    out_dir = Path(train_cfg.get("out_dir", "outputs/llava15_7b_recon"))
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(str(out_dir))

    # Save resolved config
    (out_dir / "config_resolved.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    seed_everything(int(train_cfg.get("seed", 42)), deterministic=False)

    # Load model on CPU (only used to access weights)
    load = HFLoadConfig(
        model_id=model_cfg["model_id"],
        revision=model_cfg.get("revision"),
        torch_dtype=str(model_cfg.get("torch_dtype", "float16")),
        device_map=_parse_device_map(model_cfg.get("device_map", "cpu")),
        load_in_4bit=False,
        load_in_8bit=False,
        trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
    )
    logger.info(f"Loading model on CPU: {load.model_id}")
    model, _processor = load_llava_hf(load)

    include = comp_cfg.get("include") or None
    exclude = comp_cfg.get("exclude") or None

    index = WeightChunkIndex.from_model(
        model,
        chunk_size=int(comp_cfg.get("chunk_size", 4096)),
        include=include,
        exclude=exclude,
        dtype=torch.float16,
        pin_memory=bool(comp_cfg.get("pin_memory", True)),
    )
    logger.info(f"Collected {len(index.specs)} linear weights -> {index.num_chunks_total:,} chunks")

    # Build training config
    tcfg = ReconTrainConfig(
        chunk_size=int(comp_cfg.get("chunk_size", 4096)),
        d_alpha=int(train_cfg.get("d_alpha", 64)),
        decoder_hidden=tuple(train_cfg.get("decoder_hidden", [256, 256])),
        decoder_activation=str(train_cfg.get("decoder_activation", "gelu")),
        decoder_layer_norm=bool(train_cfg.get("decoder_layer_norm", False)),
        batch_size=int(train_cfg.get("batch_size", 256)),
        steps=int(train_cfg.get("steps", 5000)),
        lr=float(train_cfg.get("lr", 2e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        code_l2=float(train_cfg.get("code_l2", 0.0)),
        seed=int(train_cfg.get("seed", 42)),
        device=str(train_cfg.get("device", "cuda")),
        dtype=str(train_cfg.get("dtype", "float16")),
        use_amp=bool(train_cfg.get("use_amp", True)),
        grad_clip=float(train_cfg.get("grad_clip", 1.0)),
        log_every=int(train_cfg.get("log_every", 50)),
        save_every=int(train_cfg.get("save_every", 500)),
        out_dir=str(out_dir),
    )

    compressor = ManifoldCompressor(index=index, cfg=tcfg)

    # Train
    final_ckpt = train_weight_reconstruction(compressor, index=index, cfg=tcfg)
    logger.info(f"Final checkpoint at: {final_ckpt}")


if __name__ == "__main__":
    main()
