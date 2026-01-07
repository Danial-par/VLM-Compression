from __future__ import annotations

import argparse
import io
import urllib.request
from pathlib import Path
from typing import Optional

from PIL import Image
import torch

from vlmcompress.models.llava import load_llava, generate_one
from vlmcompress.compression.state import load_compression_checkpoint
from vlmcompress.compression.modules import patch_model_with_manifold_linears


def _load_image(image_path: Optional[str], image_url: Optional[str]) -> Image.Image:
    if image_path:
        return Image.open(image_path).convert("RGB")
    if image_url:
        with urllib.request.urlopen(image_url) as resp:
            data = resp.read()
        return Image.open(io.BytesIO(data)).convert("RGB")
    raise ValueError("Provide either --image_path or --image_url")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True, help="Checkpoint dir (ckpt_stepXXXXXX)")
    ap.add_argument("--image_path", type=str, default=None)
    ap.add_argument("--image_url", type=str, default=None)
    ap.add_argument("--prompt", type=str, required=True)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--cache_weight", action="store_true")
    ap.add_argument("--max_decode_batch", type=int, default=2048)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    # Load base model on CPU to avoid GPU OOM before patching
    model, processor = load_llava(
        args.model_id,
        device_map="cpu",
        torch_dtype="float16",
        load_in_4bit=False,
        load_in_8bit=False,
        trust_remote_code=True,
    )

    # Load compressor ckpt
    _cfg, compressor, _meta = load_compression_checkpoint(args.ckpt, map_location="cpu")

    # Patch model weights
    patch_model_with_manifold_linears(
        model,
        compressor,
        cache_weight=args.cache_weight,
        max_decode_batch=args.max_decode_batch,
        strict=True,
    )

    # Move to GPU (compressed model is small)
    device = torch.device(args.device)
    model.to(device)

    img = _load_image(args.image_path, args.image_url)

    out = generate_one(
        model,
        processor,
        img,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
    )
    print(out)


if __name__ == "__main__":
    main()
