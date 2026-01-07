from __future__ import annotations

import argparse
from pathlib import Path
import torch
import torch.nn as nn

from vlmcompress.models.llava import load_llava
from vlmcompress.compression.state import load_compression_checkpoint
from vlmcompress.compression.modules import patch_model_with_manifold_linears
from vlmcompress.utils.module import get_module, set_module


@torch.no_grad()
def compile_manifold_to_dense(
    model: nn.Module,
    compressor,
    *,
    device: str = "cpu",
    out_dtype: str = "float16",
    max_decode_batch: int = 256,
    verbose_every: int = 20,
):
    """
    Replace ManifoldLinear modules in `model` with dense nn.Linear weights.
    Operates on `device` (CPU recommended for Kaggle).
    """
    dev = torch.device(device)

    # Make sure compressor is on the same device for decoding
    compressor.to(dev)
    compressor.eval()

    # Patch model with ManifoldLinear using the compressor
    patch_model_with_manifold_linears(
        model,
        compressor,
        cache_weight=False,               # we will materialize once and replace permanently
        max_decode_batch=max_decode_batch,
        strict=True,
    )

    # Decide output dtype for saved weights
    out_t = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }[out_dtype.lower()]

    # Move model to CPU for replacement (safer on Kaggle)
    model.to(dev)

    specs = compressor.index.specs
    total = len(specs)
    print(f"[compile] Replacing {total} linear modules with dense nn.Linear ...")

    for i, spec in enumerate(specs, start=1):
        mod = get_module(model, spec.module_name)

        # Our patch_model replaced these with ManifoldLinear
        # Materialize weight on CPU and build a fresh nn.Linear
        w = mod.materialize_weight().to(dtype=out_t, device="cpu")
        b = None
        if mod.bias is not None:
            b = mod.bias.detach().to(dtype=out_t, device="cpu")

        dense = nn.Linear(mod.in_features, mod.out_features, bias=(b is not None))
        dense.weight.data.copy_(w)
        if b is not None:
            dense.bias.data.copy_(b)

        set_module(model, spec.module_name, dense)

        if verbose_every and (i % verbose_every == 0 or i == total):
            print(f"[compile] {i:4d}/{total} done: {spec.module_name}")

    # Remove the compressor attachment if present (optional, keeps saves clean)
    if hasattr(model, "manifold_compressor"):
        delattr(model, "manifold_compressor")

    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", type=str, default="llava-hf/llava-1.5-7b-hf")
    ap.add_argument("--ckpt", type=str, required=True, help="Path to ckpt_stepXXXXXX directory")
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory for compiled HF model")
    ap.add_argument("--device", type=str, default="cpu", help="cpu recommended on Kaggle")
    ap.add_argument("--out_dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--max_decode_batch", type=int, default=256)
    ap.add_argument("--revision", type=str, default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load base model on CPU (don’t quantize here; we are writing dense weights)
    model, processor = load_llava(
        args.model_id,
        revision=args.revision,
        device_map="cpu",
        torch_dtype="float16",
        load_in_4bit=False,
        load_in_8bit=False,
        trust_remote_code=True,
    )

    # Load compressor ckpt on CPU
    _cfg, compressor, _meta = load_compression_checkpoint(args.ckpt, map_location="cpu")

    # Compile
    model = compile_manifold_to_dense(
        model,
        compressor,
        device=args.device,
        out_dtype=args.out_dtype,
        max_decode_batch=args.max_decode_batch,
    )

    # Save as a normal HF model folder (fast to load later)
    print(f"[compile] Saving compiled model to: {out_dir}")
    # Save in a Transformers-compatible way (force shards to avoid huge single file issues)
    model.save_pretrained(
        out_dir,
        safe_serialization=True,
        max_shard_size="2GB",   # ensures model-00001-of-000XX.safetensors + index json
    )

    # Save processor assets (still fine if fallback is used)
    processor.save_pretrained(out_dir)

    # Verify weights exist
    files = {p.name for p in Path(out_dir).glob("*")}
    ok = any(
        name in files for name in [
            "model.safetensors",
            "pytorch_model.bin",
            "model.safetensors.index.json"
        ]
    ) or any(name.startswith("model-00001-of-") and name.endswith(".safetensors") for name in files)

    if not ok:
        raise RuntimeError(
            f"[compile] Save seems to have failed: no weight files found in {out_dir}. "
            f"Files present: {sorted(list(files))[:50]}"
        )


    print("[compile] Done.")


if __name__ == "__main__":
    main()
