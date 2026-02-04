from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from tqdm import tqdm

from vlmcompress.models.llava import load_llava, generate_one
from vlmcompress.compression.state import load_compression_checkpoint
from vlmcompress.compression.modules import patch_model_with_manifold_linears
from vlmcompress.utils.module import get_module, set_module

from vlmcompress.eval.tasks.vqav2 import load_vqav2, make_prompt as vqav2_prompt
from vlmcompress.eval.tasks.utils import load_image
from vlmcompress.eval.vqa_metrics import vqa_soft_accuracy


def _torch_dtype(name: str) -> torch.dtype:
    n = name.lower()
    if n in ("float16", "fp16", "half"):
        return torch.float16
    if n in ("bfloat16", "bf16"):
        return torch.bfloat16
    if n in ("float32", "fp32"):
        return torch.float32
    raise ValueError(f"Unknown dtype: {name}")


@torch.no_grad()
def compile_inplace_to_dense(
    model: nn.Module,
    compressor: nn.Module,
    *,
    device: torch.device,
    out_dtype: torch.dtype,
    max_decode_batch: int,
    verbose_every: int = 20,
) -> nn.Module:
    """
    Patches model with ManifoldLinear, materializes each weight once (CPU),
    then replaces with dense nn.Linear. Does NOT save to disk.
    """
    model.to(device)
    compressor.to(device)
    compressor.eval()

    patch_model_with_manifold_linears(
        model,
        compressor,
        cache_weight=False,
        max_decode_batch=max_decode_batch,
        strict=True,
    )

    specs = compressor.index.specs
    total = len(specs)
    print(f"[compile] Compiling {total} linear layers to dense on {device.type}...")

    for i, spec in enumerate(specs, start=1):
        mod = get_module(model, spec.module_name)

        # materialize dense weight
        w = mod.materialize_weight().to(device="cpu", dtype=out_dtype)
        b = mod.bias.detach().to(device="cpu", dtype=out_dtype) if mod.bias is not None else None

        dense = nn.Linear(mod.in_features, mod.out_features, bias=(b is not None))
        dense.weight.data.copy_(w)
        if b is not None:
            dense.bias.data.copy_(b)

        set_module(model, spec.module_name, dense)

        if verbose_every and (i % verbose_every == 0 or i == total):
            print(f"[compile] {i:4d}/{total} layers compiled")

    # remove compressor reference to keep model clean
    if hasattr(model, "manifold_compressor"):
        delattr(model, "manifold_compressor")

    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", type=str, default="llava-hf/llava-1.5-7b-hf")
    ap.add_argument("--ckpt", type=str, required=True, help="ckpt_stepXXXXXX directory")
    ap.add_argument("--data_dir", type=str, required=True, help="VQAv2 Kaggle dir")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--max_new_tokens", type=int, default=16)

    ap.add_argument("--compile_dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--max_decode_batch", type=int, default=128)
    ap.add_argument("--compile_device", type=str, default="cpu", choices=["cpu"], help="keep compilation on CPU")
    ap.add_argument("--run_device", type=str, default="cuda", choices=["cuda", "cpu"])

    args = ap.parse_args()

    compile_dtype = _torch_dtype(args.compile_dtype)
    compile_device = torch.device(args.compile_device)
    run_device = torch.device(args.run_device)

    # Load base model+processor on CPU
    print("[load] Loading base model on CPU...")
    model, processor = load_llava(
        args.model_id,
        device_map="cpu",
        torch_dtype="float16",
        load_in_4bit=False,
        load_in_8bit=False,
        trust_remote_code=True,
    )

    # Load compressor ckpt on CPU
    print("[load] Loading compressor checkpoint...")
    _cfg, compressor, _meta = load_compression_checkpoint(args.ckpt, map_location="cpu")

    # Compile to dense in CPU RAM (no saving)
    model = compile_inplace_to_dense(
        model,
        compressor,
        device=compile_device,
        out_dtype=compile_dtype,
        max_decode_batch=args.max_decode_batch,
        verbose_every=20,
    )

    # Move to run device (GPU)
    print(f"[run] Moving compiled model to {run_device} ...")
    model.to(run_device)
    model.eval()

    # Load data
    print("[data] Loading VQAv2...")
    data = load_vqav2(args.data_dir, limit=args.limit)

    # Eval
    scores: List[float] = []
    for ex in tqdm(data, desc="eval-vqav2"):
        img = load_image(ex.image_path)
        prompt = vqav2_prompt(ex)
        decoded = generate_one(
            model,
            processor,
            img,
            prompt,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
        pred = decoded.strip()
        scores.append(vqa_soft_accuracy(pred, ex.answers))

    mean_score = float(sum(scores) / max(1, len(scores)))
    print(f"vqav2 mean score over {len(scores)} samples: {mean_score:.4f}")


if __name__ == "__main__":
    main()

