from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
from PIL import Image

from vlmcompress.utils.hf import HFLoadConfig, load_llava_hf


def _parse_device_map(device_map):
    if device_map is None:
        return None
    if isinstance(device_map, str):
        dm = device_map.lower()
        if dm == "cpu":
            return None
        if dm in ("cuda", "gpu", "cuda:0"):
            # Put the whole model on the first GPU
            return {"": 0}
        return device_map
    return device_map


def load_llava(
    model_id: str,
    *,
    revision: Optional[str] = None,
    device_map: Union[str, Dict[str, Any], None] = "auto",
    torch_dtype: str = "float16",
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    trust_remote_code: bool = True,
):
    cfg = HFLoadConfig(
        model_id=model_id,
        revision=revision,
        torch_dtype=torch_dtype,
        device_map=_parse_device_map(device_map),
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        trust_remote_code=trust_remote_code,
    )
    return load_llava_hf(cfg)


@torch.inference_mode()
def generate_one(
    model,
    processor,
    image: Image.Image,
    prompt: str,
    *,
    max_new_tokens: int = 64,
    do_sample: bool = False,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> str:
    """Run a single LLaVA generation."""
    device = next(model.parameters()).device
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    # Some processors output pixel_values in float32; let model handle dtype.
    inputs = {k: v.to(device) for k, v in inputs.items()}

    gen = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None,
    )
    out = processor.batch_decode(gen, skip_special_tokens=True)[0]
    return out
