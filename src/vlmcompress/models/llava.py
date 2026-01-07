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
    """Run a single LLaVA generation.

    IMPORTANT:
    Many LLaVA HF checkpoints expect their chat template; using apply_chat_template
    improves correctness vs hand-written "USER: <image>" prompts.
    """
    device = next(model.parameters()).device

    # If processor supports chat template, use it
    if hasattr(processor, "apply_chat_template"):
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    else:
        # fallback to raw prompt
        text = prompt

    inputs = processor(text=text, images=image, return_tensors="pt")
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
