from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch


@dataclass
class HFLoadConfig:
    model_id: str
    revision: Optional[str] = None
    torch_dtype: str = "float16"  # 'float16' | 'bfloat16' | 'float32'
    device_map: str | Dict[str, Any] | None = "auto"

    load_in_4bit: bool = False
    load_in_8bit: bool = False

    # Trust remote code is often needed for VLMs.
    trust_remote_code: bool = True


def _dtype_from_string(name: str) -> torch.dtype:
    name = name.lower()
    if name in ("fp16", "float16", "half"):
        return torch.float16
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unknown dtype string: {name}")


def load_llava_hf(load: HFLoadConfig) -> Tuple[Any, Any]:
    """Load a HuggingFace LLaVA model + processor.

    Works with:
      - llava-hf/llava-1.5-7b-hf
      - other HF repos exposing a LLaVA-compatible processor and model class

    Returns (model, processor).
    """
    from transformers import AutoProcessor

    # Transformers has multiple possible classes depending on version.
    # We try the most general entry point first.
    try:
        from transformers import AutoModelForVision2Seq
        model_cls = AutoModelForVision2Seq
    except Exception:
        # Fallback: many LLaVA checkpoints register as causal LM
        from transformers import AutoModelForCausalLM
        model_cls = AutoModelForCausalLM

    quant_kwargs: Dict[str, Any] = {}
    if load.load_in_4bit or load.load_in_8bit:
        # bitsandbytes optional; if missing, transformers will raise a helpful error.
        quant_kwargs["load_in_4bit"] = load.load_in_4bit
        quant_kwargs["load_in_8bit"] = load.load_in_8bit

    torch_dtype = _dtype_from_string(load.torch_dtype)

    processor = AutoProcessor.from_pretrained(
        load.model_id,
        revision=load.revision,
        trust_remote_code=load.trust_remote_code,
    )

    model = model_cls.from_pretrained(
        load.model_id,
        revision=load.revision,
        torch_dtype=torch_dtype,
        device_map=load.device_map,
        trust_remote_code=load.trust_remote_code,
        **quant_kwargs,
    )
    model.eval()
    return model, processor
