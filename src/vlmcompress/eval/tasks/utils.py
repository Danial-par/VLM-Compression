from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Optional

from PIL import Image


def load_image(path: str) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return img


def image_from_base64(b64: str) -> Image.Image:
    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    return img


def find_first_existing(paths):
    for p in paths:
        if p and Path(p).exists():
            return str(p)
    return None
