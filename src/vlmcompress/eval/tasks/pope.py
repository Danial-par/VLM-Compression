from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from vlmcompress.eval.tasks.utils import find_first_existing


@dataclass
class POPEExample:
    idx: int
    image_path: str
    question: str
    label: str  # 'yes' or 'no'


def load_pope(data_dir: str, limit: Optional[int] = None, file_path: Optional[str] = None) -> List[POPEExample]:
    d = Path(data_dir)

    if file_path is None:
        candidates = sorted(list(d.glob("*.jsonl")) + list(d.glob("*.json")))
        if not candidates:
            raise FileNotFoundError("Could not find a POPE .jsonl/.json file in data_dir.")
        file_path = str(candidates[0])

    # Find an images directory if present; otherwise assume paths in file are absolute or already correct.
    img_dirs = [
        d / "images",
        d / "val2014",
        d / "imgs",
    ]
    img_root = None
    for p in img_dirs:
        if p.exists():
            img_root = p
            break

    examples: List[POPEExample] = []
    fp = Path(file_path)
    if fp.suffix == ".jsonl":
        lines = fp.read_text(encoding="utf-8").splitlines()
        records = [json.loads(ln) for ln in lines if ln.strip()]
    else:
        blob = json.loads(fp.read_text(encoding="utf-8"))
        records = blob.get("data", blob if isinstance(blob, list) else [])

    for i, r in enumerate(records):
        q = r.get("question") or r.get("text") or r.get("query") or ""
        label = str(r.get("label") or r.get("answer") or "").lower().strip()
        if label not in ("yes", "no"):
            # some files use 0/1
            if str(label) in ("0", "false"):
                label = "no"
            elif str(label) in ("1", "true"):
                label = "yes"
            else:
                label = "unknown"

        img = r.get("image") or r.get("image_path") or ""
        if img_root is not None and img and not str(img).startswith("/"):
            img = str(img_root / img)

        examples.append(POPEExample(idx=i, image_path=str(img), question=str(q), label=label))
        if limit is not None and len(examples) >= limit:
            break

    return examples


def make_prompt(ex: POPEExample) -> str:
    return f"USER: <image>\nAnswer yes or no. {ex.question}\nASSISTANT:"
