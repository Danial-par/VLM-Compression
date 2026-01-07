from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from vlmcompress.eval.tasks.utils import find_first_existing


@dataclass
class TextVQAExample:
    qid: int
    image_id: str
    question: str
    answers: List[str]
    image_path: str


def load_textvqa(data_dir: str, limit: Optional[int] = None) -> List[TextVQAExample]:
    d = Path(data_dir)

    ann_candidates = [
        d / "TextVQA_0.5.1_val.json",
        d / "textvqa_val.json",
        d / "annotations" / "TextVQA_0.5.1_val.json",
    ]
    ann_path = find_first_existing(ann_candidates)
    if ann_path is None:
        raise FileNotFoundError(
            "Could not find TextVQA val annotations JSON. Expected TextVQA_0.5.1_val.json in data_dir."
        )

    img_dirs = [
        d / "train_val_images",
        d / "images",
        d / "val_images",
    ]
    img_root = None
    for p in img_dirs:
        if p.exists():
            img_root = p
            break
    if img_root is None:
        raise FileNotFoundError("Could not find TextVQA image directory under data_dir (train_val_images/ or images/).")

    with open(ann_path, "r", encoding="utf-8") as f:
        blob = json.load(f)

    data = blob.get("data", blob if isinstance(blob, list) else [])
    examples: List[TextVQAExample] = []
    for item in data:
        qid = int(item.get("question_id", item.get("qid", 0)))
        image_id = str(item.get("image_id"))
        q = str(item.get("question"))
        answers = item.get("answers", [])
        if isinstance(answers, str):
            answers = [answers]

        # image_id may include extension already
        if image_id.lower().endswith((".jpg", ".jpeg", ".png")):
            image_name = image_id
        else:
            image_name = f"{image_id}.jpg"
        image_path = str(img_root / image_name)

        examples.append(
            TextVQAExample(
                qid=qid,
                image_id=image_id,
                question=q,
                answers=list(answers),
                image_path=image_path,
            )
        )
        if limit is not None and len(examples) >= limit:
            break

    return examples


def make_prompt(ex: TextVQAExample) -> str:
    # Encourage short answers; TextVQA is OCR-heavy
    return f"USER: <image>\nAnswer the question with a short phrase.\nQuestion: {ex.question}\nASSISTANT:"
