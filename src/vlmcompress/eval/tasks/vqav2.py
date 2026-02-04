from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

from PIL import Image

from src.vlmcompress.eval.tasks.utils import load_image, find_first_existing


@dataclass
class VQAv2Example:
    qid: int
    image_id: int
    question: str
    answers: List[str]
    image_path: str


def _default_paths(data_dir: str):
    d = Path(data_dir)

    # ---- annotations ----
    ann = [
        # canonical VQAv2 file name
        d / "v2_mscoco_val2014_annotations.json",
        d / "annotations" / "v2_mscoco_val2014_annotations.json",

        # Kaggle common layout
        d / "v2_Annotations_Val_mscoco" / "v2_mscoco_val2014_annotations.json",
        d / "v2_Annotations_Val_mscoco" / "mscoco_val2014_annotations.json",
    ]

    # ---- questions ----
    qs = [
        # canonical VQAv2 file name
        d / "v2_OpenEnded_mscoco_val2014_questions.json",
        d / "questions" / "v2_OpenEnded_mscoco_val2014_questions.json",

        # Kaggle common layout
        d / "v2_Questions_Val_mscoco" / "v2_OpenEnded_mscoco_val2014_questions.json",
        d / "v2_Questions_Val_mscoco" / "OpenEnded_mscoco_val2014_questions.json",
    ]

    # ---- images ----
    img_dirs = [
        d / "val2014" / "val2014",   # Kaggle first
        d / "val2014",
        d / "images" / "val2014",
        d / "COCO_val2014",
    ]

    return ann, qs, img_dirs


def load_vqav2(data_dir: str, limit: Optional[int] = None) -> List[VQAv2Example]:
    ann_candidates, q_candidates, img_dirs = _default_paths(data_dir)
    ann_path = find_first_existing(ann_candidates)
    q_path = find_first_existing(q_candidates)
    if ann_path is None or q_path is None:
        raise FileNotFoundError(
            "Could not find VQAv2 annotation/question JSON files. "

            "Expected e.g. v2_mscoco_val2014_annotations.json and v2_OpenEnded_mscoco_val2014_questions.json in data_dir."
        )

    with open(ann_path, "r", encoding="utf-8") as f:
        ann = json.load(f)["annotations"]
    with open(q_path, "r", encoding="utf-8") as f:
        qs = json.load(f)["questions"]

    q_by_id: Dict[int, Dict] = {int(q["question_id"]): q for q in qs}

    # find an existing image root
    img_root = None
    for p in img_dirs:
        if p.exists():
            img_root = p
            break
    if img_root is None:
        raise FileNotFoundError(
            "Could not find VQAv2 image directory. Expected val2014/ or images/val2014/ under data_dir."
        )

    examples: List[VQAv2Example] = []
    for a in ann:
        qid = int(a["question_id"])
        q = q_by_id[qid]["question"]
        image_id = int(a["image_id"])
        answers = [x["answer"] for x in a.get("answers", [])]
        image_name = f"COCO_val2014_{image_id:012d}.jpg"
        image_path = str(img_root / image_name)

        examples.append(
            VQAv2Example(
                qid=qid,
                image_id=image_id,
                question=q,
                answers=answers,
                image_path=image_path,
            )
        )
        if limit is not None and len(examples) >= limit:
            break

    return examples


def make_prompt(ex: VQAv2Example) -> str:
    return f"{ex.question}\nAnswer with a short phrase."

