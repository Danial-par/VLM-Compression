from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from vlmcompress.eval.tasks.utils import image_from_base64


_OPTION_KEYS = ["A", "B", "C", "D", "E"]


@dataclass
class MMBenchExample:
    idx: int
    question: str
    options: Dict[str, str]  # letter -> text
    answer: str              # correct letter
    image_b64: str


def load_mmbench(data_dir: str, limit: Optional[int] = None, tsv_path: Optional[str] = None) -> List[MMBenchExample]:
    d = Path(data_dir)
    if tsv_path is None:
        tsvs = sorted(d.glob("*.tsv"))
        if not tsvs:
            raise FileNotFoundError(
                "Could not find a .tsv file for MMBench in data_dir. "

                "Provide --mmbench_tsv explicitly or place the tsv in data_dir."
            )
        tsv_path = str(tsvs[0])

    examples: List[MMBenchExample] = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row_i, row in enumerate(reader):
            q = row.get("question") or row.get("Question") or ""
            ans = (row.get("answer") or row.get("Answer") or "").strip()

            # Options are usually in columns A/B/C/D/E
            opts = {}
            for k in _OPTION_KEYS:
                if row.get(k) is not None and str(row.get(k)).strip() != "":
                    opts[k] = str(row.get(k)).strip()

            img_b64 = row.get("image") or row.get("Image") or row.get("image_base64") or ""
            if not img_b64:
                # Some files might store path instead; leave empty and let script handle.
                img_b64 = ""

            examples.append(
                MMBenchExample(
                    idx=int(row.get("index", row_i)),
                    question=str(q),
                    options=opts,
                    answer=ans,
                    image_b64=img_b64,
                )
            )
            if limit is not None and len(examples) >= limit:
                break

    return examples


def decode_image(ex: MMBenchExample):
    if not ex.image_b64:
        raise ValueError("MMBenchExample has empty base64 image; provide a TSV with base64 images.")
    return image_from_base64(ex.image_b64)


def make_prompt(ex: MMBenchExample) -> str:
    opt_lines = []
    for k in _OPTION_KEYS:
        if k in ex.options:
            opt_lines.append(f"{k}. {ex.options[k]}")
    opt_block = "\n".join(opt_lines)

    return (
        "USER: <image>\n"
        f"{ex.question}\n"
        "Options:\n"
        f"{opt_block}\n"
        "Answer with the option letter only.\n"
        "ASSISTANT:"
    )
