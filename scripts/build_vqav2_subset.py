#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory for VQAv2-mini")
    ap.add_argument("--split", type=str, default="validation", choices=["validation", "testdev", "test"])
    ap.add_argument("--num_samples", type=int, default=800, help="How many Q/A pairs to export")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--buffer_size", type=int, default=10_000, help="Streaming shuffle buffer")
    ap.add_argument("--jpg_quality", type=int, default=95)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    img_dir = out_dir / "val2014"
    img_dir.mkdir(parents=True, exist_ok=True)

    # Stream dataset to avoid full download
    ds = load_dataset("lmms-lab/VQAv2", split=args.split, streaming=True)

    # Shuffle in streaming mode (approximate shuffle using a buffer)
    ds = ds.shuffle(seed=args.seed, buffer_size=args.buffer_size)

    questions = {"questions": []}
    annotations = {"annotations": []}

    n = 0
    for ex in tqdm(ds, total=args.num_samples, desc=f"Exporting {args.num_samples} samples"):
        if n >= args.num_samples:
            break

        # Required by your loader:
        qid = int(ex["question_id"])
        image_id = int(ex["image_id"])
        question = ex["question"]

        # HF has answers as list[dict] with "answer" keys (10 answers typically)
        # Your evaluator expects list[str]
        ans_list = ex.get("answers", [])
        if isinstance(ans_list, list) and len(ans_list) > 0 and isinstance(ans_list[0], dict):
            answers = [a.get("answer", "") for a in ans_list]
        elif isinstance(ans_list, list):
            answers = [str(a) for a in ans_list]
        else:
            answers = []

        # Fallback: if for any reason answers are missing, use multiple_choice_answer if present
        if len([a for a in answers if a]) == 0:
            mca = ex.get("multiple_choice_answer", "")
            answers = [mca] if mca else [""]

        # Ensure at least 10 answers like classic VQAv2 (not strictly required, but nice)
        answers = [a if a is not None else "" for a in answers]
        if len(answers) < 10:
            answers = (answers * (10 // max(1, len(answers)) + 1))[:10]
        else:
            answers = answers[:10]

        # Save image as COCO_val2014_XXXXXXXXXXXX.jpg
        image_name = f"COCO_val2014_{image_id:012d}.jpg"
        image_path = img_dir / image_name

        # HF "image" is usually a PIL image object in streaming mode
        img = ex["image"]
        if not image_path.exists():
            img.convert("RGB").save(image_path, format="JPEG", quality=args.jpg_quality, optimize=True)

        questions["questions"].append(
            {
                "question_id": qid,
                "image_id": image_id,
                "question": question,
            }
        )

        annotations["annotations"].append(
            {
                "question_id": qid,
                "image_id": image_id,
                "answers": [{"answer": a} for a in answers],
            }
        )

        n += 1

    # Write files with EXACT names your loader searches for
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "v2_OpenEnded_mscoco_val2014_questions.json").write_text(
        json.dumps(questions), encoding="utf-8"
    )
    (out_dir / "v2_mscoco_val2014_annotations.json").write_text(
        json.dumps(annotations), encoding="utf-8"
    )

    print(f"\nDone. Wrote:")
    print(f"  {out_dir / 'v2_OpenEnded_mscoco_val2014_questions.json'}")
    print(f"  {out_dir / 'v2_mscoco_val2014_annotations.json'}")
    print(f"  images in {img_dir} (count ~{n})")


if __name__ == "__main__":
    main()
