from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

from vlmcompress.models.llava import load_llava, generate_one
from vlmcompress.compression.state import load_compression_checkpoint
from vlmcompress.compression.modules import patch_model_with_manifold_linears
from vlmcompress.eval.vqa_metrics import vqa_soft_accuracy, normalize_answer, yesno_from_text, f1_yesno

from vlmcompress.eval.tasks.vqav2 import load_vqav2, make_prompt as vqav2_prompt
from vlmcompress.eval.tasks.textvqa import load_textvqa, make_prompt as textvqa_prompt
from vlmcompress.eval.tasks.mmbench import load_mmbench, decode_image as mmbench_image, make_prompt as mmbench_prompt
from vlmcompress.eval.tasks.pope import load_pope, make_prompt as pope_prompt
from vlmcompress.eval.tasks.utils import load_image


def extract_assistant_answer(decoded: str) -> str:
    if decoded is None:
        return ""
    # HF decodes often include the prompt prefix; strip it.
    if "ASSISTANT:" in decoded:
        return decoded.split("ASSISTANT:")[-1].strip()
    if "assistant:" in decoded.lower():
        # robust fallback
        parts = decoded.split("assistant:")
        return parts[-1].strip()
    return decoded.strip()


def extract_choice_letter(text: str) -> str:
    t = extract_assistant_answer(text).strip()
    if not t:
        return ""
    # common patterns: "A", "A.", "(A)"
    for ch in t:
        if ch.upper() in ["A", "B", "C", "D", "E"]:
            return ch.upper()
    return t[:1].upper()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=str, required=True, choices=["mmbench", "vqav2", "textvqa", "pope"])
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--model_id", type=str, default="llava-hf/llava-1.5-7b-hf")
    ap.add_argument("--ckpt", type=str, default=None, help="Compression ckpt dir (ckpt_stepXXXXXX)")
    ap.add_argument("--baseline", action="store_true", help="Do NOT apply compression")
    ap.add_argument("--limit", type=int, default=None)

    # runtime
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--load_in_4bit", action="store_true")
    ap.add_argument("--load_in_8bit", action="store_true")
    ap.add_argument("--cache_weight", action="store_true")
    ap.add_argument("--max_decode_batch", type=int, default=2048)

    # generation
    ap.add_argument("--max_new_tokens", type=int, default=32)

    ap.add_argument("--save_preds", type=str, default=None, help="Optional path to save JSONL predictions")
    ap.add_argument("--mmbench_tsv", type=str, default=None, help="Optional explicit MMBench TSV path")

    args = ap.parse_args()

    device = torch.device(args.device)

    # ---- Load data ----
    if args.task == "vqav2":
        data = load_vqav2(args.data_dir, limit=args.limit)
    elif args.task == "textvqa":
        data = load_textvqa(args.data_dir, limit=args.limit)
    elif args.task == "mmbench":
        data = load_mmbench(args.data_dir, limit=args.limit, tsv_path=args.mmbench_tsv)
    elif args.task == "pope":
        data = load_pope(args.data_dir, limit=args.limit)
    else:
        raise ValueError(args.task)

    # ---- Load model ----
    if args.baseline:
        # Baseline: can use 4bit/8bit to fit on 16GB.
        model, processor = load_llava(
            args.model_id,
            device_map="auto" if device.type == "cuda" else "cpu",
            torch_dtype="float16",
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
            trust_remote_code=True,
        )
    else:
        if not args.ckpt:
            raise ValueError("Provide --ckpt for compressed evaluation, or use --baseline")
        # Load on CPU first, patch, then move to GPU.
        model, processor = load_llava(
            args.model_id,
            device_map="cpu",
            torch_dtype="float16",
            load_in_4bit=False,
            load_in_8bit=False,
            trust_remote_code=True,
        )
        _cfg, compressor, _meta = load_compression_checkpoint(args.ckpt, map_location="cpu")
        patch_model_with_manifold_linears(
            model,
            compressor,
            cache_weight=args.cache_weight,
            max_decode_batch=args.max_decode_batch,
            strict=True,
        )
        model.to(device)

    # ---- Eval loop ----
    preds_out = []
    scores = []
    yesno_preds = []
    yesno_labels = []

    for ex in tqdm(data, desc=f"eval-{args.task}"):
        if args.task == "vqav2":
            img = load_image(ex.image_path)
            prompt = vqav2_prompt(ex)
            decoded = generate_one(model, processor, img, prompt, max_new_tokens=args.max_new_tokens)
            pred = extract_assistant_answer(decoded)
            acc = vqa_soft_accuracy(pred, ex.answers)
            scores.append(acc)
            preds_out.append({"qid": ex.qid, "pred": pred, "acc": acc})

        elif args.task == "textvqa":
            img = load_image(ex.image_path)
            prompt = textvqa_prompt(ex)
            decoded = generate_one(model, processor, img, prompt, max_new_tokens=args.max_new_tokens)
            pred = extract_assistant_answer(decoded)
            acc = vqa_soft_accuracy(pred, ex.answers)
            scores.append(acc)
            preds_out.append({"qid": ex.qid, "pred": pred, "acc": acc})

        elif args.task == "mmbench":
            img = mmbench_image(ex)
            prompt = mmbench_prompt(ex)
            decoded = generate_one(model, processor, img, prompt, max_new_tokens=args.max_new_tokens)
            pred_letter = extract_choice_letter(decoded)
            gt = ex.answer.strip().upper()
            acc = 1.0 if pred_letter == gt else 0.0
            scores.append(acc)
            preds_out.append({"idx": ex.idx, "pred": pred_letter, "gt": gt, "acc": acc})

        elif args.task == "pope":
            img = load_image(ex.image_path)
            prompt = pope_prompt(ex)
            decoded = generate_one(model, processor, img, prompt, max_new_tokens=args.max_new_tokens)
            pred = yesno_from_text(decoded)
            yesno_preds.append(pred)
            yesno_labels.append(ex.label)
            acc = 1.0 if pred == ex.label else 0.0
            scores.append(acc)
            preds_out.append({"idx": ex.idx, "pred": pred, "gt": ex.label, "acc": acc})

    mean_score = float(sum(scores) / max(len(scores), 1))
    print(f"{args.task} mean score over {len(scores)} samples: {mean_score:.4f}")

    if args.task == "pope":
        f1 = f1_yesno(yesno_preds, yesno_labels)
        print(f"POPE F1 (yes as positive): {f1:.4f}")

    if args.save_preds:
        outp = Path(args.save_preds)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w", encoding="utf-8") as f:
            for r in preds_out:
                f.write(json.dumps(r) + "\n")
        print(f"Saved predictions to: {outp}")


if __name__ == "__main__":
    main()
