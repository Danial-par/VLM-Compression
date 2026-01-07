from __future__ import annotations

import re
import string
from collections import Counter
from typing import Iterable, List


_ARTICLES = {"a", "an", "the"}

_PUNCT = set(string.punctuation)


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_answer(ans: str) -> str:
    """VQA-style normalization (lightweight)."""
    if ans is None:
        return ""
    ans = ans.lower().strip()

    # Replace some common punctuation with spaces, then drop remaining punctuation.
    ans = ans.replace("/", " ").replace("-", " ")
    ans = "".join(ch if ch not in _PUNCT else " " for ch in ans)

    words = [w for w in ans.split() if w not in _ARTICLES]
    return _normalize_whitespace(" ".join(words))


def vqa_soft_accuracy(pred: str, gt_answers: Iterable[str]) -> float:
    """Standard VQA soft accuracy: min(#matching/3, 1)."""
    pred_n = normalize_answer(pred)
    gts = [normalize_answer(a) for a in gt_answers]
    match = sum(1 for a in gts if a == pred_n)
    return min(1.0, match / 3.0)


def exact_match(pred: str, gt: str) -> float:
    return 1.0 if normalize_answer(pred) == normalize_answer(gt) else 0.0


def yesno_from_text(text: str) -> str:
    """Extract yes/no from a model output."""
    t = normalize_answer(text)
    if t.startswith("yes"):
        return "yes"
    if t.startswith("no"):
        return "no"
    # fallback: search token-wise
    toks = t.split()
    if "yes" in toks and "no" not in toks:
        return "yes"
    if "no" in toks and "yes" not in toks:
        return "no"
    return "unknown"


def f1_yesno(preds: List[str], labels: List[str]) -> float:
    """Binary F1 for yes/no labels."""
    # treat 'yes' as positive
    tp = sum((p == "yes") and (y == "yes") for p, y in zip(preds, labels))
    fp = sum((p == "yes") and (y == "no") for p, y in zip(preds, labels))
    fn = sum((p == "no") and (y == "yes") for p, y in zip(preds, labels))
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    return 2 * precision * recall / (precision + recall + 1e-8)
