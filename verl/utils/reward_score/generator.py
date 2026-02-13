import re
from typing import List
import string
import collections
import os
import datetime
import random
import copy

# =========================
# Logging (robust for Ray workers)
# =========================

DEBUG_DIR = "debug_logs"
DEBUG_ENABLE = os.environ.get("REWARD_DEBUG_ENABLE", "1") == "1"
# Optional: if logs are too large, set REWARD_DEBUG_MAX_CHARS=2000 to truncate
DEBUG_MAX_CHARS = int(os.environ.get("REWARD_DEBUG_MAX_CHARS", "0"))  # 0 means no truncation

from typing import Set

THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", flags=re.DOTALL | re.IGNORECASE)
DOC_RE = re.compile(r"Doc\s*\[\s*(\d+)\s*\]", flags=re.IGNORECASE)

def extract_think(text: str) -> str:
    m = THINK_RE.search(text or "")
    return m.group(1).strip() if m else ""

def count_unique_doc_cites_in_think(predict_str: str, max_doc_id: int = 31) -> int:
    """
    Count unique Doc [i] mentions in <think> where 1 <= i <= max_doc_id.
    """
    think = extract_think(predict_str)
    if not think:
        return 0
    ids: Set[int] = set()
    for m in DOC_RE.finditer(think):
        i = int(m.group(1))
        if 1 <= i <= max_doc_id:
            ids.add(i)
    return len(ids)

def cite_reward_peaked(
    predict_str: str,
    doc_cite: int = 2,
    max_doc_id: int = 31,
) -> float:
    """
    Peaked cite reward:
      - best when the number of unique cited docs == doc_cite (default 2)
      - smaller for 0/1/3/4/...
    Shape:
      c==2 -> 1.0
      c==1 or 3 -> 0.5
      c==0 or >=4 -> 0.0
    """
    c = count_unique_doc_cites_in_think(predict_str, max_doc_id=max_doc_id)

    if c == doc_cite:
        return 1.0
    if c == doc_cite - 1 or c == doc_cite + 1:
        return 0.5
    return 0.0


def _get_worker_log_path() -> str:
    # Each worker writes to a separate file to avoid concurrency issues
    worker_id = os.environ.get("RAY_WORKER_ID") or os.environ.get("RAY_RANK") or str(os.getpid())
    return os.path.join(DEBUG_DIR, f"reward_debug_{worker_id}.log")

def debug_log_block(lines):
    if not DEBUG_ENABLE:
        return
    try:
        os.makedirs(DEBUG_DIR, exist_ok=True)
        path = _get_worker_log_path()
        with open(path, "a", buffering=1) as f:
            for line in lines:
                f.write(line + "\n")
    except Exception:
        return

def _maybe_truncate(s: str) -> str:
    if DEBUG_MAX_CHARS and len(s) > DEBUG_MAX_CHARS:
        return s[:DEBUG_MAX_CHARS] + f"\n...[TRUNCATED {len(s)-DEBUG_MAX_CHARS} chars]"
    return s

# =============== Extract answer from <answer>...</answer> ===============

def normalize_answer(s: str) -> str:
    """Lowercase + remove punctuation + remove articles + whitespace normalization (SQuAD official implementation)."""
    if s is None:
        return ""

    def lower(text):
        return text.lower()

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def squad_exact_match(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def squad_f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()

    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return float(pred_tokens == truth_tokens)

    common = (collections.Counter(pred_tokens) & collections.Counter(truth_tokens))
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)


def extract_answer_tag(text: str) -> str:
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.DOTALL)
    if not m:
        return ""
    return m.group(1).strip()


def format_reward(predict_str: str) -> float:
    has_answer = re.search(r"<answer>.*?</answer>", predict_str, flags=re.DOTALL) is not None
    return 1.0 if has_answer else 0.0


def acc_reward(predict_str: str, ground_truth: List[str]) -> float:
    """
    Returns: 0.7*maxF1 + 0.3*maxEM in [0,1]
    """
    answer = extract_answer_tag(predict_str)
    if answer == "":
        return 0.0

    f1_scores = [squad_f1_score(answer, gt) for gt in ground_truth]
    em_scores = [squad_exact_match(answer, gt) for gt in ground_truth]

    max_f1 = max(f1_scores) if f1_scores else 0.0
    max_em = max(em_scores) if em_scores else 0.0
    return 0.7 * max_f1 + 0.3 * max_em


def compute_score(
    predict_str: str,
    ground_truth: str,
    format_score: float = 0.1,
    **kwargs,
):
    original_predict_str = copy.copy(predict_str)
    if "extra_info" not in kwargs:
        raise ValueError("compute_score needs extra_info")
    extra_info = kwargs["extra_info"]
    ground_truth_answers = extra_info.get("answers", []) or []

    # 1) format check
    fmt = format_reward(predict_str)

    # 2) reward
    if fmt < 1.0:
        reward = 0.0
        max_f1 = 0.0
        max_em = 0.0
        ans = extract_answer_tag(predict_str)
    else:
        ans = extract_answer_tag(predict_str)
        f1_scores = [squad_f1_score(ans, gt) for gt in ground_truth_answers] if ans else []
        em_scores = [float(normalize_answer(gt) == normalize_answer(ans)) for gt in ground_truth_answers] if ans else []
        max_f1 = max(f1_scores) if f1_scores else 0.0
        max_em = max(em_scores) if em_scores else 0.0
        reward = float(0.7 * max_f1 + 0.3 * max_em)
        alpha = 0.8
        reward = float(alpha * reward + (1 - alpha) * cite_reward_peaked(predict_str, doc_cite=2, max_doc_id=31))

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    debug_log_block([
        "=" * 100,
        f"[{now}] fmt={fmt} reward={reward:.6f} max_f1={max_f1:.6f} max_em={max_em:.6f}",
        f"[{now}] answer_extracted={_maybe_truncate(ans)}",
        f"[{now}] gold_answers={ground_truth_answers}",
        f"[{now}] predict_str={original_predict_str}",
        "=" * 100,
        "",
    ])

    return reward
