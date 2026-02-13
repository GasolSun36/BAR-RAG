#!/usr/bin/env python3
"""
Selector Reward Function for TRL GRPO Training

This module implements the reward function for selector training.
The selector outputs in the format: <think>...</think><answer>...</answer>.
An external Generator API is called to produce answers, and rewards are computed based on answer quality.

Key features:
- Selector output format: <think>reasoning</think><answer>[1], [3], [5]</answer>
- Generator is called via an external API (does not use local GPU)
- Supports multiple rollouts to estimate uncertainty reward
- Generator reward = 0.8 * (0.7*F1 + 0.3*EM) + 0.2 * cite_reward
"""

import os
import re
import json
import string
import collections
import asyncio
import aiohttp
import datetime
import copy
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple, Set
from concurrent.futures import ThreadPoolExecutor

# =================== Debug Logging ===================

DEBUG_DIR = os.environ.get("REWARD_DEBUG_DIR", "/tmp/selector_reward_logs")
DEBUG_ENABLE = os.environ.get("REWARD_DEBUG_ENABLE", "1") == "1"
DEBUG_MAX_CHARS = int(os.environ.get("REWARD_DEBUG_MAX_CHARS", "0"))  # 0 means no truncation


def _get_worker_log_path() -> str:
    """Each worker writes to a separate file to avoid concurrency issues."""
    worker_id = os.environ.get("RAY_WORKER_ID") or os.environ.get("RAY_RANK") or str(os.getpid())
    return os.path.join(DEBUG_DIR, f"selector_reward_debug_{worker_id}.log")


def debug_log_block(lines: List[str]):
    """Write a debug log block."""
    if not DEBUG_ENABLE:
        return
    try:
        os.makedirs(DEBUG_DIR, exist_ok=True)
        path = _get_worker_log_path()
        with open(path, "a", buffering=1) as f:
            for line in lines:
                f.write(line + "\n")
    except Exception:
        pass


def _maybe_truncate(s: str, max_chars: int = 0) -> str:
    """Truncate overly long strings."""
    if max_chars <= 0:
        max_chars = DEBUG_MAX_CHARS
    if max_chars and len(s) > max_chars:
        return s[:max_chars] + f"\n...[TRUNCATED {len(s) - max_chars} chars]"
    return s

# =================== Configuration ===================

# Generator API configuration
GENERATOR_API_URL = os.environ.get("GENERATOR_API_URL", "http://localhost:8000/v1/chat/completions")
GENERATOR_MODEL_NAME = os.environ.get("GENERATOR_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
GENERATOR_API_KEY = os.environ.get("GENERATOR_API_KEY", "")

# LLM Judge API configuration (independent of Generator)
LLM_JUDGE_API_URL = os.environ.get("LLM_JUDGE_API_URL", "")
LLM_JUDGE_MODEL_NAME = os.environ.get("LLM_JUDGE_MODEL_NAME", "")
LLM_JUDGE_API_KEY = os.environ.get("LLM_JUDGE_API_KEY", "")

# Rollout configuration
K_ROLLOUTS = int(os.environ.get("K_ROLLOUTS", "8"))
GENERATOR_TEMPERATURE = float(os.environ.get("GENERATOR_TEMPERATURE", "0.7"))
GENERATOR_TOP_P = float(os.environ.get("GENERATOR_TOP_P", "0.95"))
GENERATOR_MAX_TOKENS = int(os.environ.get("GENERATOR_MAX_TOKENS", "512"))

# Generator input token budget
GENERATOR_MAX_INPUT_TOKENS = int(os.environ.get("GENERATOR_MAX_INPUT_TOKENS", "2048"))
GENERATOR_TOKENIZER_NAME_OR_PATH = os.environ.get("GENERATOR_TOKENIZER_NAME_OR_PATH", GENERATOR_MODEL_NAME)

# Generator reward configuration
GENERATOR_REWARD_CFG: Dict[str, Any] = {
    "acc_weight": float(os.environ.get("GENERATOR_ACC_WEIGHT", "0.8")),
    "f1_weight": float(os.environ.get("GENERATOR_F1_WEIGHT", "0.7")),
    "em_weight": float(os.environ.get("GENERATOR_EM_WEIGHT", "0.3")),
    "target_cite_count": int(os.environ.get("GENERATOR_TARGET_CITE", "2")),
    "max_doc_id": int(os.environ.get("GENERATOR_MAX_DOC_ID", "31")),
    "reward_threshold": float(os.environ.get("GENERATOR_REWARD_THRESHOLD", "0.5")),
}

# Selector reward configuration
REWARD_CFG: Dict[str, Any] = {
    "target_center": float(os.environ.get("REWARD_TARGET_CENTER", "0.6")),
    "max_unc_reward": float(os.environ.get("REWARD_MAX_UNC", "1.0")),
    "max_rel_reward": float(os.environ.get("REWARD_MAX_REL", "0.5")),
    "lambda_unc": float(os.environ.get("REWARD_LAMBDA_UNC", "1.0")),
    "lambda_rel": float(os.environ.get("REWARD_LAMBDA_REL", "0.2")),
    "default_k_docs": int(os.environ.get("REWARD_DEFAULT_K_DOCS", "5")),
    "target_num_docs": int(os.environ.get("REWARD_TARGET_NUM_DOCS", "5")),
    "wrong_count_penalty": float(os.environ.get("REWARD_WRONG_COUNT_PENALTY", "0.5")),
    "max_count_penalty": float(os.environ.get("REWARD_MAX_COUNT_PENALTY", "1.0")),
    "format_reward": float(os.environ.get("REWARD_FORMAT_BONUS", "0.1")),
    "format_penalty": float(os.environ.get("REWARD_FORMAT_PENALTY", "0.2")),
}

# =================== Generator Token Truncation ===================

_GENERATOR_TOKENIZER = None


def _lazy_load_tokenizer(name_or_path: str):
    if not name_or_path:
        return None
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=True, use_fast=True)
        return tok
    except Exception:
        return None


def get_generator_tokenizer():
    global _GENERATOR_TOKENIZER
    if _GENERATOR_TOKENIZER is not None:
        return _GENERATOR_TOKENIZER
    _GENERATOR_TOKENIZER = _lazy_load_tokenizer(GENERATOR_TOKENIZER_NAME_OR_PATH)
    return _GENERATOR_TOKENIZER


def count_tokens(text: str, tokenizer) -> int:
    if tokenizer is None:
        return 0
    if not text:
        return 0
    try:
        return len(tokenizer.encode(text, add_special_tokens=False))
    except Exception:
        return 0


def truncate_to_tokens(text: str, tokenizer, max_tokens: int) -> str:
    """Truncate to max_tokens keeping the prefix. If tokenizer fails, return original."""
    if tokenizer is None:
        return text
    if max_tokens <= 0:
        return ""
    try:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) <= max_tokens:
            return text
        ids = ids[:max_tokens]
        return tokenizer.decode(ids, skip_special_tokens=True)
    except Exception:
        return text


def allocate_and_truncate_texts_equal(
    texts: List[str],
    tokenizer,
    total_budget_tokens: int,
) -> List[str]:
    """Split total_budget_tokens evenly across texts and truncate each (prefix kept)."""
    n = len(texts)
    if n == 0:
        return []
    if tokenizer is None:
        return texts
    if total_budget_tokens <= 0:
        return [""] * n

    per = total_budget_tokens // n
    rem = total_budget_tokens - per * n

    out = []
    for i, t in enumerate(texts):
        budget = per + (1 if i < rem else 0)
        out.append(truncate_to_tokens(t, tokenizer, budget))
    return out


# =================== Data Structures ===================

@dataclass
class EvidenceDoc:
    """Candidate document."""
    doc_id: str
    text: str
    title: str
    score: float  # selector_score


@dataclass
class RolloutResult:
    """Single generator rollout result."""
    raw_output: str
    extracted_answer: str
    total_reward: float
    acc_reward: float
    cite_reward: float
    max_f1: float
    max_em: float
    cite_count: int
    has_format: bool
    is_correct: bool


# =================== Text Utilities ===================

def normalize_answer(s: str) -> str:
    """SQuAD-style normalization: lowercase, remove punctuation, remove articles, normalize whitespace."""
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


def squad_f1_score(prediction: str, ground_truth: str) -> float:
    """Compute SQuAD token-level F1."""
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


def squad_exact_match(prediction: str, ground_truth: str) -> float:
    """Compute exact match."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


# =================== Generator Output Parsing and Reward ===================

def extract_answer_tag(text: str) -> str:
    """Extract content inside <answer>...</answer>."""
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.DOTALL)
    if not m:
        return ""
    return m.group(1).strip()


def extract_think_tag(text: str) -> str:
    """Extract content inside <think>...</think>."""
    m = re.search(r"<think>\s*(.*?)\s*</think>", text, flags=re.DOTALL | re.IGNORECASE)
    if not m:
        return ""
    return m.group(1).strip()


def count_unique_doc_cites_in_think(predict_str: str, max_doc_id: int = 31) -> int:
    """
    Count unique Doc [i] mentions in <think> where 1 <= i <= max_doc_id.
    """
    think = extract_think_tag(predict_str)
    if not think:
        return 0

    doc_pattern = re.compile(r"Doc\s*\[\s*(\d+)\s*\]", flags=re.IGNORECASE)
    ids: Set[int] = set()
    for m in doc_pattern.finditer(think):
        i = int(m.group(1))
        if 1 <= i <= max_doc_id:
            ids.add(i)
    return len(ids)


def cite_reward_peaked(
    predict_str: str,
    target_cite_count: int = 2,
    max_doc_id: int = 31,
) -> float:
    """
    Peaked cite reward:
      - best (1.0) when the number of unique cited docs == target_cite_count
      - 0.5 for ±1
      - 0.0 otherwise
    """
    c = count_unique_doc_cites_in_think(predict_str, max_doc_id=max_doc_id)

    if c == target_cite_count:
        return 1.0
    if c == target_cite_count - 1 or c == target_cite_count + 1:
        return 0.5
    return 0.0


def compute_generator_reward(
    predict_str: str,
    golden_answers: List[str],
    max_doc_id: int = 31,
    target_cite_count: int = 2,
    acc_weight: float = 0.8,
    f1_weight: float = 0.7,
    em_weight: float = 0.3,
) -> Dict[str, Any]:
    """Compute generator reward and return a detailed breakdown."""
    result = {
        "total_reward": 0.0,
        "acc_reward": 0.0,
        "cite_reward": 0.0,
        "max_f1": 0.0,
        "max_em": 0.0,
        "extracted_answer": "",
        "cite_count": 0,
        "has_format": False,
    }

    has_answer = re.search(r"<answer>.*?</answer>", predict_str, flags=re.DOTALL) is not None
    result["has_format"] = has_answer

    if not has_answer:
        return result

    answer = extract_answer_tag(predict_str)
    result["extracted_answer"] = answer

    if not answer or not golden_answers:
        return result

    f1_scores = [squad_f1_score(answer, gt) for gt in golden_answers]
    em_scores = [squad_exact_match(answer, gt) for gt in golden_answers]

    max_f1 = max(f1_scores) if f1_scores else 0.0
    max_em = max(em_scores) if em_scores else 0.0

    result["max_f1"] = max_f1
    result["max_em"] = max_em

    acc_reward = f1_weight * max_f1 + em_weight * max_em
    result["acc_reward"] = acc_reward

    cite_count = count_unique_doc_cites_in_think(predict_str, max_doc_id=max_doc_id)
    cite_reward = cite_reward_peaked(predict_str, target_cite_count=target_cite_count, max_doc_id=max_doc_id)
    result["cite_count"] = cite_count
    result["cite_reward"] = cite_reward

    total_reward = acc_weight * acc_reward + (1 - acc_weight) * cite_reward
    result["total_reward"] = total_reward

    return result


# =================== Selector Output Parsing ===================

def extract_selector_answer(text: str) -> str:
    """Extract content inside <answer>...</answer> from selector output."""
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.DOTALL)
    if not m:
        return ""
    return m.group(1).strip()


def extract_selector_think(text: str) -> str:
    """Extract content inside <think>...</think> from selector output."""
    m = re.search(r"<think>\s*(.*?)\s*</think>", text, flags=re.DOTALL)
    if not m:
        return ""
    return m.group(1).strip()


def check_selector_format(text: str) -> bool:
    """Check if selector output contains a valid <answer>...</answer> block."""
    has_answer = re.search(r"<answer>.*?</answer>", text, flags=re.DOTALL) is not None
    return has_answer


def parse_selector_output_to_doc_indices(
    predict_str: str,
    num_candidates: int,
    default_k: int = 5
) -> Tuple[List[int], bool, bool]:
    """
    Parse selector output and return:
      - selected document indices (0-based)
      - format_valid
      - has_duplicate
    """
    has_answer_tag = re.search(r"<answer>.*?</answer>", predict_str, flags=re.DOTALL) is not None
    answer_content = extract_selector_answer(predict_str)

    selected_indices_raw: List[int] = []
    if answer_content:
        nums = re.findall(r"\d+", answer_content)
        for x in nums:
            idx = int(x)
            if 1 <= idx <= num_candidates:
                selected_indices_raw.append(idx - 1)

    has_duplicate = len(selected_indices_raw) != len(set(selected_indices_raw))
    format_valid = has_answer_tag and len(selected_indices_raw) > 0

    seen = set()
    selected_indices = []
    for i in selected_indices_raw:
        if i not in seen:
            selected_indices.append(i)
            seen.add(i)

    if not selected_indices:
        k = min(default_k, num_candidates)
        selected_indices = list(range(k))

    return selected_indices, format_valid, has_duplicate


# =================== Generator Prompt (truncate here only) ===================

def _format_docs_for_generator_with_budget(
    docs: List[EvidenceDoc],
    tokenizer,
    total_budget_tokens: int
) -> str:
    """
    Keep Doc/Title/Content headers untruncated; truncate document text by evenly splitting budget (prefix kept).
    """
    if not docs:
        return ""

    if tokenizer is None:
        parts = []
        for i, d in enumerate(docs, start=1):
            parts.append(f"Doc [{i}]:\nTitle: {d.title}\nContent: {d.text}\n")
        return "\n".join(parts)

    overhead_tokens = []
    contents = []
    headers = []
    for i, d in enumerate(docs, start=1):
        header = f"Doc [{i}]:\nTitle: {d.title}\nContent: "
        headers.append(header)
        overhead_tokens.append(count_tokens(header, tokenizer) + count_tokens("\n", tokenizer))
        contents.append(str(d.text))

    total_overhead = sum(overhead_tokens)
    budget_for_contents = total_budget_tokens - total_overhead
    if budget_for_contents < 0:
        budget_for_contents = 0

    truncated_contents = allocate_and_truncate_texts_equal(contents, tokenizer, budget_for_contents)

    parts = []
    for i, d in enumerate(docs, start=1):
        parts.append(f"{headers[i-1]}{truncated_contents[i-1]}\n")
    return "\n".join(parts)


def build_generator_messages(question: str, docs: List[EvidenceDoc]) -> List[Dict[str, str]]:
    """
    Build generator chat messages and apply GENERATOR_MAX_INPUT_TOKENS truncation:
      - Count tokens for fixed prompt parts
      - Allocate remaining budget evenly across document contents (prefix kept)
    """
    tokenizer = get_generator_tokenizer()

    prompt_prefix = (
        "You are given a question and retrieved documents.\n"
        "You MUST answer the question using ONLY information from the retrieved documents.\n"
        "Even for yes/no questions, decide yes or no by reasoning from facts in the documents.\n\n"
        "Output format (STRICT):\n"
        "<think> ... </think>\n"
        "<answer> ... </answer>\n\n"
        "Evidence citation rule:\n"
        "- Whenever you use a piece of evidence from the documents in your reasoning, you MUST cite it inline as Doc [i].\n"
        "- You may cite one or multiple documents, but only cite documents that are actually relevant.\n"
        "- Keep <think> concise (1-3 sentences).\n\n"
        "Answer rules:\n"
        "- If the question is yes/no, <answer> must be exactly one of: yes / no / unknown.\n"
        "- Otherwise, <answer> must be a short phrase.\n"
        "- If the documents do not provide enough information, output <answer> unknown </answer>.\n"
        "- Do NOT output anything outside <think> and <answer>.\n\n"
        "Example (do NOT copy the content, only follow the style):\n"
        "<think>Doc [1] states that Future Ted serves as the show's narrator, and Doc [4] confirms the narrator is voiced by Bob Saget.</think>\n"
        "<answer> Ted Mosby </answer>\n\n"
        f"=== QUESTION ===\n"
        f"{question}\n"
        f"=== END QUESTION ===\n\n"
        f"=== RETRIEVED DOCUMENTS ===\n"
    )
    prompt_suffix = "=== END DOCUMENTS ===\n"

    if tokenizer is None:
        docs_str = _format_docs_for_generator_with_budget(docs, tokenizer=None, total_budget_tokens=0)
        prompt_text = prompt_prefix + docs_str + prompt_suffix
        return [{"role": "user", "content": prompt_text}]

    fixed_tokens = count_tokens(prompt_prefix, tokenizer) + count_tokens(prompt_suffix, tokenizer)
    budget_for_docs = GENERATOR_MAX_INPUT_TOKENS - fixed_tokens
    if budget_for_docs < 0:
        budget_for_docs = 0

    docs_str = _format_docs_for_generator_with_budget(docs, tokenizer, budget_for_docs)
    prompt_text = prompt_prefix + docs_str + prompt_suffix

    return [{"role": "user", "content": prompt_text}]


# =================== Generator API Calls ===================

def call_generator_api_sync(
    messages: List[Dict[str, str]],
    n: int = 1,
) -> List[str]:
    """Synchronous generator API call."""
    import requests

    headers = {"Content-Type": "application/json"}
    if GENERATOR_API_KEY:
        headers["Authorization"] = f"Bearer {GENERATOR_API_KEY}"

    payload = {
        "model": GENERATOR_MODEL_NAME,
        "messages": messages,
        "n": n,
        "temperature": GENERATOR_TEMPERATURE,
        "top_p": GENERATOR_TOP_P,
        "max_tokens": GENERATOR_MAX_TOKENS,
    }

    try:
        resp = requests.post(GENERATOR_API_URL, json=payload, headers=headers, timeout=120)
        if resp.status_code != 200:
            print(f"[Generator API Error] Status: {resp.status_code}, Response: {resp.text}")
            return []

        result = resp.json()
        outputs = []
        for choice in result.get("choices", []):
            content = choice.get("message", {}).get("content", "")
            outputs.append(content.strip())
        return outputs
    except Exception as e:
        print(f"[Generator API Error] {e}")
        return []


async def call_generator_api_async(
    session: aiohttp.ClientSession,
    messages: List[Dict[str, str]],
    n: int = 1,
) -> List[str]:
    """Asynchronous external generator API call."""
    headers = {"Content-Type": "application/json"}
    if GENERATOR_API_KEY:
        headers["Authorization"] = f"Bearer {GENERATOR_API_KEY}"

    payload = {
        "model": GENERATOR_MODEL_NAME,
        "messages": messages,
        "n": n,
        "temperature": GENERATOR_TEMPERATURE,
        "top_p": GENERATOR_TOP_P,
        "max_tokens": GENERATOR_MAX_TOKENS,
    }

    try:
        async with session.post(GENERATOR_API_URL, json=payload, headers=headers, timeout=120) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                print(f"[Generator API Error] Status: {resp.status}, Response: {error_text}")
                return []

            result = await resp.json()
            outputs = []
            for choice in result.get("choices", []):
                content = choice.get("message", {}).get("content", "")
                outputs.append(content.strip())
            return outputs
    except asyncio.TimeoutError:
        print("[Generator API Error] Request timeout")
        return []
    except Exception as e:
        print(f"[Generator API Error] {e}")
        return []


# =================== Reward Computation ===================

def compute_rollouts(
    question: str,
    selected_docs: List[EvidenceDoc],
    ground_truth_answers: List[str],
    k: int,
) -> List[RolloutResult]:
    """Run K rollouts on selected docs and compute per-rollout results."""
    messages = build_generator_messages(question, selected_docs)
    outputs = call_generator_api_sync(messages, n=k)

    if not outputs:
        return []

    rollout_results: List[RolloutResult] = []

    acc_weight = GENERATOR_REWARD_CFG["acc_weight"]
    f1_weight = GENERATOR_REWARD_CFG["f1_weight"]
    em_weight = GENERATOR_REWARD_CFG["em_weight"]
    target_cite_count = GENERATOR_REWARD_CFG["target_cite_count"]
    max_doc_id = GENERATOR_REWARD_CFG["max_doc_id"]
    reward_threshold = GENERATOR_REWARD_CFG["reward_threshold"]

    for _, text in enumerate(outputs):
        reward_info = compute_generator_reward(
            predict_str=text,
            golden_answers=ground_truth_answers,
            max_doc_id=max_doc_id,
            target_cite_count=target_cite_count,
            acc_weight=acc_weight,
            f1_weight=f1_weight,
            em_weight=em_weight,
        )

        is_correct = reward_info["total_reward"] >= reward_threshold

        rollout_results.append(RolloutResult(
            raw_output=text,
            extracted_answer=reward_info["extracted_answer"],
            total_reward=reward_info["total_reward"],
            acc_reward=reward_info["acc_reward"],
            cite_reward=reward_info["cite_reward"],
            max_f1=reward_info["max_f1"],
            max_em=reward_info["max_em"],
            cite_count=reward_info["cite_count"],
            has_format=reward_info["has_format"],
            is_correct=is_correct,
        ))

    return rollout_results


def triangular_uncertainty_reward(
    rollouts: List[RolloutResult],
    center: float = 0.6,
    max_reward: float = 1.0,
) -> float:
    """Triangular uncertainty reward: highest when accuracy is closest to center."""
    if not rollouts:
        return 0.0

    total = len(rollouts)
    correct = sum(1 for r in rollouts if r.is_correct)
    p_correct = correct / total

    if p_correct <= center:
        score = p_correct / center
    else:
        score = (1.0 - p_correct) / (1.0 - center)

    score = max(0.0, min(score, 1.0))
    return max_reward * score


def relevance_reward(
    docs: List[EvidenceDoc],
    max_reward: float = 0.5,
) -> float:
    """Relevance reward computed from selector_score."""
    if not docs:
        return 0.0

    scores = [max(d.score, 0.0) for d in docs]
    avg_score = sum(scores) / len(scores)

    C = 10.0
    normalized = avg_score / (avg_score + C)
    return max_reward * normalized


def doc_count_penalty(
    num_selected: int,
    target_num: int = 5,
    penalty_per_diff: float = 0.5,
    max_penalty: float = 1.0,
) -> float:
    """Penalty when the number of selected docs differs from the target."""
    diff = abs(num_selected - target_num)
    if diff == 0:
        return 0.0

    penalty = diff * penalty_per_diff
    return min(penalty, max_penalty)


def compute_single_reward(
    completion: str,
    extra_info: Dict[str, Any],
) -> float:
    """Compute reward for a single sample."""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    original_completion = copy.copy(completion)

    question = extra_info.get("question", "")
    docs_raw = extra_info.get("ctxs", [])
    ground_truth_answers = extra_info.get("golden_answers", [])

    if not question or not docs_raw or not ground_truth_answers:
        debug_log_block([
            "=" * 100,
            f"[{now}] SKIP - incomplete data",
            f"[{now}] has_question={bool(question)} num_docs={len(docs_raw)} num_answers={len(ground_truth_answers)}",
            f"[{now}] >>> FINAL REWARD: 0.0 <<<",
            "=" * 100,
            "",
        ])
        return 0.0

    candidate_docs = [
        EvidenceDoc(
            doc_id=str(i),
            text=str(d.get("doc", "")),
            title=str(d.get("title", "")),
            score=float(d.get("selector_score", d.get("rerank_score", 0.0))),
        )
        for i, d in enumerate(docs_raw)
    ]

    selected_indices, format_valid, has_duplicate = parse_selector_output_to_doc_indices(
        completion,
        len(candidate_docs),
        REWARD_CFG["default_k_docs"]
    )

    selector_think = extract_selector_think(completion)
    selector_answer = extract_selector_answer(completion)

    if not format_valid:
        debug_log_block([
            "=" * 100,
            f"[{now}] FORMAT INVALID",
            f"[{now}] question={_maybe_truncate(question, 200)}",
            f"[{now}] selector_output={_maybe_truncate(original_completion, 500)}",
            f"[{now}] format_valid={format_valid} has_duplicate={has_duplicate}",
            f"[{now}] selector_answer='{selector_answer}'",
            f"[{now}] >>> FINAL REWARD: 0.0 <<<",
            "=" * 100,
            "",
        ])
        return 0.0

    if has_duplicate:
        debug_log_block([
            "=" * 100,
            f"[{now}] DUPLICATE INDICES DETECTED",
            f"[{now}] question={_maybe_truncate(question, 200)}",
            f"[{now}] selector_answer='{selector_answer}'",
            f"[{now}] selected_indices_after_dedup={[i+1 for i in selected_indices]}",
            f"[{now}] >>> FINAL REWARD: 0.0 <<<",
            "=" * 100,
            "",
        ])
        return 0.0

    selected_docs = [candidate_docs[i] for i in selected_indices if i < len(candidate_docs)]

    num_selected = len(selected_indices)
    target_num = REWARD_CFG["target_num_docs"]
    count_penalty = doc_count_penalty(
        num_selected=num_selected,
        target_num=target_num,
        penalty_per_diff=REWARD_CFG["wrong_count_penalty"],
        max_penalty=REWARD_CFG["max_count_penalty"],
    )

    rollouts = compute_rollouts(
        question=question,
        selected_docs=selected_docs,
        ground_truth_answers=ground_truth_answers,
        k=K_ROLLOUTS,
    )

    if not rollouts:
        debug_log_block([
            "=" * 100,
            f"[{now}] API ERROR - no rollouts returned",
            f"[{now}] question={_maybe_truncate(question, 200)}",
            f"[{now}] selected_docs={[i+1 for i in selected_indices]}",
            f"[{now}] >>> FINAL REWARD: -0.5 <<<",
            "=" * 100,
            "",
        ])
        return -0.5

    num_correct = sum(1 for r in rollouts if r.is_correct)
    accuracy = num_correct / len(rollouts)
    avg_total_reward = sum(r.total_reward for r in rollouts) / len(rollouts)
    avg_acc_reward = sum(r.acc_reward for r in rollouts) / len(rollouts)
    avg_cite_reward = sum(r.cite_reward for r in rollouts) / len(rollouts)
    avg_f1 = sum(r.max_f1 for r in rollouts) / len(rollouts)
    avg_em = sum(r.max_em for r in rollouts) / len(rollouts)
    avg_cite_count = sum(r.cite_count for r in rollouts) / len(rollouts)

    unc_reward = triangular_uncertainty_reward(
        rollouts,
        center=REWARD_CFG["target_center"],
        max_reward=REWARD_CFG["max_unc_reward"],
    )

    rel_reward = relevance_reward(
        selected_docs,
        max_reward=REWARD_CFG["max_rel_reward"],
    )

    format_bonus = REWARD_CFG["format_reward"]
    total = (
        REWARD_CFG["lambda_unc"] * unc_reward +
        REWARD_CFG["lambda_rel"] * rel_reward +
        format_bonus -
        count_penalty
    )

    rollout_details = []
    for i, r in enumerate(rollouts):
        rollout_details.append(
            f"  [Rollout {i+1}] answer='{_maybe_truncate(r.extracted_answer, 50)}' "
            f"f1={r.max_f1:.3f} em={r.max_em:.1f} cite={r.cite_count} "
            f"total_reward={r.total_reward:.3f} is_correct={r.is_correct}"
        )

    debug_log_block([
        "=" * 100,
        f"[{now}] REWARD COMPUTATION",
        f"[{now}] question={_maybe_truncate(question, 300)}",
        f"[{now}] ground_truth={ground_truth_answers}",
        f"[{now}] selector_think={selector_think}",
        f"[{now}] selector_answer='{selector_answer}'",
        f"[{now}] selected_docs={[i+1 for i in selected_indices]} (num={num_selected}, target={target_num})",
        f"[{now}] format_valid={format_valid} has_duplicate={has_duplicate}",
        "-" * 50,
        f"[{now}] GENERATOR ROLLOUTS (K={K_ROLLOUTS}):",
        *rollout_details,
        "-" * 50,
        f"[{now}] ROLLOUT STATS:",
        f"[{now}]   num_correct={num_correct}/{len(rollouts)} accuracy={accuracy:.3f}",
        f"[{now}]   avg_total_reward={avg_total_reward:.4f}",
        f"[{now}]   avg_acc_reward={avg_acc_reward:.4f} (avg_f1={avg_f1:.4f}, avg_em={avg_em:.4f})",
        f"[{now}]   avg_cite_reward={avg_cite_reward:.4f} (avg_cite_count={avg_cite_count:.1f})",
        "-" * 50,
        f"[{now}] SELECTOR REWARD COMPONENTS:",
        f"[{now}]   unc_reward={unc_reward:.4f} (center={REWARD_CFG['target_center']}, max={REWARD_CFG['max_unc_reward']})",
        f"[{now}]   rel_reward={rel_reward:.4f} (max={REWARD_CFG['max_rel_reward']})",
        f"[{now}]   format_bonus={format_bonus:.4f}",
        f"[{now}]   count_penalty={count_penalty:.4f}",
        "-" * 50,
        f"[{now}] FINAL CALCULATION:",
        f"[{now}]   total = {REWARD_CFG['lambda_unc']:.1f}*{unc_reward:.4f} + {REWARD_CFG['lambda_rel']:.1f}*{rel_reward:.4f} + {format_bonus:.4f} - {count_penalty:.4f}",
        f"[{now}] >>> FINAL REWARD: {total:.4f} <<<",
        "=" * 100,
        "",
    ])

    return float(total)


# =================== compute_score: Entry Point ===================

def compute_score(
    predict_str: str,
    ground_truth: str,
    **kwargs,
) -> float:
    """Main entry point for verl to compute selector reward."""
    if "extra_info" not in kwargs:
        raise ValueError("compute_score needs extra_info in kwargs")

    extra_info = kwargs["extra_info"]
    reward = compute_single_reward(predict_str, extra_info)
    return reward


# =================== TRL-Compatible Reward Function ===================

def selector_reward_func(
    prompts: List,
    completions: List,
    extra_info: Optional[List[Dict]] = None,
    **kwargs,
) -> List[float]:
    """TRL GRPOTrainer-compatible reward function."""
    rewards = []

    if completions and isinstance(completions[0], list):
        completion_texts = [comp[-1]["content"] if comp else "" for comp in completions]
    else:
        completion_texts = completions

    if extra_info is None:
        extra_info = kwargs.get("extra_info", [{}] * len(completions))

    for i, completion in enumerate(completion_texts):
        info = extra_info[i] if i < len(extra_info) else {}
        reward = compute_single_reward(completion, info)
        rewards.append(reward)

    return rewards


# =================== Convenience Factory ===================

def create_selector_reward_func(
    generator_api_url: Optional[str] = None,
    generator_model_name: Optional[str] = None,
    k_rollouts: Optional[int] = None,
    acc_weight: Optional[float] = None,
    f1_weight: Optional[float] = None,
    em_weight: Optional[float] = None,
    target_cite_count: Optional[int] = None,
    reward_threshold: Optional[float] = None,
    **reward_config,
):
    """Create a configured reward function."""
    global GENERATOR_API_URL, GENERATOR_MODEL_NAME, K_ROLLOUTS
    global REWARD_CFG, GENERATOR_REWARD_CFG

    if generator_api_url:
        GENERATOR_API_URL = generator_api_url
    if generator_model_name:
        GENERATOR_MODEL_NAME = generator_model_name
    if k_rollouts:
        K_ROLLOUTS = k_rollouts

    if acc_weight is not None:
        GENERATOR_REWARD_CFG["acc_weight"] = acc_weight
    if f1_weight is not None:
        GENERATOR_REWARD_CFG["f1_weight"] = f1_weight
    if em_weight is not None:
        GENERATOR_REWARD_CFG["em_weight"] = em_weight
    if target_cite_count is not None:
        GENERATOR_REWARD_CFG["target_cite_count"] = target_cite_count
    if reward_threshold is not None:
        GENERATOR_REWARD_CFG["reward_threshold"] = reward_threshold

    for key, value in reward_config.items():
        if key in REWARD_CFG:
            REWARD_CFG[key] = value

    return selector_reward_func


if __name__ == "__main__":
    test_completion = "<think>Doc [1] directly states Paris is the capital, and Doc [3] provides supporting context about France.</think>\n<answer> [1], [3] </answer>"
    test_extra_info = {
        "question": "What is the capital of France?",
        "ctxs": [
            {"title": "France", "doc": "Paris is the capital of France.", "selector_score": 0.9},
            {"title": "Germany", "doc": "Berlin is the capital of Germany.", "selector_score": 0.5},
            {"title": "Paris", "doc": "Paris is a beautiful city located in France.", "selector_score": 0.8},
        ],
        "golden_answers": ["Paris"],
    }

    print("Testing reward function...")
    print(f"Completion: {test_completion}")
    print(f"Extra info: {test_extra_info}")
    print()

    indices, format_valid, has_dup = parse_selector_output_to_doc_indices(test_completion, 3, 5)
    print(f"Parsed indices: {indices}")
    print(f"Format valid: {format_valid}")
    print(f"Has duplicate: {has_dup}")
    print(f"Selector think: {extract_selector_think(test_completion)}")
    print(f"Selector answer: {extract_selector_answer(test_completion)}")
