#!/usr/bin/env python3
import argparse
import os
import json
from typing import List, Dict, Any, Tuple
from collections import Counter

import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm


SYSTEM_CONTENT = (
    "You are an expert evidence-set selector for RAG. Your goal is to select a SMALL set of "
    "documents that makes the question answerable, BUT NOT trivial. Prefer evidence sets that "
    "sit near the model's competence boundary: solvable with careful multi-step reasoning, "
    "yet not so direct that the answer is obvious from a single passage.\n\n"
    "Principles:\n"
    "1) Answerability (must-have): The selected set must contain enough information to deduce the "
    "   correct answer. Do NOT select sets that make the question impossible.\n"
    "2) Non-triviality (must-have): Avoid sets where one document directly states the answer with "
    "   no integration needed. If a direct-answer passage is unavoidable for solvability, include "
    "   it ONLY together with supporting/context passages that require cross-document integration.\n"
    "3) Multi-hop integration: Prefer sets that require combining at least TWO complementary clues "
    "   (e.g., entity linking, temporal alignment, resolving aliases, chaining relations).\n"
    "4) Controlled noise: Mildly conflicting or distracting details are allowed if the set remains "
    "   answerable; do not include documents that are irrelevant or make the set unsolvable.\n"
    "5) Diversity: Prefer complementary documents covering different parts of the reasoning chain, "
    "   rather than near-duplicates.\n\n"
    "Output format (STRICT):\n"
    "<think> Your reasoning process for selecting documents. Briefly explain which documents "
    "contain relevant clues, how they complement each other, and why this set is answerable "
    "but requires integration. Keep it concise (2-4 sentences). </think>\n"
    "<answer> [doc_id1], [doc_id2], [doc_id3], [doc_id4], [doc_id5] </answer>\n\n"
    "Rules:\n"
    "- Select exactly 5 documents.\n"
    "- In <answer>, list ONLY the document identifiers in brackets, separated by commas.\n"
    "- Do NOT output anything outside <think> and <answer>.\n\n"
    "Example (do NOT copy the content, only follow the style):\n"
    "<think>Doc [3] mentions the person's birthplace, Doc [7] provides their career timeline, "
    "and Doc [12] links their alias to their real name. Together these require cross-referencing "
    "to answer the question, while Doc [5] and Doc [9] provide supporting context.</think>\n"
    "<answer> [3], [5], [7], [9], [12] </answer>\n"
)
SYSTEM_MSG = {"role": "system", "content": SYSTEM_CONTENT}


def build_selector_conversation(question: str, retrieved_content: str) -> List[Dict[str, str]]:
    user_msg = {"role": "user", "content": f"Query: {question}\n\n{retrieved_content}"}
    return [SYSTEM_MSG, user_msg]


def _safe_parse_extra_info(extra_info: Any) -> Dict[str, Any]:
    if extra_info is None:
        return {}
    if isinstance(extra_info, dict):
        return extra_info
    if isinstance(extra_info, str):
        s = extra_info.strip()
        if not s:
            return {}
        try:
            return json.loads(s)
        except Exception:
            return {}
    return {}


def normalize_ctxs(ctxs: Any) -> List[Dict[str, Any]]:
    if ctxs is None:
        return []
    if isinstance(ctxs, pd.Series):
        ctxs = ctxs.tolist()
    if isinstance(ctxs, np.ndarray):
        ctxs = ctxs.tolist()
    if hasattr(ctxs, "tolist") and not isinstance(ctxs, list):
        try:
            ctxs = ctxs.tolist()
        except Exception:
            pass
    if isinstance(ctxs, tuple):
        ctxs = list(ctxs)
    if not isinstance(ctxs, list):
        return []
    out = []
    for x in ctxs:
        if isinstance(x, dict):
            out.append(x)
    return out


def count_chat_tokens_apply_template(tokenizer: Any, messages: List[Dict[str, str]]) -> int:
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    return len(ids)


def bucket_overflow(x: int) -> str:
    if x == 0:
        return "0"
    if 1 <= x <= 50:
        return "1-50"
    if 51 <= x <= 100:
        return "51-100"
    if 101 <= x <= 200:
        return "101-200"
    if 201 <= x <= 500:
        return "201-500"
    if 501 <= x <= 1000:
        return "501-1000"
    if 1001 <= x <= 2000:
        return "1001-2000"
    if 2001 <= x <= 4000:
        return "2001-4000"
    return "4001+"


def build_docs_with_budget(
    tokenizer: Any,
    ctxs: List[Dict[str, Any]],
    top_n: int,
    doc_budget_tokens: int,
) -> Tuple[str, int, int]:
    """
    Tokenize docs chunk-by-chunk and truncate to a budget.

    Returns:
      - docs_text: decoded text after truncation
      - total_doc_tokens: tokens for top_n docs without truncation
      - used_doc_tokens: tokens actually used (<= doc_budget_tokens)
    """
    if doc_budget_tokens <= 0 or not ctxs:
        return "", 0, 0

    used_ids: List[int] = []
    used = 0
    total = 0

    for i, ctx in enumerate(ctxs[:top_n], start=1):
        title = ctx.get("title", "") or ""
        content = ctx.get("doc", "") or ""
        chunk = f"Doc [{i}]:\nTitle: {title}\nContent: {content}\n\n"

        chunk_ids = tokenizer.encode(chunk, add_special_tokens=False)
        chunk_len = len(chunk_ids)
        total += chunk_len

        if used >= doc_budget_tokens:
            continue

        remaining = doc_budget_tokens - used
        if chunk_len <= remaining:
            used_ids.extend(chunk_ids)
            used += chunk_len
        else:
            used_ids.extend(chunk_ids[:remaining])
            used += remaining

    docs_text = tokenizer.decode(used_ids, skip_special_tokens=True)
    return docs_text, total, used


def process_row_fast(
    row: Dict[str, Any],
    top_n: int,
    tokenizer: Any,
    max_input_tokens: int,
    safety_margin: int,
) -> Tuple[Dict[str, Any], int, int, int]:
    row["data_source"] = "selector_training"

    extra_info = _safe_parse_extra_info(row.get("extra_info"))
    question = extra_info.get("question", "") or ""
    ctxs = normalize_ctxs(extra_info.get("ctxs", None))

    user_base = {"role": "user", "content": f"Query: {question}\n\n"}
    base_tokens = count_chat_tokens_apply_template(tokenizer, [SYSTEM_MSG, user_base])

    remaining = max_input_tokens - base_tokens - safety_margin
    if remaining < 0:
        remaining = 0

    docs_text, total_doc_tokens, used_doc_tokens = build_docs_with_budget(
        tokenizer=tokenizer,
        ctxs=ctxs,
        top_n=top_n,
        doc_budget_tokens=remaining,
    )

    full_tokens_est = base_tokens + total_doc_tokens
    overflow_tokens = max(0, full_tokens_est - max_input_tokens)
    final_tokens_est = base_tokens + used_doc_tokens + safety_margin

    final_prompt = build_selector_conversation(question, docs_text)
    row["prompt"] = json.dumps(final_prompt, ensure_ascii=False)

    return row, overflow_tokens, full_tokens_est, final_tokens_est


def main():
    parser = argparse.ArgumentParser(
        description="Process parquet with a Llama tokenizer for selector RL training"
    )
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--top_n", type=int, default=30)
    parser.add_argument("--max_input_tokens", type=int, default=8192)
    parser.add_argument("--tokenizer_name_or_path", default=os.environ.get("TOKENIZER_NAME_OR_PATH", None))
    parser.add_argument("--max_examples", type=int, default=-1)
    parser.add_argument("--print_overflow_topk", type=int, default=30)
    parser.add_argument(
        "--safety_margin",
        type=int,
        default=32,
        help="Extra token budget to avoid boundary mismatch."
    )
    args = parser.parse_args()

    if args.tokenizer_name_or_path is None:
        raise ValueError("Missing --tokenizer_name_or_path (or env TOKENIZER_NAME_OR_PATH).")

    if args.output_path is None:
        base, ext = os.path.splitext(args.input_path)
        args.output_path = f"{base}_processed{ext}"

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"Loading tokenizer from: {args.tokenizer_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path, trust_remote_code=True, use_fast=True)

    print(f"Reading from: {args.input_path}")
    df = pd.read_parquet(args.input_path)
    if args.max_examples != -1:
        df = df.head(args.max_examples)
    print(f"Loaded {len(df)} examples")

    overflow_counter = Counter()
    overflow_bucket_counter = Counter()
    max_overflow = 0
    sum_overflow = 0
    overflowed_n = 0
    max_full_tokens = 0
    max_final_tokens = 0

    processed_rows = []

    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Processing"):
        row_dict = row._asdict()

        processed_row, overflow_tokens, full_tokens, final_tokens = process_row_fast(
            row=row_dict,
            top_n=args.top_n,
            tokenizer=tokenizer,
            max_input_tokens=args.max_input_tokens,
            safety_margin=args.safety_margin,
        )
        processed_rows.append(processed_row)

        overflow_counter[overflow_tokens] += 1
        overflow_bucket_counter[bucket_overflow(overflow_tokens)] += 1
        max_overflow = max(max_overflow, overflow_tokens)
        sum_overflow += overflow_tokens
        if overflow_tokens > 0:
            overflowed_n += 1
        max_full_tokens = max(max_full_tokens, full_tokens)
        max_final_tokens = max(max_final_tokens, final_tokens)

    processed_df = pd.DataFrame(processed_rows)

    def convert_to_json_serializable(obj):
        if isinstance(obj, np.ndarray):
            return [convert_to_json_serializable(x) for x in obj.tolist()]
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_to_json_serializable(x) for x in obj]
        if isinstance(obj, tuple):
            return [convert_to_json_serializable(x) for x in obj]
        return obj

    def safe_json_dumps(val):
        if isinstance(val, str):
            return val
        try:
            converted = convert_to_json_serializable(val)
            return json.dumps(converted, ensure_ascii=False)
        except Exception:
            return str(val)

    for col in processed_df.columns:
        if processed_df[col].dtype == "object":
            processed_df[col] = processed_df[col].apply(safe_json_dumps)

    processed_df.to_parquet(args.output_path, index=False)

    print(f"\nDone! Processed {len(processed_df)} examples")
    print(f"Output saved to: {args.output_path}")

    print("\n" + "=" * 60)
    print("Sample prompt (first example):")
    print("=" * 60)
    sample_prompt = json.loads(processed_df.iloc[0]["prompt"])
    sample_tokens = count_chat_tokens_apply_template(tokenizer, sample_prompt)
    print(f"Sample total tokens (exact): {sample_tokens} / max {args.max_input_tokens}")
    print(f"System message (first 500 chars):\n{sample_prompt[0]['content'][:500]}...")
    print(f"\nUser message (first 300 chars):\n{sample_prompt[1]['content'][:300]}...")

    n = len(processed_df)
    mean_overflow = sum_overflow / max(1, n)
    pct_overflow = 100.0 * overflowed_n / max(1, n)

    print("\n" + "=" * 60)
    print("Overflow token statistics (estimated on untruncated docs):")
    print("=" * 60)
    print(f"Total examples: {n}")
    print(f"Overflowed examples: {overflowed_n} ({pct_overflow:.2f}%)")
    print(f"Mean overflow tokens: {mean_overflow:.2f}")
    print(f"Max overflow tokens: {max_overflow}")
    print(f"Max full tokens (est): {max_full_tokens}")
    print(f"Max final tokens (est): {max_final_tokens} (target <= {args.max_input_tokens})")

    print("\nBucketed overflow distribution:")
    bucket_order = ["0", "1-50", "51-100", "101-200", "201-500", "501-1000", "1001-2000", "2001-4000", "4001+"]
    for b in bucket_order:
        if b in overflow_bucket_counter:
            print(f"  {b:>9}: {overflow_bucket_counter[b]}")

    print("\nExact overflow distribution (first few keys, ascending):")
    keys_sorted = sorted(overflow_counter.keys())
    for k in keys_sorted[:args.print_overflow_topk]:
        print(f"  overflow={k:>5}: {overflow_counter[k]}")
    if len(keys_sorted) > args.print_overflow_topk:
        print(f"  ... ({len(keys_sorted) - args.print_overflow_topk} more overflow values not shown)")

    nonzero_keys = [k for k in keys_sorted if k > 0]
    if nonzero_keys:
        print("\nLargest overflow values (top 10):")
        for k in sorted(nonzero_keys, reverse=True)[:10]:
            print(f"  overflow={k:>5}: {overflow_counter[k]}")


if __name__ == "__main__":
    main()
