#!/usr/bin/env python3
"""
Use vLLM as a selector to select documents, then build generator SFT dataset.

Pipeline per example:
1. Use vLLM to generate a list of document identifiers (e.g., [1], [3], [7])
   given the query and the top-N retrieved documents (ctxs).
2. Parse the identifiers into a document index list.
3. Use this document list to select the corresponding docs from ctxs,
   and format them as Doc [1]: Title ... Content ... etc.
4. Build SFT data:
   - prompt: thinking template + question + selected documents
   - response: gold answer from the jsonl (answers or golden_answers)

Expected input jsonl schema (example):

{
  "id": "train_0",
  "question": "total number of death row inmates in the us?",
  "answers": ["2,718"],             # or golden_answers
  "data_source": "nq",
  "ctxs": [
    {"title": "Death row", "doc": "...", "score": 0.85, ...},
    ...
  ]
}

Output:
- <split_name>.parquet  # SFT schema
- <split_name>.jsonl    # same schema, one example per line
"""

import argparse
import json
import os
import re
from typing import Any, Dict, List, Tuple

from datasets import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load data from a jsonl file."""
    data: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


# ------------------- Selector System Prompt -------------------
SELECTOR_SYSTEM_CONTENT = (
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
SELECTOR_SYSTEM_MSG = {"role": "system", "content": SELECTOR_SYSTEM_CONTENT}


def normalize_ground_truth(ex: Dict[str, Any]) -> List[str]:
    """Normalize ground truth into List[str]."""
    gt = ex.get("ground_truth", ex.get("answers", ex.get("golden_answers", [])))

    if gt is None:
        return []
    if isinstance(gt, str):
        gt = [gt]
    elif isinstance(gt, (list, tuple)):
        gt = list(gt)
    else:
        gt = [str(gt)]

    out: List[str] = []
    for a in gt:
        s = (a or "").strip()
        if s:
            out.append(s)
    return out


def format_doc_block(i: int, title: str, content: str) -> str:
    """Format a single document block."""
    return f"Doc [{i}]:\nTitle: {title}\nContent: {content}\n\n"


def format_top_n_docs(ctxs: List[Dict[str, Any]], n: int) -> str:
    """
    Format ctxs[:n] into Doc [1], Doc [2], ...
    """
    parts: List[str] = []
    for i, ctx in enumerate(ctxs[:n], start=1):
        title = ctx.get("title", "")
        content = ctx.get("doc", "")
        parts.append(format_doc_block(i, title, content))
    return "".join(parts)


def build_docs_with_budget(
    tokenizer: Any,
    ctxs: List[Dict[str, Any]],
    top_n: int,
    doc_budget_tokens: int,
) -> Tuple[str, int, int]:
    """
    Format docs sequentially and truncate by token budget.
    Returns:
      - docs_text (decoded once)
      - total_doc_tokens: total tokens without truncation
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
        chunk = format_doc_block(i, title, content)

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


def count_chat_tokens(tokenizer: Any, messages: List[Dict[str, str]]) -> int:
    """Count tokens for chat messages."""
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    return len(ids)


def build_selector_conversation_with_budget(
    tokenizer: Any,
    question: str,
    ctxs: List[Dict[str, Any]],
    top_n: int,
    max_input_tokens: int,
    safety_margin: int = 32,
) -> Tuple[List[Dict[str, str]], int, int]:
    """
    Build selector conversation with token-budget truncation.
    Returns: (messages, total_doc_tokens, used_doc_tokens)
    """
    user_base = {"role": "user", "content": f"Query: {question}\n\n"}
    base_tokens = count_chat_tokens(tokenizer, [SELECTOR_SYSTEM_MSG, user_base])

    remaining = max_input_tokens - base_tokens - safety_margin
    if remaining < 0:
        remaining = 0

    docs_text, total_doc_tokens, used_doc_tokens = build_docs_with_budget(
        tokenizer=tokenizer,
        ctxs=ctxs,
        top_n=top_n,
        doc_budget_tokens=remaining,
    )

    user_msg = {"role": "user", "content": f"Query: {question}\n\n{docs_text}"}
    return [SELECTOR_SYSTEM_MSG, user_msg], total_doc_tokens, used_doc_tokens


def build_selector_conversation(question: str, retrieved_content: str) -> List[Dict[str, str]]:
    """Build selector conversation without token budgeting."""
    user_msg = {"role": "user", "content": f"Query: {question}\n\n{retrieved_content}"}
    return [SELECTOR_SYSTEM_MSG, user_msg]


def parse_selected_ids(text: str) -> List[int]:
    """
    Parse selector output and extract doc ids ONLY from <answer>.
    Format: "<think>...</think>\n<answer> [1], [3], [7] </answer>"
    Returns a 1-based index list.
    """
    if text is None:
        return []

    s = text.strip()
    if not s or s.lower() == "none":
        return []

    answer_match = re.search(r"<answer>\s*(.*?)\s*</answer>", s, re.IGNORECASE | re.DOTALL)
    if answer_match:
        answer_content = answer_match.group(1).strip()
        if answer_content.lower() == "none":
            return []
        nums = re.findall(r"\[(\d+)\]", answer_content)
        if nums:
            return [int(x) for x in nums]

    return []


def build_generator_prompt(question: str, selected_docs_str: str) -> str:
    """Build generator prompt."""
    prefix = (
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
        "- The answer should be a short phrase directly from the documents.\n"
        "- Do NOT output anything outside <think> and <answer>.\n\n"
        "Example (do NOT copy the content, only follow the style):\n"
        "<think>Doc [1] states that Future Ted serves as the show's narrator, and Doc [4] confirms the narrator is voiced by Bob Saget.</think>\n"
        "<answer> Ted Mosby </answer>\n\n"
        "=== QUESTION ===\n"
        f"{question}\n"
        "=== END QUESTION ===\n\n"
        "=== RETRIEVED DOCUMENTS ===\n"
    )

    suffix = "=== END DOCUMENTS ===\n"

    if selected_docs_str.strip():
        return prefix + selected_docs_str + suffix
    else:
        return prefix + suffix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        required=True,
        help="Path to the jsonl file (with question, ctxs, answers/ground_truth...).",
    )
    parser.add_argument(
        "--local_save_dir",
        default="~/data/generator_sft_from_vllm_selector",
        help="Directory to save the preprocessed dataset (parquet + jsonl).",
    )
    parser.add_argument(
        "--split_name",
        default="train",
        help="Split name for extra_info (e.g., train/test/dev).",
    )
    parser.add_argument(
        "--selector_top_n",
        type=int,
        default=25,
        help="How many ctxs to include as candidates for the selector (from the front of ctxs).",
    )
    parser.add_argument(
        "--max_selector_input_tokens",
        type=int,
        default=4096,
        help="Max input tokens for selector.",
    )
    parser.add_argument(
        "--safety_margin",
        type=int,
        default=32,
        help="Safety margin for token budget.",
    )

    parser.add_argument(
        "--model_name",
        required=True,
        help="Model name or path for vLLM selector.",
    )
    parser.add_argument(
        "--tokenizer_name",
        default=None,
        help="Tokenizer name or path (defaults to model_name if not provided).",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Tensor parallel size for vLLM.",
    )
    parser.add_argument(
        "--max_new_tokens_selector",
        type=int,
        default=256,
        help="Max new tokens for selector outputs (doc id list + thinking).",
    )
    parser.add_argument(
        "--temperature_selector",
        type=float,
        default=0.0,
        help="Temperature for selector.",
    )
    parser.add_argument(
        "--top_p_selector",
        type=float,
        default=1.0,
        help="Top-p for selector.",
    )
    parser.add_argument(
        "--batch_size_selector",
        type=int,
        default=8,
        help="Batch size for vLLM selector.",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization for vLLM (0.0 to 1.0).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode: only process first debug_batch_size examples and print first example.",
    )
    parser.add_argument(
        "--debug_batch_size",
        type=int,
        default=8,
        help="Number of examples to process in debug mode.",
    )

    args = parser.parse_args()

    tokenizer_name = args.tokenizer_name or args.model_name
    print(f"Loading tokenizer from: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True, use_fast=True)

    print(f"Loading jsonl data from {args.input_path}...")
    raw_examples = load_jsonl(args.input_path)
    print(f"Loaded {len(raw_examples)} examples")
    if raw_examples:
        print(f"First example keys: {list(raw_examples[0].keys())}")

    if args.debug:
        raw_examples = raw_examples[:args.debug_batch_size]
        print(f"[DEBUG MODE] Only processing first {len(raw_examples)} examples")

    selector_convs: List[List[Dict[str, str]]] = []
    meta_list: List[Dict[str, Any]] = []
    truncation_stats = {"total": 0, "truncated": 0}

    for idx, ex in enumerate(tqdm(raw_examples, desc="Building selector prompts")):
        question = ex.get("question", "")
        ctxs = ex.get("ctxs", []) or []
        candidate_ctxs = ctxs[: args.selector_top_n]

        conv, total_doc_tokens, used_doc_tokens = build_selector_conversation_with_budget(
            tokenizer=tokenizer,
            question=question,
            ctxs=candidate_ctxs,
            top_n=args.selector_top_n,
            max_input_tokens=args.max_selector_input_tokens,
            safety_margin=args.safety_margin,
        )
        selector_convs.append(conv)

        truncation_stats["total"] += 1
        if total_doc_tokens > used_doc_tokens:
            truncation_stats["truncated"] += 1

        raw_id = ex.get("id", None)
        str_id = str(raw_id) if raw_id is not None else None
        meta_list.append(
            {
                "index": idx,
                "id": str_id,
                "question": question,
                "ctxs": ctxs,
            }
        )

    print(
        f"Selector input truncation: {truncation_stats['truncated']}/{truncation_stats['total']} examples truncated"
    )

    print(f"Initializing vLLM selector model: {args.model_name}")
    llm = LLM(
        model=args.model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
    )

    selector_sampling_params = SamplingParams(
        temperature=args.temperature_selector,
        top_p=args.top_p_selector,
        max_tokens=args.max_new_tokens_selector,
    )

    all_selector_texts: List[str] = []
    total = len(selector_convs)

    for start in tqdm(range(0, total, args.batch_size_selector), desc="Running vLLM selector"):
        end = min(start + args.batch_size_selector, total)
        batch_convs = selector_convs[start:end]

        outputs = llm.chat(
            messages=batch_convs,
            sampling_params=selector_sampling_params,
            use_tqdm=False,
        )

        for out in outputs:
            text = out.outputs[0].text.strip()
            all_selector_texts.append(text)

    assert len(all_selector_texts) == len(raw_examples)

    sft_examples: List[Dict[str, Any]] = []
    parse_stats = {"total": 0, "failed": 0}
    first_failed_example = None

    for ex, meta, selector_text in zip(raw_examples, meta_list, all_selector_texts):
        question = meta["question"]
        ctxs = meta["ctxs"]

        indices = parse_selected_ids(selector_text)
        parse_stats["total"] += 1

        selected_ctxs: List[Dict[str, Any]] = []
        for i in indices:
            if 1 <= i <= len(ctxs):
                selected_ctxs.append(ctxs[i - 1])

        if not selected_ctxs:
            parse_stats["failed"] += 1
            if first_failed_example is None:
                first_failed_example = {
                    "index": meta["index"],
                    "id": meta["id"],
                    "question": question,
                    "selector_raw_output": selector_text,
                }
            continue

        selected_docs_str = format_top_n_docs(selected_ctxs, len(selected_ctxs))
        final_prompt_text = build_generator_prompt(question, selected_docs_str)

        ground_truth = normalize_ground_truth(ex)
        response = ground_truth[0] if ground_truth else ""

        sft_example = {
            "data_source": "rg-generator-training",
            "prompt": [{"role": "user", "content": final_prompt_text}],
            "response": response,
            "reward_model": {"style": "rule", "ground_truth": ground_truth},
            "extra_info": {
                "split": args.split_name,
                "index": meta["index"],
                "id": meta["id"],
                "question": question,
                "answers": ground_truth,
                "selector_raw_output": selector_text,
                "selected_doc_indices": indices,
                "num_selected_docs": len(selected_ctxs),
            },
        }
        sft_examples.append(sft_example)

    print(f"\nParsing failed: {parse_stats['failed']}/{parse_stats['total']} examples skipped (no valid <answer> tag)")
    print(f"Successfully parsed: {len(sft_examples)} examples")

    if first_failed_example:
        print("\n" + "=" * 80)
        print("[FAILED EXAMPLE] First failed parsing example:")
        print("=" * 80)
        print(f"Index: {first_failed_example['index']}")
        print(f"ID: {first_failed_example['id']}")
        print(f"Question: {first_failed_example['question']}")
        print(f"Selector raw output:\n{first_failed_example['selector_raw_output']}")
        print("=" * 80 + "\n")

    if args.debug and sft_examples:
        print("\n" + "=" * 80)
        print("[DEBUG] First example:")
        print("=" * 80)
        first_example = sft_examples[0]
        print("\nPROMPT (user content):")
        print("-" * 80)
        print(first_example["prompt"][0]["content"])
        print("-" * 80)
        print(f"\nGROUND TRUTH (list): {first_example['reward_model']['ground_truth']}")
        print(f"\nSELECTOR RAW OUTPUT: {first_example['extra_info']['selector_raw_output']}")
        print(f"SELECTED DOC INDICES: {first_example['extra_info']['selected_doc_indices']}")
        print(f"NUM SELECTED DOCS: {first_example['extra_info']['num_selected_docs']}")
        print("=" * 80 + "\n")

    local_save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    parquet_path = os.path.join(local_save_dir, f"{args.split_name}.parquet")
    jsonl_path = os.path.join(local_save_dir, f"{args.split_name}.jsonl")

    hf_dataset = Dataset.from_list(sft_examples)
    hf_dataset.to_parquet(parquet_path)
    hf_dataset.to_json(jsonl_path)

    print(f"\nDone! Processed {len(hf_dataset)} examples.")
    print(f"Saved to: {parquet_path}")
    print(f"Saved to: {jsonl_path}")
    print(f"Dataset columns: {hf_dataset.column_names}")


if __name__ == "__main__":
    main()
