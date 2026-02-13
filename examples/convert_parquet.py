#!/usr/bin/env python3

import argparse
import os
from typing import List, Dict, Any, Optional

import datasets

# Optional HDFS utils (verl). If you don't need HDFS, you can remove these imports and code block.
try:
    from verl.utils.hdfs_io import copy, makedirs
except Exception:
    copy = None
    makedirs = None


def format_top_n_docs(ctxs: List[Dict[str, Any]], n: int) -> str:
    """Concatenate top-n docs into 'Doc [i] + Title + Content' format."""
    parts = []
    for i, ctx in enumerate(ctxs[:n], start=1):
        title = ctx.get("title", "")
        content = ctx.get("doc", "")
        parts.append(f"Doc [{i}]:\nTitle: {title}\nContent: {content}\n")
    return "\n".join(parts)


def build_instruction_messages(query: str, retrieved_content: str):
    """Build chat-style prompt: system + user."""
    system_msg = {
        "role": "system",
        "content": (
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
        ),
    }
    user_msg = {
        "role": "user",
        "content": f"Query: {query}\n\n{retrieved_content}",
    }
    return [system_msg, user_msg]


def build_rerank_output(ctxs: List[Dict[str, Any]], n: int) -> str:
    """
    From top-n docs, select those with rerank_label=True, sort by rerank_score desc,
    output: "[1], [3], [7]" or "None".
    """
    candidates = []
    for idx, ctx in enumerate(ctxs[:n], start=1):
        label = ctx.get("rerank_label", False)
        if isinstance(label, str):
            label_bool = label.lower() == "true"
        else:
            label_bool = bool(label)

        if not label_bool:
            continue

        score = ctx.get("rerank_score", 0.0)
        try:
            score = float(score)
        except Exception:
            score = 0.0

        candidates.append((idx, score))

    if not candidates:
        return "None"

    candidates.sort(key=lambda x: x[1], reverse=True)
    indices = [f"[{idx}]" for idx, _ in candidates]
    return ", ".join(indices)


def make_map_fn(split: str, top_n: int):
    """Factory for datasets.map()."""

    def process_fn(example, idx):
        question = example.get("question", "")
        ctxs = example.get("ctxs", []) or []
        golden_answers = example.get("golden_answers", None)
        example_id = example.get("id", None)

        retrieved_content = format_top_n_docs(ctxs, top_n)
        prompt = build_instruction_messages(question, retrieved_content)
        response = build_rerank_output(ctxs, top_n)

        return {
            "data_source": "selector_training",
            "prompt": prompt,
            "response": response,
            "reward_model": {"style": "rule", "ground_truth": response},
            "extra_info": {
                "split": split,
                "index": idx,
                "id": example_id,
                "question": question,
                "golden_answers": golden_answers,
                "ctxs": ctxs,
            },
        }

    return process_fn


def load_json_dataset(input_path: str, split_name: str):
    """
    Load json/jsonl via HF datasets.
    - If input is jsonl: load_dataset("json", data_files={split: path})
    - If input is a single json array: load_dataset("json", data_files=...) also works in most cases.
    """
    ds_dict = datasets.load_dataset(
        "json",
        data_files={split_name: input_path},
    )
    return ds_dict[split_name]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True, help="Path to json/jsonl file.")
    parser.add_argument("--split_name", default="train", help="Split name stored in extra_info.")
    parser.add_argument("--top_n", type=int, default=10, help="Number of top documents in prompt.")
    parser.add_argument("--local_save_dir", default="~/data/rerank_parquet", help="Output dir.")
    parser.add_argument(
        "--output_name",
        default=None,
        help="Output parquet filename (default: <split_name>.parquet).",
    )
    parser.add_argument(
        "--hdfs_dir",
        default=None,
        help="Optional HDFS directory to copy the output directory to (requires verl).",
    )

    args = parser.parse_args()

    raw_split = load_json_dataset(args.input_path, args.split_name)
    num_total = len(raw_split)
    print(f"[Info] Loaded {num_total} examples from: {args.input_path}")

    mapped_split = raw_split.map(
        function=make_map_fn(args.split_name, args.top_n),
        with_indices=True,
    )

    local_save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    out_name = args.output_name or f"{args.split_name}.parquet"
    out_path = os.path.join(local_save_dir, out_name)

    mapped_split.to_parquet(out_path)
    print(f"[OK] Parquet saved to: {out_path}")

    # Optional: copy to HDFS
    if args.hdfs_dir is not None:
        if makedirs is None or copy is None:
            raise RuntimeError(
                "verl.utils.hdfs_io is not available in your env, but --hdfs_dir is set."
            )
        makedirs(args.hdfs_dir)
        copy(src=local_save_dir, dst=args.hdfs_dir)
        print(f"[OK] Copied data from {local_save_dir} to {args.hdfs_dir}")


if __name__ == "__main__":
    main()
