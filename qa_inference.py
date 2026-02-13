#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QA inference with vLLM.

- doc selection: use FIRST k ctxs (ctxs[:k])
- doc formatting: Doc [i] / Title / Content
- prompt template: fixed to prompt3
- no token budget, no truncation
"""

import argparse
import json
from dataclasses import dataclass
from typing import List, Dict, Any

from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


DEFAULT_K = 5


@dataclass
class QASample:
    qid: str
    question: str
    answers: List[str]
    ctxs: List[Dict[str, Any]]


def load_qa_data(path: str) -> List[QASample]:
    data: List[QASample] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            data.append(
                QASample(
                    qid=str(obj.get("id", "")),
                    question=obj.get("question", ""),
                    answers=obj.get("answers", []) or [],
                    ctxs=obj.get("ctxs", []) or [],
                )
            )
    return data


def build_prompt_prefix(question: str) -> str:
    return (
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


def format_doc_block(i: int, title: str, content: str) -> str:
    return f"Doc [{i}]:\nTitle: {title}\nContent: {content}\n\n"


def build_prompt(question: str, selected_ctxs: List[Dict[str, Any]]) -> str:
    prefix = build_prompt_prefix(question)
    suffix = "=== END DOCUMENTS ===\n"

    doc_blocks = []
    for i, ctx in enumerate(selected_ctxs, start=1):
        title = ctx.get("title", "")
        content = ctx.get("doc", "")
        doc_blocks.append(format_doc_block(i, title, content))

    return prefix + "".join(doc_blocks) + suffix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)

    parser.add_argument("--top_k_ctx", type=int, default=DEFAULT_K)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)

    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--max_model_len", type=int, default=None)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)

    parser.add_argument("--print_first_prompt", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    k = int(args.top_k_ctx)
    if k <= 0:
        raise ValueError(f"--top_k_ctx must be positive, got {k}")

    print("Loading QA data...")
    samples = load_qa_data(args.data_path)
    print(f"Loaded {len(samples)} samples from {args.data_path}")

    start_index = 0
    if args.resume:
        import os

        if os.path.exists(args.output_path):
            with open(args.output_path, "r", encoding="utf-8") as f:
                completed_count = sum(1 for line in f if line.strip())
            if completed_count > 0:
                start_index = completed_count
                print(f"Found {completed_count} completed records. Resuming.")
        else:
            print("Output file not found. Starting from beginning.")

    if start_index > 0:
        if start_index >= len(samples):
            print(f"All {len(samples)} samples already processed.")
            return
        samples = samples[start_index:]
        print(f"Remaining samples: {len(samples)}")

    print("Building prompts...")
    prompts: List[str] = []
    meta: List[Dict[str, Any]] = []

    for ex in tqdm(samples, desc="Prompting", unit="sample"):
        selected_ctxs = (ex.ctxs or [])[:k]
        prompt = build_prompt(ex.question, selected_ctxs)

        prompts.append(prompt)
        meta.append(
            {
                "prompt_type": "prompt3",
                "docs_used": len(selected_ctxs),
            }
        )

    if args.print_first_prompt and prompts:
        print("\n" + "=" * 80)
        print("FIRST PROMPT PREVIEW")
        print("=" * 80)
        print(prompts[0])
        print("=" * 80 + "\n")

    print("Initializing vLLM...")
    llm_kwargs = dict(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = int(args.max_model_len)

    llm = LLM(**llm_kwargs)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        n=1,
        stop=None,
    )

    print("Generating...")
    total = len(prompts)
    pbar = tqdm(total=total, desc="Inference", unit="sample")

    file_mode = "a" if (args.resume and start_index > 0) else "w"

    with open(args.output_path, file_mode, encoding="utf-8") as fout:
        for i in range(0, total, args.batch_size):
            batch_prompts = prompts[i : i + args.batch_size]
            batch_samples = samples[i : i + args.batch_size]
            batch_meta = meta[i : i + args.batch_size]

            batch_outputs = llm.generate(batch_prompts, sampling_params, use_tqdm=False)

            for ex, out, m in zip(batch_samples, batch_outputs, batch_meta):
                pred = out.outputs[0].text.strip() if out.outputs else ""
                record = {
                    "id": ex.qid,
                    "question": ex.question,
                    "answers": ex.answers,
                    "prediction": pred,
                    "meta": m,
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")

            pbar.update(len(batch_samples))

    pbar.close()
    print(f"Done. Saved predictions to: {args.output_path}")


if __name__ == "__main__":
    main()
