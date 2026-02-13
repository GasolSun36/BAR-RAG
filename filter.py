#!/usr/bin/env python3
"""
Hard-but-solvable filtering with dual vLLM workers.

Dual vLLM worker design:
- Selector: GPU 0-3 (separate process)
- Generator: GPU 4-7 (separate process)
- Main process does NOT import vLLM; only scheduling + data processing

Outputs (no analysis fields):
- filtered_{timestamp}.jsonl: only kept samples (minimal fields)
- filtered_ids_{timestamp}.json: ids + filter config
"""

import os
import json
import argparse
import multiprocessing as mp
from multiprocessing import Queue
from queue import Empty
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import numpy as np
from tqdm import tqdm
from datetime import datetime
import re
import string
import collections

from datasets import load_dataset
from transformers import AutoTokenizer


# =================== Data structures ===================

@dataclass
class GeneratorRollout:
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


@dataclass
class SelectorRollout:
    selector_output: str
    selector_think: str
    selector_answer: str
    selected_doc_indices: List[int]  # 0-based
    generator_rollouts: List[GeneratorRollout]
    num_correct: int
    accuracy: float


@dataclass
class SampleResult:
    sample_id: str
    question: str
    golden_answers: List[str]
    ctxs: List[Dict[str, Any]]
    num_candidate_docs: int
    selector_rollouts: List[SelectorRollout]
    accuracy_list: List[float]
    accuracy_mean: float
    accuracy_std: float
    accuracy_variance: float
    hbs_score: float
    keep_for_grpo: bool


@dataclass
class PreparedSample:
    sample_id: str
    question: str
    golden_answers: List[str]
    ctxs: List[Dict[str, Any]]
    num_candidate_docs: int
    selector_conv: List[Dict[str, str]]


# =================== Text utils ===================

def normalize_answer(s: str) -> str:
    if s is None:
        return ""

    def lower(text): return text.lower()

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def remove_articles(text): return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text): return " ".join(text.split())

    return white_space_fix(remove_articles(remove_punc(lower(s))))


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


def extract_think_tag(text: str) -> str:
    m = re.search(r"<think>\s*(.*?)\s*</think>", text, flags=re.DOTALL | re.IGNORECASE)
    if not m:
        return ""
    return m.group(1).strip()


def count_unique_doc_cites_in_think(predict_str: str, max_doc_id: int = 31) -> int:
    think = extract_think_tag(predict_str)
    if not think:
        return 0

    doc_pattern = re.compile(r"Doc\s*\[\s*(\d+)\s*\]", flags=re.IGNORECASE)
    ids: set = set()
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
) -> Dict[str, float]:
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
    em_scores = [float(normalize_answer(answer) == normalize_answer(gt)) for gt in golden_answers]

    max_f1 = max(f1_scores) if f1_scores else 0.0
    max_em = max(em_scores) if em_scores else 0.0

    result["max_f1"] = max_f1
    result["max_em"] = max_em

    acc_reward = 0.7 * max_f1 + 0.3 * max_em
    result["acc_reward"] = acc_reward

    cite_count = count_unique_doc_cites_in_think(predict_str, max_doc_id=max_doc_id)
    cite_reward = cite_reward_peaked(predict_str, target_cite_count=target_cite_count, max_doc_id=max_doc_id)
    result["cite_count"] = cite_count
    result["cite_reward"] = cite_reward

    total_reward = acc_weight * acc_reward + (1 - acc_weight) * cite_reward
    result["total_reward"] = total_reward
    return result


# =================== Selector output parsing ===================

def extract_selector_answer(text: str) -> str:
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.DOTALL)
    if not m:
        return text
    return m.group(1).strip()


def extract_selector_think(text: str) -> str:
    m = re.search(r"<think>\s*(.*?)\s*</think>", text, flags=re.DOTALL)
    if not m:
        return ""
    return m.group(1).strip()


def parse_selector_output_to_doc_indices(
    predict_str: str,
    num_candidates: int,
    default_k: int = 5
) -> List[int]:
    answer_content = extract_selector_answer(predict_str)

    selected_indices: List[int] = []
    nums = re.findall(r"\d+", answer_content)
    if nums:
        for x in nums:
            idx = int(x)
            if 1 <= idx <= num_candidates:
                selected_indices.append(idx - 1)

    if not selected_indices:
        k = min(default_k, num_candidates)
        selected_indices = list(range(k))

    seen = set()
    dedup = []
    for i in selected_indices:
        if i not in seen:
            dedup.append(i)
            seen.add(i)
    return dedup


# =================== Selector prompt ===================

def format_top_n_docs(ctxs: List[Dict[str, Any]], n: int) -> str:
    parts = []
    for i, ctx in enumerate(ctxs[:n], start=1):
        title = ctx.get("title", "")
        content = ctx.get("doc", "")
        parts.append(f"Doc [{i}]:\nTitle: {title}\nContent: {content}\n")
    return "\n".join(parts)


def build_selector_conversation(question: str, retrieved_content: str) -> List[Dict[str, str]]:
    system_msg = {
        "role": "system",
        "content": (
            "You are an expert evidence-set selector for RAG. Your goal is to select a SMALL set of "
            "documents that makes the question answerable, BUT NOT trivial. Prefer evidence sets that "
            "sit near the model's competence boundary: solvable with careful multi-step reasoning, "
            "yet not so direct that the answer is obvious from a single passage.\n\n"
            "Output format (STRICT):\n"
            "<think> Briefly explain which documents contain relevant clues and why this set is answerable "
            "but requires integration (2-4 sentences). </think>\n"
            "<answer> [doc_id1], [doc_id2], [doc_id3], [doc_id4], [doc_id5] </answer>\n\n"
            "Rules:\n"
            "- Select exactly 5 documents.\n"
            "- In <answer>, list ONLY the document identifiers in brackets, separated by commas.\n"
            "- Do NOT output anything outside <think> and <answer>.\n"
        )
    }
    user_msg = {"role": "user", "content": f"Query: {question}\n\n{retrieved_content}"}
    return [system_msg, user_msg]


# =================== Generator prompt (budgeted) ===================

def build_generator_prompt_prefix(question: str) -> str:
    return (
        "You are given a question and retrieved documents.\n"
        "You MUST answer the question using ONLY information from the retrieved documents.\n"
        "Even for yes/no questions, decide yes or no by reasoning from facts in the documents.\n\n"
        "Output format (STRICT):\n"
        "<think> ... </think>\n"
        "<answer> ... </answer>\n\n"
        "Evidence citation rule:\n"
        "- Whenever you use a piece of evidence from the documents in your reasoning, you MUST cite it inline as Doc [i].\n"
        "- Keep <think> concise (1-3 sentences).\n\n"
        "Answer rules:\n"
        "- If the question is yes/no, <answer> must be exactly one of: yes / no / unknown.\n"
        "- Otherwise, <answer> must be a short phrase.\n"
        "- If the documents do not provide enough information, output <answer> unknown </answer>.\n"
        "- Do NOT output anything outside <think> and <answer>.\n\n"
        "=== QUESTION ===\n"
        f"{question}\n"
        "=== END QUESTION ===\n\n"
        "=== RETRIEVED DOCUMENTS ===\n"
    )


def format_doc_block(i: int, title: str, content: str) -> str:
    return f"Doc [{i}]:\nTitle: {title}\nContent: {content}\n\n"


def count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def truncate_to_tokens(tokenizer, text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return text
    ids = ids[:max_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True)


def build_generator_prompt_with_budget(
    tokenizer,
    question: str,
    selected_ctxs: List[Dict[str, Any]],
    max_prompt_tokens: int,
    reserve_for_generation_tokens: int = 0,
) -> str:
    prefix = build_generator_prompt_prefix(question)
    suffix = "=== END DOCUMENTS ===\n"

    effective_budget = max_prompt_tokens - max(0, reserve_for_generation_tokens)
    if effective_budget <= 0:
        raise ValueError(f"effective_budget <= 0: {effective_budget}")

    prefix_tokens = count_tokens(tokenizer, prefix)
    suffix_tokens = count_tokens(tokenizer, suffix)
    if prefix_tokens + suffix_tokens >= effective_budget:
        raise ValueError(
            f"Prefix+suffix already exceed budget: {prefix_tokens}+{suffix_tokens} >= {effective_budget}"
        )

    remaining = effective_budget - prefix_tokens - suffix_tokens
    out_blocks: List[str] = []

    for i, ctx in enumerate(selected_ctxs, start=1):
        title = ctx.get("title", "")
        content = ctx.get("doc", "")

        header = format_doc_block(i, title, "")
        header_tokens = count_tokens(tokenizer, header)
        if header_tokens >= remaining:
            break

        max_content_tokens = remaining - header_tokens
        truncated_content = truncate_to_tokens(tokenizer, content, max_content_tokens)

        block = format_doc_block(i, title, truncated_content)
        block_tokens = count_tokens(tokenizer, block)

        if block_tokens > remaining:
            overflow = block_tokens - remaining
            new_max = max(0, max_content_tokens - overflow - 8)
            truncated_content = truncate_to_tokens(tokenizer, content, new_max)
            block = format_doc_block(i, title, truncated_content)
            block_tokens = count_tokens(tokenizer, block)
            if block_tokens > remaining:
                break

        out_blocks.append(block)
        remaining -= block_tokens
        if remaining <= 0:
            break

    return prefix + "".join(out_blocks) + suffix


# =================== Hard-but-solvable ===================

def compute_hbs_score(mean: float, var: float, sigma: float = 0.20, center: float = 0.5) -> float:
    if sigma is None or sigma <= 0:
        return float(var)
    return float(var * np.exp(-((mean - center) ** 2) / (2.0 * (sigma ** 2))))


def pass_hbs_filter(
    acc_list: List[float],
    mean: float,
    var: float,
    mean_min: float,
    mean_max: float,
    var_min: float,
    require_cross_50: bool = False,
) -> bool:
    if not acc_list:
        return False
    if mean <= 0.0 or mean >= 1.0:
        return False
    if len(set(acc_list)) <= 1:
        return False
    if not (mean_min <= mean <= mean_max):
        return False
    if var < var_min:
        return False
    if require_cross_50:
        if not (min(acc_list) <= 0.5 and max(acc_list) >= 0.5):
            return False
    return True


# =================== vLLM worker processes ===================

def _selector_worker(
    in_q: Queue,
    out_q: Queue,
    model_path: str,
    tp_size: int,
    gpu_util: float,
    gpu_ids: str,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids

    from vllm import LLM, SamplingParams

    print(f"[Selector Worker] Initializing on GPUs: {gpu_ids}")
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tp_size,
        gpu_memory_utilization=gpu_util,
        trust_remote_code=True,
    )
    print("[Selector Worker] Ready")
    out_q.put("READY")

    while True:
        try:
            item = in_q.get(timeout=1)
        except Empty:
            continue

        if item is None:
            print("[Selector Worker] Shutting down...")
            break

        messages = item["messages"]
        sampling_kwargs = item["sampling"]

        sp = SamplingParams(
            temperature=sampling_kwargs.get("temperature", 0.7),
            top_p=sampling_kwargs.get("top_p", 1.0),
            max_tokens=sampling_kwargs.get("max_tokens", 1024),
        )

        try:
            outputs = llm.chat(messages=messages, sampling_params=sp, use_tqdm=False)
            results = [o.outputs[0].text.strip() for o in outputs]
            out_q.put({"status": "ok", "results": results})
        except Exception as e:
            out_q.put({"status": "error", "error": str(e)})


def _generator_worker(
    in_q: Queue,
    out_q: Queue,
    model_path: str,
    tp_size: int,
    gpu_util: float,
    gpu_ids: str,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids

    from vllm import LLM, SamplingParams

    print(f"[Generator Worker] Initializing on GPUs: {gpu_ids}")
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tp_size,
        gpu_memory_utilization=gpu_util,
        trust_remote_code=True,
    )
    print("[Generator Worker] Ready")
    out_q.put("READY")

    while True:
        try:
            item = in_q.get(timeout=1)
        except Empty:
            continue

        if item is None:
            print("[Generator Worker] Shutting down...")
            break

        messages = item["messages"]
        sampling_kwargs = item["sampling"]

        sp = SamplingParams(
            temperature=sampling_kwargs.get("temperature", 0.7),
            top_p=sampling_kwargs.get("top_p", 0.95),
            max_tokens=sampling_kwargs.get("max_tokens", 512),
            n=sampling_kwargs.get("n", 1),
        )

        try:
            outputs = llm.chat(messages=messages, sampling_params=sp, use_tqdm=False)
            results = [[x.text for x in o.outputs] for o in outputs]
            out_q.put({"status": "ok", "results": results})
        except Exception as e:
            out_q.put({"status": "error", "error": str(e)})


# =================== vLLM client wrapper ===================

class VLLMClient:
    def __init__(
        self,
        worker_fn,
        model_path: str,
        tp_size: int,
        gpu_util: float,
        gpu_ids: str,
        timeout: int = 600,
    ):
        self.in_q = mp.Queue(maxsize=4)
        self.out_q = mp.Queue(maxsize=4)
        self.timeout = timeout

        self.proc = mp.Process(
            target=worker_fn,
            args=(self.in_q, self.out_q, model_path, tp_size, gpu_util, gpu_ids),
        )
        self.proc.start()

        ready_signal = self.out_q.get(timeout=timeout)
        assert ready_signal == "READY", f"Worker failed to start: {ready_signal}"

    def chat(self, messages: List[List[Dict[str, str]]], **sampling_kwargs):
        self.in_q.put({"messages": messages, "sampling": sampling_kwargs})
        response = self.out_q.get(timeout=self.timeout)

        if response["status"] == "error":
            raise RuntimeError(f"Worker error: {response['error']}")

        return response["results"]

    def close(self):
        try:
            self.in_q.put(None)
            self.proc.join(timeout=30)
        except Exception as e:
            print(f"Warning: failed to close worker gracefully: {e}")
            self.proc.terminate()
        finally:
            self.in_q.close()
            self.out_q.close()


# =================== Sample preparation ===================

def prepare_samples(samples: List[Dict[str, Any]]) -> List[PreparedSample]:
    prepared = []
    for sample in samples:
        extra_info_raw = sample.get("extra_info", "{}")
        if isinstance(extra_info_raw, str):
            extra_info = json.loads(extra_info_raw)
        else:
            extra_info = extra_info_raw

        sample_id = extra_info.get("id", "unknown")
        question = extra_info.get("question", "")
        golden_answers = extra_info.get("golden_answers", [])
        ctxs = extra_info.get("ctxs", [])
        num_candidate_docs = len(ctxs)

        retrieved_content = format_top_n_docs(ctxs, num_candidate_docs)
        selector_conv = build_selector_conversation(question, retrieved_content)

        prepared.append(
            PreparedSample(
                sample_id=sample_id,
                question=question,
                golden_answers=golden_answers,
                ctxs=ctxs,
                num_candidate_docs=num_candidate_docs,
                selector_conv=selector_conv,
            )
        )
    return prepared


# =================== Batch processing ===================

def process_batch_samples(
    samples: List[Dict[str, Any]],
    selector_client: VLLMClient,
    generator_client: VLLMClient,
    tokenizer,
    max_prompt_tokens: int,
    reserve_for_generation_tokens: int,
    selector_rollouts: int,
    generator_rollouts: int,
    selector_temperature: float,
    generator_temperature: float,
    reward_threshold: float,
    acc_weight: float,
    target_cite_count: int,
    max_doc_id: int,
    mean_min: float,
    mean_max: float,
    var_min: float,
    hbs_sigma: float,
    require_cross_50: bool,
    variance_ddof: int = 0,
) -> List[SampleResult]:
    if not samples:
        return []

    prepared_samples = prepare_samples(samples)

    all_selector_convs = []
    sample_indices = []
    for idx, prep in enumerate(prepared_samples):
        for _ in range(selector_rollouts):
            all_selector_convs.append(prep.selector_conv)
            sample_indices.append(idx)

    all_selector_outputs = selector_client.chat(
        all_selector_convs,
        temperature=selector_temperature,
        top_p=0.95,
        max_tokens=256,
    )

    selector_outputs_by_sample: List[List[str]] = [[] for _ in prepared_samples]
    for i, output in enumerate(all_selector_outputs):
        sample_idx = sample_indices[i]
        selector_outputs_by_sample[sample_idx].append(output)

    generator_messages_batch: List[List[Dict[str, str]]] = []
    request_metadata = []

    for sample_idx, prep in enumerate(prepared_samples):
        for rollout_idx, selector_text in enumerate(selector_outputs_by_sample[sample_idx]):
            selector_think = extract_selector_think(selector_text)
            selector_answer = extract_selector_answer(selector_text)
            selected_indices = parse_selector_output_to_doc_indices(
                selector_text, prep.num_candidate_docs, default_k=5
            )
            selected_docs = [prep.ctxs[i] for i in selected_indices if 0 <= i < len(prep.ctxs)]

            prompt_text = build_generator_prompt_with_budget(
                tokenizer=tokenizer,
                question=prep.question,
                selected_ctxs=selected_docs,
                max_prompt_tokens=max_prompt_tokens,
                reserve_for_generation_tokens=reserve_for_generation_tokens,
            )
            generator_messages_batch.append([{"role": "user", "content": prompt_text}])
            request_metadata.append((sample_idx, rollout_idx, selector_text, selector_think, selector_answer, selected_indices))

    all_generator_outputs_nested = generator_client.chat(
        generator_messages_batch,
        temperature=generator_temperature,
        top_p=0.95,
        max_tokens=1024,
        n=generator_rollouts,
    )

    generator_outputs_by_sample: List[List[Tuple[str, str, str, List[int], List[str]]]] = [[] for _ in prepared_samples]
    for req_idx, (sample_idx, rollout_idx, selector_text, selector_think, selector_answer, selected_indices) in enumerate(request_metadata):
        gen_outputs = all_generator_outputs_nested[req_idx] if req_idx < len(all_generator_outputs_nested) else []
        generator_outputs_by_sample[sample_idx].append((selector_text, selector_think, selector_answer, selected_indices, gen_outputs))

    results: List[SampleResult] = []
    for sample_idx, prep in enumerate(prepared_samples):
        all_selector_rollouts: List[SelectorRollout] = []
        accuracy_list: List[float] = []

        for selector_text, selector_think, selector_answer, selected_indices, generator_outputs in generator_outputs_by_sample[sample_idx]:
            gen_rollouts: List[GeneratorRollout] = []
            num_correct = 0

            for gen_text in generator_outputs:
                reward_info = compute_generator_reward(
                    predict_str=gen_text,
                    golden_answers=prep.golden_answers,
                    max_doc_id=max_doc_id,
                    target_cite_count=target_cite_count,
                    acc_weight=acc_weight,
                )
                is_correct = reward_info["total_reward"] >= reward_threshold
                if is_correct:
                    num_correct += 1

                gen_rollouts.append(
                    GeneratorRollout(
                        raw_output=gen_text,
                        extracted_answer=reward_info["extracted_answer"],
                        total_reward=reward_info["total_reward"],
                        acc_reward=reward_info["acc_reward"],
                        cite_reward=reward_info["cite_reward"],
                        max_f1=reward_info["max_f1"],
                        max_em=reward_info["max_em"],
                        cite_count=reward_info["cite_count"],
                        has_format=reward_info["has_format"],
                        is_correct=is_correct,
                    )
                )

            accuracy = num_correct / len(generator_outputs) if generator_outputs else 0.0
            accuracy_list.append(float(accuracy))

            all_selector_rollouts.append(
                SelectorRollout(
                    selector_output=selector_text,
                    selector_think=selector_think,
                    selector_answer=selector_answer,
                    selected_doc_indices=selected_indices,
                    generator_rollouts=gen_rollouts,
                    num_correct=num_correct,
                    accuracy=float(accuracy),
                )
            )

        accuracy_array = np.array(accuracy_list, dtype=np.float32)
        if len(accuracy_array) > 0:
            accuracy_mean = float(np.mean(accuracy_array))
            accuracy_std = float(np.std(accuracy_array, ddof=variance_ddof))
            accuracy_variance = float(np.var(accuracy_array, ddof=variance_ddof))
        else:
            accuracy_mean, accuracy_std, accuracy_variance = 0.0, 0.0, 0.0

        hbs_score = compute_hbs_score(accuracy_mean, accuracy_variance, sigma=hbs_sigma, center=0.5)
        keep_for_grpo = pass_hbs_filter(
            acc_list=accuracy_list,
            mean=accuracy_mean,
            var=accuracy_variance,
            mean_min=mean_min,
            mean_max=mean_max,
            var_min=var_min,
            require_cross_50=require_cross_50,
        )

        results.append(
            SampleResult(
                sample_id=prep.sample_id,
                question=prep.question,
                golden_answers=prep.golden_answers,
                ctxs=prep.ctxs,
                num_candidate_docs=prep.num_candidate_docs,
                selector_rollouts=all_selector_rollouts,
                accuracy_list=accuracy_list,
                accuracy_mean=accuracy_mean,
                accuracy_std=accuracy_std,
                accuracy_variance=accuracy_variance,
                hbs_score=hbs_score,
                keep_for_grpo=keep_for_grpo,
            )
        )

    return results


# =================== Main ===================

def main():
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(description="Hard-but-solvable filtering (selector variance)")

    parser.add_argument("--selector_model", type=str, required=True, help="Selector model path")
    parser.add_argument("--selector_tensor_parallel_size", type=int, default=4, help="Selector vLLM TP")
    parser.add_argument("--selector_gpu_memory_utilization", type=float, default=0.9, help="Selector vLLM mem util")
    parser.add_argument("--selector_gpus", type=str, default="0,1,2,3", help="Selector GPU IDs")

    parser.add_argument("--generator_model", type=str, required=True, help="Generator model path")
    parser.add_argument("--generator_tensor_parallel_size", type=int, default=4, help="Generator vLLM TP")
    parser.add_argument("--generator_gpu_memory_utilization", type=float, default=0.9, help="Generator vLLM mem util")
    parser.add_argument("--generator_gpus", type=str, default="4,5,6,7", help="Generator GPU IDs")

    parser.add_argument(
        "--generator_tokenizer_path",
        type=str,
        default=None,
        help="HF tokenizer path for generator prompt budget (default = generator_model)",
    )
    parser.add_argument("--max_prompt_tokens", type=int, default=2048, help="Max prompt tokens (input side)")
    parser.add_argument("--reserve_for_generation_tokens", type=int, default=0, help="Reserve tokens inside budget")

    parser.add_argument("--train_data_path", type=str, required=True, help="Training data path")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to process (-1 for all)")
    parser.add_argument("--start_idx", type=int, default=0, help="Start index")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size in samples")

    parser.add_argument("--selector_rollouts", type=int, default=8, help="Selector rollouts (N)")
    parser.add_argument("--generator_rollouts", type=int, default=8, help="Generator rollouts (K)")
    parser.add_argument("--selector_temperature", type=float, default=0.7, help="Selector sampling temperature")
    parser.add_argument("--generator_temperature", type=float, default=0.7, help="Generator sampling temperature")

    parser.add_argument("--reward_threshold", type=float, default=0.5, help="Correctness threshold on total reward")
    parser.add_argument("--acc_weight", type=float, default=0.8, help="Weight for acc_reward")
    parser.add_argument("--target_cite_count", type=int, default=2, help="Target citation count in <think>")
    parser.add_argument("--max_doc_id", type=int, default=31, help="Max document id considered in citation counting")

    parser.add_argument("--variance_ddof", type=int, default=0, help="ddof for np.var/np.std")

    parser.add_argument("--output_dir", type=str, default="filtered_data", help="Output directory")

    parser.add_argument("--enable_filter", action="store_true", help="Write filtered subset")
    parser.add_argument("--mean_min", type=float, default=0.25)
    parser.add_argument("--mean_max", type=float, default=0.85)
    parser.add_argument("--var_min", type=float, default=0.02)
    parser.add_argument("--hbs_sigma", type=float, default=0.20)
    parser.add_argument("--hbs_center", type=float, default=0.5)
    parser.add_argument("--require_cross_50", action="store_true")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"Loading data from {args.train_data_path}...")
    if args.train_data_path.endswith(".parquet"):
        dataset = load_dataset("parquet", data_files=args.train_data_path, split="train")
    else:
        dataset = load_dataset("json", data_files=args.train_data_path, split="train")

    end_idx = len(dataset) if args.num_samples == -1 else min(args.start_idx + args.num_samples, len(dataset))
    samples_to_process = list(range(args.start_idx, end_idx))
    print(f"Loaded {len(dataset)} samples; processing {len(samples_to_process)} samples")

    tok_path = args.generator_tokenizer_path or args.generator_model
    print(f"Loading generator tokenizer from: {tok_path}")
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True, use_fast=True)

    print("Starting dual vLLM workers...")
    selector_client = VLLMClient(
        worker_fn=_selector_worker,
        model_path=args.selector_model,
        tp_size=args.selector_tensor_parallel_size,
        gpu_util=args.selector_gpu_memory_utilization,
        gpu_ids=args.selector_gpus,
        timeout=600,
    )
    generator_client = VLLMClient(
        worker_fn=_generator_worker,
        model_path=args.generator_model,
        tp_size=args.generator_tensor_parallel_size,
        gpu_util=args.generator_gpu_memory_utilization,
        gpu_ids=args.generator_gpus,
        timeout=600,
    )

    filtered_jsonl_path = os.path.join(args.output_dir, f"filtered_{timestamp}.jsonl")
    filtered_ids_path = os.path.join(args.output_dir, f"filtered_ids_{timestamp}.json")

    filtered_ids: List[str] = []
    kept_count = 0

    ffiltered = open(filtered_jsonl_path, "w", encoding="utf-8") if args.enable_filter else None

    num_batches = (len(samples_to_process) + args.batch_size - 1) // args.batch_size

    try:
        for batch_idx in tqdm(range(num_batches), desc="Filtering batches"):
            batch_start = batch_idx * args.batch_size
            batch_end = min(batch_start + args.batch_size, len(samples_to_process))
            batch_indices = samples_to_process[batch_start:batch_end]
            batch_samples = [dataset[idx] for idx in batch_indices]

            batch_results = process_batch_samples(
                samples=batch_samples,
                selector_client=selector_client,
                generator_client=generator_client,
                tokenizer=tokenizer,
                max_prompt_tokens=args.max_prompt_tokens,
                reserve_for_generation_tokens=args.reserve_for_generation_tokens,
                selector_rollouts=args.selector_rollouts,
                generator_rollouts=args.generator_rollouts,
                selector_temperature=args.selector_temperature,
                generator_temperature=args.generator_temperature,
                reward_threshold=args.reward_threshold,
                acc_weight=args.acc_weight,
                target_cite_count=args.target_cite_count,
                max_doc_id=args.max_doc_id,
                mean_min=args.mean_min,
                mean_max=args.mean_max,
                var_min=args.var_min,
                hbs_sigma=args.hbs_sigma,
                require_cross_50=args.require_cross_50,
                variance_ddof=args.variance_ddof,
            )

            for res in batch_results:
                if not res.keep_for_grpo:
                    continue

                kept_count += 1
                filtered_ids.append(res.sample_id)

                if ffiltered is not None:
                    minimal_record = {
                        "id": res.sample_id,
                        "question": res.question,
                        "golden_answers": res.golden_answers,
                        "ctxs": res.ctxs,
                    }
                    ffiltered.write(json.dumps(minimal_record, ensure_ascii=False) + "\n")

    finally:
        if ffiltered is not None:
            ffiltered.close()
        print("Shutting down workers...")
        selector_client.close()
        generator_client.close()

    if args.enable_filter:
        with open(filtered_ids_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "filter_config": {
                        "mean_min": args.mean_min,
                        "mean_max": args.mean_max,
                        "var_min": args.var_min,
                        "hbs_sigma": args.hbs_sigma,
                        "hbs_center": args.hbs_center,
                        "require_cross_50": args.require_cross_50,
                        "variance_ddof": args.variance_ddof,
                        "reward_threshold": args.reward_threshold,
                        "acc_weight": args.acc_weight,
                        "target_cite_count": args.target_cite_count,
                        "max_doc_id": args.max_doc_id,
                        "max_prompt_tokens": args.max_prompt_tokens,
                        "reserve_for_generation_tokens": args.reserve_for_generation_tokens,
                        "selector_rollouts": args.selector_rollouts,
                        "generator_rollouts": args.generator_rollouts,
                        "selector_temperature": args.selector_temperature,
                        "generator_temperature": args.generator_temperature,
                    },
                    "num_filtered": len(filtered_ids),
                    "filtered_ids": filtered_ids,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        print(f"Kept samples: {kept_count}")
        print(f"Filtered JSONL: {filtered_jsonl_path}")
        print(f"Filtered IDs:   {filtered_ids_path}")
    else:
        print(f"Kept samples: {kept_count} (enable_filter is False, no files written)")


if __name__ == "__main__":
    main()
