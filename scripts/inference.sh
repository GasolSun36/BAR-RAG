#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES="0,1,2,3"
export HF_HOME=.cache/huggingface
export HF_DATASETS_CACHE=.cache/huggingface/datasets

DATA_DIR="/Your-path/BAR-RAG/eval_data/eval_data_ctxs"
OUT_DIR="results"
MODEL_PATH="/Your-path/Generator-model"

mkdir -p "${OUT_DIR}"

DATASETS=(
  "nq.jsonl"
  "triviaqa.jsonl"
  "popqa.jsonl"
  "hotpotqa.jsonl"
  "2wikimQA.jsonl"
  "musique.jsonl"
  "bamboogle.jsonl"
)

TOPKS=5

for DATASET in "${DATASETS[@]}"; do
  NAME=$(basename "${DATASET}" .jsonl)
  OUT_PATH="${OUT_DIR}/${NAME}_topk${TOPK}_model.jsonl"

  python qa_inference.py \
    --model_path "${MODEL_PATH}" \
    --data_path "${DATA_DIR}/${DATASET}" \
    --output_path "${OUT_PATH}" \
    --top_k_ctx "${TOPK}" \
    --max_new_tokens 1024 \
    --temperature 0.3 \
    --top_p 1.0 \
    --batch_size 16 \
    --tensor_parallel_size 4 \
    --gpu_memory_utilization 0.9 \
    --max_model_len 16384 \
    --resume
done
