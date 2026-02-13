#!/bin/bash
set -e

echo "Evaluating EM & F1..."
python eval.py \
    --pred_file results/nq_topk5_model.jsonl
