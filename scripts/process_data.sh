#!/bin/bash

python convert_to_parquet.py \
  --input_path "data/train.jsonl" \
  --split_name "train" \
  --top_n "25" \
  --local_save_dir "data" \
  --output_name "train.parquet"

python process_selector_data.py \
  --input_path data/train_rl.parquet \
  --output_path data/selector_train.parquet \
  --tokenizer_name_or_path "/Your-path/SELECTOR-model" \
  --max_input_tokens 4096 \
  --top_n 25

python fix_parquet.py \
  --input_file data/selector_train.parquet \
  --output data/selector_train_fix.parquet

