#!/bin/bash
#
export HF_HOME=.cache/huggingface
export HF_DATASETS_CACHE=.cache/huggingface/datasets

SELECTOR_MODEL="/Your-path/SELECTOR-model"
GENERATOR_MODEL="/Your-path/Generator-model"

TRAIN_DATA_PATH="data/train.parquet"


OUTPUT_DIR="./filtered_data"

SELECTOR_GPUS="0,1,2,3"
GENERATOR_GPUS="4,5,6,7"

python filter.py \
    --selector_model ${SELECTOR_MODEL} \
    --generator_model ${GENERATOR_MODEL} \
    --train_data_path ${TRAIN_DATA_PATH} \
    --output_dir ${OUTPUT_DIR} \
    \
    --selector_gpus "${SELECTOR_GPUS}" \
    --generator_gpus "${GENERATOR_GPUS}" \
    --selector_tensor_parallel_size 4 \
    --generator_tensor_parallel_size 4 \
    --selector_gpu_memory_utilization 0.9 \
    --generator_gpu_memory_utilization 0.9 \
    \
    --num_samples -1 \
    --start_idx 0 \
    --batch_size 8 \
    \
    --selector_rollouts 8 \
    --generator_rollouts 10 \
    --selector_temperature 1.0 \
    --generator_temperature 0.7 \
    \
    --max_prompt_tokens 2048 \
    --reserve_for_generation_tokens 0 \
    \
    --reward_threshold 0.8 \
    --acc_weight 0.8 \
    --target_cite_count 2 \
    --max_doc_id 31 \
    \
    --enable_filter \
    --mean_min 0.25 \
    --mean_max 0.85 \
    --var_min 0.02 \
    --hbs_sigma 0.20 \
    --variance_ddof 0