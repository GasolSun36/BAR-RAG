export HF_HOME=.cache/huggingface
export HF_DATASETS_CACHE=.cache/huggingface/datasets


export CUDA_VISIBLE_DEVICES=0,1,2,3
python examples/candidate_generation.py \
    --input_path data/selector_train_fix.parquet \
    --local_save_dir data/generator_train \
    --split_name train \
    --model_name "/Your-path/SELECTOR-model" \
    --selector_top_n 25 \
    --max_selector_input_tokens 4096 \
    --max_new_tokens_selector 1024 \
    --tensor_parallel_size 4 \
    --batch_size_selector 128 \
    --gpu_memory_utilization 0.9