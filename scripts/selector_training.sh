#!/bin/bash

set -e

HOME=/YOUR-PATH

export VLLM_TORCH_COMPILE_LEVEL=0 
export TRITON_CACHE_DIR="/tmp/triton_cache_$$"
export TORCH_COMPILE_DISABLE=1

export GENERATOR_API_URL="http://localhost:8193/v1/chat/completions"
export GENERATOR_MODEL_NAME="/Your-path/Generator-model"
export GENERATOR_API_KEY=""


export K_ROLLOUTS="10"
export GENERATOR_TEMPERATURE="1.0"
export GENERATOR_TOP_P="1.0"
export GENERATOR_MAX_TOKENS="1024"

export USE_LLM_JUDGE="false"
export LLM_JUDGE_API_URL="http://localhost:8194/v1/chat/completions"
export LLM_JUDGE_MODEL_NAME="/Your-path/Judge-model"
export LLM_JUDGE_API_KEY=""


export GENERATOR_ACC_WEIGHT="0.8"
export GENERATOR_F1_WEIGHT="0.7"
export GENERATOR_EM_WEIGHT="0.3"
export GENERATOR_TARGET_CITE="2"
export GENERATOR_MAX_DOC_ID="25"
export GENERATOR_REWARD_THRESHOLD="0.8"

export REWARD_DEBUG_ENABLE="1"
export REWARD_DEBUG_DIR="debug_logs"
export REWARD_DEBUG_MAX_CHARS="2000"

export REWARD_TARGET_CENTER="0.6"
export REWARD_MAX_UNC="1.0"
export REWARD_MAX_REL="1.0"
export REWARD_LAMBDA_UNC="1.0"
export REWARD_LAMBDA_REL="0.2"
export REWARD_DEFAULT_K_DOCS="5"
export REWARD_TARGET_NUM_DOCS="5"
export REWARD_WRONG_COUNT_PENALTY="0.5"
export REWARD_MAX_COUNT_PENALTY="1.0"
export REWARD_FORMAT_BONUS="0.0"

export WANDB_API_KEY="YOUR-WANDB-API-KEY"


mkdir -p logs

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/selector_training_${TIMESTAMP}.log"

export GENERATOR_MAX_INPUT_TOKENS=2048
export GENERATOR_TOKENIZER_NAME_OR_PATH="/Your-path/Generator-model"

SELECTOR_MODEL="/Your-path/SELECTOR-model"
TRAIN_DATA="data/train.parquet"
VAL_DATA="data/val.parquet"

echo "=============================================="
echo "SELECTOR GRPO"
echo "=============================================="
echo "Generator API: $GENERATOR_API_URL"
echo "Generator Model: $GENERATOR_MODEL_NAME"
echo "K Rollouts: $K_ROLLOUTS"
echo "----------------------------------------------"
echo "Generator Reward Config:"
echo "  ACC_WEIGHT: $GENERATOR_ACC_WEIGHT (F1: $GENERATOR_F1_WEIGHT, EM: $GENERATOR_EM_WEIGHT)"
echo "  Target Cite: $GENERATOR_TARGET_CITE"
echo "  Reward Threshold: $GENERATOR_REWARD_THRESHOLD"
echo "----------------------------------------------"
echo "Use LLM Judge: $USE_LLM_JUDGE"
if [ "$USE_LLM_JUDGE" = "true" ]; then
    echo "LLM Judge API: $LLM_JUDGE_API_URL"
    echo "LLM Judge Model: $LLM_JUDGE_MODEL_NAME"
fi
echo "----------------------------------------------"
echo "Selector Reward Config:"
echo "  Target Center: $REWARD_TARGET_CENTER"
echo "  Lambda UNC: $REWARD_LAMBDA_UNC, Lambda REL: $REWARD_LAMBDA_REL"
echo "  Target Docs: $REWARD_TARGET_NUM_DOCS"
echo "  Format Bonus: $REWARD_FORMAT_BONUS"
echo "=============================================="
echo "LOG：$LOG_FILE"


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
nohup python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_DATA \
    data.val_files=$VAL_DATA \
    data.train_batch_size=8 \
    data.max_prompt_length=4096 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=False \
    data.truncation='right' \
    actor_rollout_ref.model.path=$SELECTOR_MODEL \
    actor_rollout_ref.actor.optim.lr=4e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.02 \
    actor_rollout_ref.actor.optim.lr_scheduler_type=cosine \
    actor_rollout_ref.actor.optim.min_lr_ratio=0.1 \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.model.use_shm=True \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=16 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
    actor_rollout_ref.actor.entropy_checkpointing=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=9216 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["wandb"]' \
    +ray_kwargs.ray_init.runtime_env.env_vars.WANDB_API_KEY=$WANDB_API_KEY \
    trainer.project_name='verl-selector-training' \
    trainer.experiment_name='selector-training' \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=500 \
    trainer.test_freq=10000 \
    trainer.total_epochs=1 \
    > "$LOG_FILE" 2>&1 &
