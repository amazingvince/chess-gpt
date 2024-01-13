#!/bin/bash
# fine-tune a causal language model

# configure environment variables
export TOKENIZERS_PARALLELISM=true
export WANDB_WATCH="gradients"
export WANDB_PROJECT="chess"
echo "wandb watching: $WANDB_WATCH"

# script arguments
cpu_cores=$(nproc)
NUM_WORKERS=$((cpu_cores - 4))
NUM_WORKERS=$(( NUM_WORKERS > 48 ? 48 : NUM_WORKERS ))
NUM_WORKERS=$(( NUM_WORKERS < 1 ? 1 : NUM_WORKERS ))
hf_model_tag="chess-llama-mini-v3"
MAX_SOURCE_LEN=1024


DATASET_TAG="laion/strategic_game_chess"
# DATASET_CFG="Lichess-stockfish mix"

DS_SHORT_NAME="$(basename $DATASET_TAG)"
SHORT_NAME="$(basename $hf_model_tag)"

RUN_NAME="$SHORT_NAME-${MAX_SOURCE_LEN}"
RUNTIME_DIR="runtime/autoregressive/$RUN_NAME"
LOGGING_DIR="$RUNTIME_DIR/logs"
RUN_SEED=$RANDOM

TOKENIZER_NAME="/home/vince/chess_gpt_2/chess-llama"

NUM_EPOCHS=1
LEARNING_RATE=4e-4
WARMUP_RATIO=100
BATCH_SIZE=8
EVAL_BATCH_SIZE=8
WEIGHT_DECAY=0.1

# optimizer
OPTIMIZER_ID="adamw_torch_fused" # adamw_8bit
GC_STEPS=8
OPTIM_BETA1=0.90
OPTIM_BETA2=0.95
LR_SCHEDULER_TYPE="constant"
GRAD_CHKPTING=True
MAX_GRAD_NORM=1.0

# checkpointing and logging
CHK_STEPS=1000
SAVE_STRATEGY="steps"
SAVE_LIMIT=1
LOGGING_STEPS=5
REPORT_TO="wandb"


DATA_TYPE="--bf16 --bf16_full_eval True"

USE_TF32=True

# eval
EVAL_STRATEGY=no #epoch, steps, no
EVAL_STEPS=200
MAX_EVAL_SAMPLES=2048

DEEPSPEED_CONFIG="deepspeed.json"

# runtime directory
mkdir -p $RUNTIME_DIR
echo "runtime directory: $RUNTIME_DIR"

pip install -U -q wandb

#     --block_size $MAX_SOURCE_LEN \
# --config_name
# --model_name_or_path
# python run_clm.py \
# --dataset_config_name "$DATASET_CFG" \
# NCCL_P2P_DISABLE=0 NCCL_IB_DISABLE=0
ACCELERATE_LOG_LEVEL=info accelerate launch --num_processes 1 run_clm.py \
    --dataset_name "$DATASET_TAG" \
    --tokenizer_name "$TOKENIZER_NAME" \
    --do_train \
    --streaming True \
    --config_name "chess-llama/config.json" \
    --num_train_epochs $NUM_EPOCHS \
    --save_strategy $SAVE_STRATEGY \
    --evaluation_strategy $EVAL_STRATEGY \
    --block_size 1024 \
    --data_seed $RANDOM \
    --dataloader_num_workers $NUM_WORKERS \
    --dataloader_pin_memory True \
    --eval_steps $EVAL_STEPS \
    --gradient_accumulation_steps $GC_STEPS \
    --gradient_checkpointing $GRAD_CHKPTING \
    --hub_model_id $RUN_NAME \
    --hub_private_repo True \
    --learning_rate $LEARNING_RATE \
    --logging_steps $LOGGING_STEPS \
    --logging_dir $LOGGING_DIR \
    --lr_scheduler_type $LR_SCHEDULER_TYPE \
    --max_grad_norm $MAX_GRAD_NORM \
    --adam_beta1 $OPTIM_BETA1 --adam_beta2 $OPTIM_BETA2 \
    --adam_epsilon 1e-6 \
    --optim $OPTIMIZER_ID \
    --output_dir $RUNTIME_DIR \
    --overwrite_output_dir \
    --per_device_eval_batch_size $EVAL_BATCH_SIZE \
    --per_device_train_batch_size $BATCH_SIZE \
    --preprocessing_num_workers $NUM_WORKERS \
    --report_to $REPORT_TO \
    --run_name "$RUN_NAME" \
    --save_steps $CHK_STEPS \
    --save_total_limit $SAVE_LIMIT \
    --seed $RUN_SEED \
    --tf32 $USE_TF32 \
    --warmup_steps $WARMUP_RATIO \
    --weight_decay $WEIGHT_DECAY \
    --push_to_hub --save_safetensors \
    $DATA_TYPE \
    --max_eval_samples $MAX_EVAL_SAMPLES \
    --save_total_limit 1 \
    --low_cpu_mem_usage True \
    --auto_find_batch_size True \
    --block_size $MAX_SOURCE_LEN \
    --use_fast_tokenizer True \
    --trust_remote_code True  \
    --max_steps 10000000 \
    # --torch_compile_backend inductor \
    # --streaming True 

#    --max_steps 1000
#    --deepspeed $DEEPSPEED_CONFIG \

# --resume_from_checkpoint $CHECKPOINT_PATH
#    --max_steps 125 \