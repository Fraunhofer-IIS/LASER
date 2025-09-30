#!/bin/bash

module load release/24.10 GCCcore/13.3.0 CUDA/12.6.0
module load Python/3.12.3

export TRL_DIR=./trl/
export TRL_ENV=./trl-env/
export EXP_DATA_DIR=../../data/emnlp25/

source ${TRL_ENV}/bin/activate

N_GPUS=4
N_NODES=1

MODEL=$1
TOKENIZER=$2
OUTPUT=$3
RUN_NAME=$(basename ${OUTPUT})
DATASET=${EXP_DATA_DIR}/$4

if [[ $DATASET == *"_1k"* ]]; then
  BATCH_SIZE=8
  GRADIENT_ACCUMULATION=16
  EPOCHS=4
  LEARNING_RATE=5.0e-06
else
  if [[ $RUN_NAME == *"falcon"* ]]; then
    BATCH_SIZE=4
    GRADIENT_ACCUMULATION=16
    EPOCHS=2
    LEARNING_RATE=2.0e-05
    WARMUP_RATIO=0.03
    WEIGHT_DECAY=0.1
  elif [[ $RUN_NAME == *"llama"* ]]; then
    BATCH_SIZE=8
    GRADIENT_ACCUMULATION=8
    EPOCHS=2
    LEARNING_RATE=5.0e-06
    WARMUP_RATIO=0.03
    WEIGHT_DECAY=0.01
  elif [[ $RUN_NAME == *"mistral"* ]]; then
    BATCH_SIZE=8
    GRADIENT_ACCUMULATION=16
    EPOCHS=2
    LEARNING_RATE=5.0e-06
    WARMUP_RATIO=0.1
    WEIGHT_DECAY=0.01
  elif [[ $RUN_NAME == *"qwen"* ]]; then
    BATCH_SIZE=8
    GRADIENT_ACCUMULATION=8
    EPOCHS=2
    LEARNING_RATE=2.0e-05
    WARMUP_RATIO=0.03
    WEIGHT_DECAY=0.01
  elif [[ $RUN_NAME == *"smollm"* ]]; then
    BATCH_SIZE=32
    GRADIENT_ACCUMULATION=16
    EPOCHS=3
    LEARNING_RATE=1.0e-04
    WARMUP_RATIO=0.03
    WEIGHT_DECAY=0.01
  else
    BATCH_SIZE=8
    GRADIENT_ACCUMULATION=16
    EPOCHS=2
    LEARNING_RATE=5.0e-06
    WARMUP_RATIO=0.1
    WEIGHT_DECAY=0.01
  fi
fi

export WANDB_PROJECT='trl_sft'

accelerate launch --config_file ${TRL_DIR}/examples/accelerate_configs/deepspeed_zero2.yaml \
                --num_processes ${N_GPUS} --num_machines ${N_NODES} \
                ${TRL_DIR}/scripts/sft.py \
                --dataset_name ${DATASET}  \
                --model_name_or_path ${MODEL} \
                --attn_implementation flash_attention_2 \
                --trust_remote_code \
                --num_train_epochs ${EPOCHS}  \
                --learning_rate ${LEARNING_RATE}  \
                --lr_scheduler_type cosine  \
                --warmup_ratio ${WARMUP_RATIO} \
                --max_length 4096 \
                --packing False \
                --packing_strategy bfd \
                --padding_free True \
                --completion_only_loss True \
                --per_device_train_batch_size ${BATCH_SIZE} \
                --per_device_eval_batch_size ${BATCH_SIZE} \
                --gradient_accumulation_steps ${GRADIENT_ACCUMULATION}  \
                --gradient_checkpointing \
                --gradient_checkpointing_kwargs='{"use_reentrant": false}' \
                --logging_steps 1  \
                --save_strategy steps \
                --eval_strategy steps \
                --eval_steps 20 \
                --output_dir ${OUTPUT}  \
                --overwrite_output_dir \
                --bf16 True \
                --tf32 True \
                --weight_decay ${WEIGHT_DECAY} \
                --logging_first_step \
                --report_to wandb \
                --run_name ${RUN_NAME} \
                --neftune_noise_alpha=5 \
                --dataset_name_eval ${EVAL_DATASET}  \
                --use_liger \
                --save_only_model \
                --save_total_limit 1 \



