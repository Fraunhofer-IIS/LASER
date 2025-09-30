#!/bin/bash
#SBATCH -J finetuning
#SBATCH --error=./logs/%j.err
#SBATCH --output=./logs/%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=250000
#SBATCH --time=24:00:00

DATASET_NAME=$2
if [[ $1 == "falcon" ]]; then
  MODEL=tiiuae/Falcon3-10B-Base
  TOKENIZER=tiiuae/Falcon3-10B-Instruct
  MODEL_PREFIX=falcon10B-v3
elif [[ $1 == "mistral" ]]; then
  MODEL=mistralai/Mistral-7B-v0.3
  TOKENIZER=mistralai/Mistral-7B-Instruct-v0.3
  MODEL_PREFIX=mistral7B-v0.3
elif [[ $1 == "llama" ]]; then
  MODEL=meta-llama/Llama-3.1-8B
  TOKENIZER=meta-llama/Llama-3.1-8B-Instruct
  MODEL_PREFIX=llama8B-v3.1
elif [[ $1 == "qwen" ]]; then
  MODEL=Qwen/Qwen2.5-3B
  TOKENIZER=Qwen/Qwen2.5-3B-Instruct
  MODEL_PREFIX=qwen3B-v2.5
elif [[ $1 == "smollm" ]]; then
  MODEL=HuggingFaceTB/SmolLM2-1.7B
  TOKENIZER=HuggingFaceTB/SmolLM2-1.7B-Instruct
  MODEL_PREFIX=smol1.7B-v2
else
  MODEL=mistralai/Mistral-7B-v0.3
  TOKENIZER=mistralai/Mistral-7B-Instruct-v0.3
  MODEL_PREFIX=mistral7B-v0.3
fi

RUN_NAME=${MODEL_PREFIX}_${DATASET_NAME}
OUTPUT=./sft_models_emnlp25/${RUN_NAME}

cd trl_finetuning/

bash finetuning.sh ${MODEL} ${TOKENIZER} ${OUTPUT} ${DATASET_NAME}.jsonl
