#!/bin/bash
#SBATCH -J sft_data_analysis
#SBATCH --error=./logs/%j.err
#SBATCH --output=./logs/%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40000
#SBATCH --time=24:00:00

ANALYSIS=$1
SOURCE_DATASET=$2

EXP_DATA_DIR=../data

source ../laser-env/bin/activate

if [[ $SOURCE_DATASET == "main" ]]; then
  DATASETS="alpaca_gpt4,open_math_instruct_2,flan_v2_90k,sharegpt_en,wizardlm_evol_instruct,colm25_200k_agentinst_random,ifeval_like_5k,ultrainteract_coding"
elif [[ $SOURCE_DATASET == "strong" ]]; then
  DATASETS="tulu_3_sft_mixture_0225"
else
  # source dataset (weak)
  DATASETS="alpaca_gpt4,flan_v2_90k,alpaca,daring_anteater,conifer_v1,numina_math_cot_v1,longform,dolly_15k"
fi

cd ../analysis

python -m run_analysis --dataset ${DATASETS} --analysis ${ANALYSIS} \
  --output_dir ${EXP_DATA_DIR}/analysis \
  --num_devices 1 --repeat_analysis \
  --request_batch_size 1024


