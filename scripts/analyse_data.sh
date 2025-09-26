#!/bin/bash

cd ../analysis

# source dataset main experiments
DATASETS="alpaca_gpt4,open_math_instruct_2,flan_v2_90k,sharegpt_en,wizardlm_evol_instruct,colm25_200k_agentinst_random,ifeval_like_5k,ultrainteract_coding"

python run_analysis --dataset ${DATASETS} --analysis tokens --request_batch_size 8
python run_analysis --dataset ${DATASETS} --analysis categories_v2 --request_batch_size 8
python run_analysis --dataset ${DATASETS} --analysis difficulty_v2 --request_batch_size 8 
python run_analysis --dataset ${DATASETS} --analysis quality --request_batch_size 8
python run_analysis --dataset ${DATASETS} --analysis if_quality --request_batch_size 8
python run_analysis --dataset ${DATASETS} --analysis code_quality --request_batch_size 8

# source dataset (strong)
DATASETS="tulu_3_sft_mixture_0225"

python run_analysis --dataset ${DATASETS} --analysis tokens --request_batch_size 8
python run_analysis --dataset ${DATASETS} --analysis categories_v2 --request_batch_size 8
python run_analysis --dataset ${DATASETS} --analysis difficulty_v2 --request_batch_size 8 
python run_analysis --dataset ${DATASETS} --analysis quality --request_batch_size 8
python run_analysis --dataset ${DATASETS} --analysis if_quality --request_batch_size 8
python run_analysis --dataset ${DATASETS} --analysis code_quality --request_batch_size 8

# source dataset (weak)
DATASETS="alpaca_gpt4,flan_v2_90k,alpaca,daring_anteater,conifer_v1,numina_math_cot_v1,longform,dolly_15k"

python run_analysis --dataset ${DATASETS} --analysis tokens --request_batch_size 8
python run_analysis --dataset ${DATASETS} --analysis categories_v2 --request_batch_size 8
python run_analysis --dataset ${DATASETS} --analysis difficulty_v2 --request_batch_size 8 
python run_analysis --dataset ${DATASETS} --analysis quality --request_batch_size 8
python run_analysis --dataset ${DATASETS} --analysis if_quality --request_batch_size 8
python run_analysis --dataset ${DATASETS} --analysis code_quality --request_batch_size 8