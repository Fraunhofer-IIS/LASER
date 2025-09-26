#!/bin/bash

cd ../analysis

DATASETS="alpaca_gpt4,open_math_instruct_2,flan_v2_90k,sharegpt_en,wizardlm_evol_instruct,colm25_200k_agentinst_random,ifeval_like_5k,ultrainteract_coding"

python run_analysis --dataset ${DATASETS} --analysis tokens --request_batch_size 8
python run_analysis --dataset ${DATASETS} --analysis categories_v2 --request_batch_size 8
python run_analysis --dataset ${DATASETS} --analysis difficulty_v2 --request_batch_size 8 
python run_analysis --dataset ${DATASETS} --analysis quality --request_batch_size 8
python run_analysis --dataset ${DATASETS} --analysis if_quality --request_batch_size 8
python run_analysis --dataset ${DATASETS} --analysis code_quality --request_batch_size 8

