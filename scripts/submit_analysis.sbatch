#!/bin/bash
#SBATCH -J sft_data_analysis
#SBATCH --error=/home/pami666g/logs/%j.err
#SBATCH --output=/home/pami666g/logs/%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=40000
#SBATCH --time=24:00:00

module load release/24.10 GCCcore/13.3.0 CUDA/12.6.0
module load Python/3.12.3

ANALYSIS=$1
EXP_DATA_DIR=../../data/emnlp25/

if [[ $ANALYSIS == "categories_v2" ]]; then
  source ./setfit-env/bin/activate
else
  source ./analysis-env/bin/activate
fi

cd ../analysis

### analysis type
# categories_v2,if_quality,code_quality,process_reward_modelling,tagging,difficulty_v2,complexity,quality,tokens,embeddings,dataset_stats

### dataset
# lima,
# alpaca_gpt4,emnlp25_200k_agentinst_random,flan_v2_90k,ifeval_like_5k,open_math_instruct_2,sharegpt_en,ultrainteract_coding,wizardlm_evol_instruct,
# sigma_80k,dolly_15k,alpaca,conifer_v1,daring_anteater,evol_codealpaca_v1,longform,numina_math_cot_v1

python -m run_analysis --dataset lima --analysis ${ANALYSIS} \
  --output_dir ${EXP_DATA_DIR}/analysis \
  --num_devices 2 --repeat_analysis \
  --request_batch_size 1024


