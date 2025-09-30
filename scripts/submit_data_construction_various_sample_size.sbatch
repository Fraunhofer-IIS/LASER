#!/bin/bash
#SBATCH -J sft_data_construction
#SBATCH --error=./logs/%j.err
#SBATCH --output=./logs/%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180000
#SBATCH --time=24:00:00

module load release/25.06 GCCcore/13.3.0
module load Python/3.12.3

export DATA_ENV=./setfit-env/
source ${DATA_ENV}/bin/activate

EXP_DATA_DIR=../../data/emnlp25/

cd ../dataset_mixer

# 50k
python finetuning_data_mixer.py --config emnlp25/ablation_sample_size/50k_random.yaml --train_split -1.0 \
   --analysis_dir ${EXP_DATA_DIR}/analysis \
   --output_dir ${EXP_DATA_DIR}

python finetuning_data_mixer.py --config emnlp25/ablation_sample_size/50k_longest.yaml --train_split -1.0 \
   --analysis_dir ${EXP_DATA_DIR}/analysis \
   --output_dir ${EXP_DATA_DIR}

python finetuning_data_mixer.py --config emnlp25/ablation_sample_size/50k_deita.yaml --train_split -1.0 \
    --analysis_dir ${EXP_DATA_DIR}/analysis \
    --output_dir ${EXP_DATA_DIR}

python finetuning_data_mixer.py --config emnlp25/ablation_sample_size/50k_clustering_cat_combination_v5.yaml --train_split -1.0 \
   --analysis_dir ${EXP_DATA_DIR}/analysis \
   --output_dir ${EXP_DATA_DIR}

# 20k
python finetuning_data_mixer.py --config emnlp25/ablation_sample_size/25k_random.yaml --train_split -1.0 \
   --analysis_dir ${EXP_DATA_DIR}/analysis \
   --output_dir ${EXP_DATA_DIR}

python finetuning_data_mixer.py --config emnlp25/ablation_sample_size/25k_longest.yaml --train_split -1.0 \
   --analysis_dir ${EXP_DATA_DIR}/analysis \
   --output_dir ${EXP_DATA_DIR}

python finetuning_data_mixer.py --config emnlp25/ablation_sample_size/25k_deita.yaml --train_split -1.0 \
    --analysis_dir ${EXP_DATA_DIR}/analysis \
    --output_dir ${EXP_DATA_DIR}

python finetuning_data_mixer.py --config emnlp25/ablation_sample_size/25k_clustering_cat_combination_v5.yaml --train_split -1.0 \
   --analysis_dir ${EXP_DATA_DIR}/analysis \
   --output_dir ${EXP_DATA_DIR}

# 10k
python finetuning_data_mixer.py --config emnlp25/ablation_sample_size/10k_random.yaml --train_split -1.0 \
   --analysis_dir ${EXP_DATA_DIR}/analysis \
   --output_dir ${EXP_DATA_DIR}

python finetuning_data_mixer.py --config emnlp25/ablation_sample_size/10k_longest.yaml --train_split -1.0 \
   --analysis_dir ${EXP_DATA_DIR}/analysis \
   --output_dir ${EXP_DATA_DIR}

python finetuning_data_mixer.py --config emnlp25/ablation_sample_size/10k_deita.yaml --train_split -1.0 \
    --analysis_dir ${EXP_DATA_DIR}/analysis \
    --output_dir ${EXP_DATA_DIR}

python finetuning_data_mixer.py --config emnlp25/ablation_sample_size/10k_clustering_cat_combination_v5.yaml --train_split -1.0 \
   --analysis_dir ${EXP_DATA_DIR}/analysis \
   --output_dir ${EXP_DATA_DIR}

# 5k
python finetuning_data_mixer.py --config emnlp25/ablation_sample_size/5k_random.yaml --train_split -1.0 \
   --analysis_dir ${EXP_DATA_DIR}/analysis \
   --output_dir ${EXP_DATA_DIR}

python finetuning_data_mixer.py --config emnlp25/ablation_sample_size/5k_longest.yaml --train_split -1.0 \
   --analysis_dir ${EXP_DATA_DIR}/analysis \
   --output_dir ${EXP_DATA_DIR}

python finetuning_data_mixer.py --config emnlp25/ablation_sample_size/5k_deita.yaml --train_split -1.0 \
    --analysis_dir ${EXP_DATA_DIR}/analysis \
    --output_dir ${EXP_DATA_DIR}

python finetuning_data_mixer.py --config emnlp25/ablation_sample_size/5k_clustering_cat_combination_v5.yaml --train_split -1.0 \
   --analysis_dir ${EXP_DATA_DIR}/analysis \
   --output_dir ${EXP_DATA_DIR}

# 1k
python finetuning_data_mixer.py --config emnlp25/ablation_sample_size/1k_random.yaml --train_split -1.0 \
   --analysis_dir ${EXP_DATA_DIR}/analysis \
   --output_dir ${EXP_DATA_DIR}

python finetuning_data_mixer.py --config emnlp25/ablation_sample_size/1k_longest.yaml --train_split -1.0 \
   --analysis_dir ${EXP_DATA_DIR}/analysis \
   --output_dir ${EXP_DATA_DIR}

python finetuning_data_mixer.py --config emnlp25/ablation_sample_size/1k_deita.yaml --train_split -1.0 \
    --analysis_dir ${EXP_DATA_DIR}/analysis \
    --output_dir ${EXP_DATA_DIR}

python finetuning_data_mixer.py --config emnlp25/ablation_sample_size/1k_clustering_cat_combination_v5.yaml --train_split -1.0 \
   --analysis_dir ${EXP_DATA_DIR}/analysis \
   --output_dir ${EXP_DATA_DIR}