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

python finetuning_data_mixer.py --config emnlp25/all.yaml --train_split -1.0 \
   --analysis_dir ${EXP_DATA_DIR}/analysis \
   --output_dir ${EXP_DATA_DIR}

python finetuning_data_mixer.py --config emnlp25/100k_random.yaml --train_split -1.0 \
   --analysis_dir ${EXP_DATA_DIR}/analysis \
   --output_dir ${EXP_DATA_DIR}

python finetuning_data_mixer.py --config emnlp25/100k_longest.yaml --train_split -1.0 \
   --analysis_dir ${EXP_DATA_DIR}/analysis \
   --output_dir ${EXP_DATA_DIR}

# This will take around 10 hours!!
python finetuning_data_mixer.py --config emnlp25/100k_deita.yaml --train_split -1.0 \
   --analysis_dir ${EXP_DATA_DIR}/analysis \
   --output_dir ${EXP_DATA_DIR}

# LASER (full pipeline)
python finetuning_data_mixer.py --config emnlp25/100k_clustering_cat_combination_v5.yaml --train_split -1.0 \
   --analysis_dir ${EXP_DATA_DIR}/analysis \
   --output_dir ${EXP_DATA_DIR}

# LASER (ablation components)
python finetuning_data_mixer.py --config emnlp25/ablation_components/100k_deita_scoring.yaml --train_split -1.0 \
    --analysis_dir ${EXP_DATA_DIR}/analysis \
    --output_dir ${EXP_DATA_DIR}

python finetuning_data_mixer.py --config emnlp25/ablation_components/100k_dedicated_quality_v2.yaml --train_split -1.0 \
    --analysis_dir ${EXP_DATA_DIR}/analysis \
    --output_dir ${EXP_DATA_DIR}

python finetuning_data_mixer.py --config emnlp25/ablation_components/100k_difficulty_v2.yaml --train_split -1.0 \
    --analysis_dir ${EXP_DATA_DIR}/analysis \
    --output_dir ${EXP_DATA_DIR}

python finetuning_data_mixer.py --config emnlp25/ablation_components/100k_combination_v5.yaml --train_split -1.0 \
    --analysis_dir ${EXP_DATA_DIR}/analysis \
    --output_dir ${EXP_DATA_DIR}

python finetuning_data_mixer.py --config emnlp25/ablation_components/100k_combination_v5_cat.yaml --train_split -1.0 \
    --analysis_dir ${EXP_DATA_DIR}/analysis \
    --output_dir ${EXP_DATA_DIR}

python finetuning_data_mixer.py --config emnlp25/ablation_components/25k_combination_v5.yaml --train_split -1.0 \
    --analysis_dir ${EXP_DATA_DIR}/analysis \
    --output_dir ${EXP_DATA_DIR}

python finetuning_data_mixer.py --config emnlp25/ablation_components/25k_dedicated_quality_v2.yaml --train_split -1.0 \
    --analysis_dir ${EXP_DATA_DIR}/analysis \
    --output_dir ${EXP_DATA_DIR}

python finetuning_data_mixer.py --config emnlp25/ablation_components/25k_deita_quality.yaml --train_split -1.0 \
    --analysis_dir ${EXP_DATA_DIR}/analysis \
    --output_dir ${EXP_DATA_DIR}

python finetuning_data_mixer.py --config emnlp25/ablation_components/25k_deita_quality_code_only.yaml --train_split -1.0 \
    --analysis_dir ${EXP_DATA_DIR}/analysis \
    --output_dir ${EXP_DATA_DIR}

python finetuning_data_mixer.py --config emnlp25/ablation_components/25k_deita_quality_fqa_extr_reas_only.yaml --train_split -1.0 \
    --analysis_dir ${EXP_DATA_DIR}/analysis \
    --output_dir ${EXP_DATA_DIR}

python finetuning_data_mixer.py --config emnlp25/ablation_components/25k_deita_quality_gen_brain_only.yaml --train_split -1.0 \
    --analysis_dir ${EXP_DATA_DIR}/analysis \
    --output_dir ${EXP_DATA_DIR}

python finetuning_data_mixer.py --config emnlp25/ablation_components/25k_deita_quality_math_only.yaml --train_split -1.0 \
    --analysis_dir ${EXP_DATA_DIR}/analysis \
    --output_dir ${EXP_DATA_DIR}

# LASER (source quality)
python finetuning_data_mixer.py --config emnlp25/source_quality/high_quality_all_dedicated_quality_v2.yaml --train_split -1.0 \
    --analysis_dir ${EXP_DATA_DIR}/analysis \
    --output_dir ${EXP_DATA_DIR}

python finetuning_data_mixer.py --config emnlp25/source_quality/high_quality_all_difficulty_v2.yaml --train_split -1.0 \
    --analysis_dir ${EXP_DATA_DIR}/analysis \
    --output_dir ${EXP_DATA_DIR}

python finetuning_data_mixer.py --config emnlp25/source_quality/low_quality_all_dedicated_quality_v2.yaml --train_split -1.0 \
    --analysis_dir ${EXP_DATA_DIR}/analysis \
    --output_dir ${EXP_DATA_DIR}

python finetuning_data_mixer.py --config emnlp25/source_quality/low_quality_all_difficulty_v2.yaml --train_split -1.0 \
    --analysis_dir ${EXP_DATA_DIR}/analysis \
    --output_dir ${EXP_DATA_DIR}    

