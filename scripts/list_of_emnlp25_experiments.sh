#!/bin/bash

# Data analysis
SOURCE_DATASETS=(
    "main"
    "strong"
    "weak"
)
for SOURCE_DATASET in "${SOURCE_DATASETS[@]}"; do
    jobid=$(sbatch --parsable submit_analysis.sbatch categories_v2 ${SOURCE_DATASET})
    sbatch --dependency=afterok:${jobid} submit_analysis.sbatch if_quality ${SOURCE_DATASET}
    sbatch --dependency=afterok:${jobid} submit_analysis.sbatch code_quality ${SOURCE_DATASET}
    sbatch --dependency=afterok:${jobid} submit_analysis.sbatch process_reward_modelling ${SOURCE_DATASET}
    sbatch --dependency=afterok:${jobid} submit_analysis.sbatch tagging ${SOURCE_DATASET}
    sbatch --dependency=afterok:${jobid} submit_analysis.sbatch difficulty_v2 ${SOURCE_DATASET}
    sbatch --dependency=afterok:${jobid} submit_analysis.sbatch complexity ${SOURCE_DATASET}
    sbatch --dependency=afterok:${jobid} submit_analysis.sbatch quality ${SOURCE_DATASET}
    sbatch --dependency=afterok:${jobid} submit_analysis.sbatch tokens ${SOURCE_DATASET}
    sbatch --dependency=afterok:${jobid} submit_analysis.sbatch embeddings ${SOURCE_DATASET}
    sbatch --dependency=afterok:${jobid} submit_analysis.sbatch dataset_stats ${SOURCE_DATASET}
done

# Data preparation
sbatch submit_data_construction.sbatch
sbatch submit_data_construction_various_sample_size.sbatch

# Main EMNLP 2025 experiments script
# falcon
jobid=$(sbatch --parsable submit_finetuning.sbatch falcon emnlp25_100k_random)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch falcon emnlp25_100k_random

jobid=$(sbatch --parsable  submit_finetuning.sbatch falcon emnlp25_100k_longest)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch falcon emnlp25_100k_longest

jobid=$(sbatch --parsable  submit_finetuning.sbatch falcon emnlp25_100k_deita)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch falcon emnlp25_100k_deita

jobid=$(sbatch --parsable submit_finetuning.sbatch falcon emnlp25_100k_dedicated_quality_v2)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch falcon emnlp25_100k_dedicated_quality_v2

jobid=$(sbatch --parsable  submit_finetuning.sbatch falcon emnlp25_100k_difficulty_v2)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch falcon emnlp25_100k_difficulty_v2

jobid=$(sbatch --parsable  submit_finetuning.sbatch falcon emnlp25_100k_combination_v5)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch falcon emnlp25_100k_combination_v5

jobid=$(sbatch --parsable  submit_finetuning.sbatch falcon emnlp25_100k_combination_v5_cat)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch falcon emnlp25_100k_combination_v5_cat

jobid=$(sbatch --parsable  submit_finetuning.sbatch falcon emnlp25_100k_clustering_cat_combination_v5)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch falcon emnlp25_100k_clustering_cat_combination_v5

jobid=$(sbatch --parsable  submit_finetuning.sbatch falcon emnlp25_all)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch falcon emnlp25_all

# llama
jobid=$(sbatch --parsable  submit_finetuning.sbatch llama emnlp25_100k_random)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch llama emnlp25_100k_random

jobid=$(sbatch --parsable  submit_finetuning.sbatch llama emnlp25_100k_longest)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch llama emnlp25_100k_longest

jobid=$(sbatch --parsable  submit_finetuning.sbatch llama emnlp25_100k_deita)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch llama emnlp25_100k_deita

jobid=$(sbatch --parsable  submit_finetuning.sbatch llama emnlp25_100k_dedicated_quality_v2)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch llama emnlp25_100k_dedicated_quality_v2

jobid=$(sbatch --parsable  submit_finetuning.sbatch llama emnlp25_100k_difficulty_v2)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch llama emnlp25_100k_difficulty_v2

jobid=$(sbatch --parsable  submit_finetuning.sbatch llama emnlp25_100k_combination_v5)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch llama emnlp25_100k_combination_v5

jobid=$(sbatch --parsable  submit_finetuning.sbatch llama emnlp25_100k_combination_v5_cat)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch llama emnlp25_100k_combination_v5_cat

jobid=$(sbatch --parsable  submit_finetuning.sbatch llama emnlp25_100k_clustering_cat_combination_v5)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch llama emnlp25_100k_clustering_cat_combination_v5

jobid=$(sbatch --parsable  submit_finetuning.sbatch llama emnlp25_all)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch llama emnlp25_all

# mistral
jobid=$(sbatch --parsable  submit_finetuning.sbatch mistral emnlp25_100k_random)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch mistral emnlp25_100k_random

jobid=$(sbatch --parsable  submit_finetuning.sbatch mistral emnlp25_100k_longest)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch mistral emnlp25_100k_longest

jobid=$(sbatch --parsable  submit_finetuning.sbatch mistral emnlp25_100k_deita)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch mistral emnlp25_100k_deita

jobid=$(sbatch --parsable  submit_finetuning.sbatch mistral emnlp25_100k_dedicated_quality_v2)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch mistral emnlp25_100k_dedicated_quality_v2

jobid=$(sbatch --parsable  submit_finetuning.sbatch mistral emnlp25_100k_difficulty_v2)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch mistral emnlp25_100k_difficulty_v2

jobid=$(sbatch --parsable  submit_finetuning.sbatch mistral emnlp25_100k_combination_v5)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch mistral emnlp25_100k_combination_v5

jobid=$(sbatch --parsable  submit_finetuning.sbatch mistral emnlp25_100k_combination_v5_cat)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch mistral emnlp25_100k_combination_v5_cat

jobid=$(sbatch --parsable  submit_finetuning.sbatch mistral emnlp25_100k_clustering_cat_combination_v5)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch mistral emnlp25_100k_clustering_cat_combination_v5

jobid=$(sbatch --parsable  submit_finetuning.sbatch mistral emnlp25_all)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch mistral emnlp25_all

# qwen
jobid=$(sbatch --parsable  submit_finetuning.sbatch qwen emnlp25_100k_random)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch qwen emnlp25_100k_random

jobid=$(sbatch --parsable  submit_finetuning.sbatch qwen emnlp25_100k_longest)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch qwen emnlp25_100k_longest

jobid=$(sbatch --parsable  submit_finetuning.sbatch qwen emnlp25_100k_deita)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch qwen emnlp25_100k_deita

jobid=$(sbatch --parsable  submit_finetuning.sbatch qwen emnlp25_100k_dedicated_quality_v2)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch qwen emnlp25_100k_dedicated_quality_v2

jobid=$(sbatch --parsable  submit_finetuning.sbatch qwen emnlp25_100k_difficulty_v2)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch qwen emnlp25_100k_difficulty_v2

jobid=$(sbatch --parsable  submit_finetuning.sbatch qwen emnlp25_100k_combination_v5)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch qwen emnlp25_100k_combination_v5

jobid=$(sbatch --parsable  submit_finetuning.sbatch qwen emnlp25_100k_combination_v5_cat)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch qwen emnlp25_100k_combination_v5_cat

jobid=$(sbatch --parsable  submit_finetuning.sbatch qwen emnlp25_100k_clustering_cat_combination_v5)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch qwen emnlp25_100k_clustering_cat_combination_v5

jobid=$(sbatch --parsable  submit_finetuning.sbatch qwen emnlp25_all)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch qwen emnlp25_all

# smollm
jobid=$(sbatch --parsable  submit_finetuning.sbatch smollm emnlp25_100k_random)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch smollm emnlp25_100k_random

jobid=$(sbatch --parsable  submit_finetuning.sbatch smollm emnlp25_100k_longest)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch smollm emnlp25_100k_longest

jobid=$(sbatch --parsable  submit_finetuning.sbatch smollm emnlp25_100k_deita)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch smollm emnlp25_100k_deita

jobid=$(sbatch --parsable  submit_finetuning.sbatch smollm emnlp25_100k_dedicated_quality_v2)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch smollm emnlp25_100k_dedicated_quality_v2

jobid=$(sbatch --parsable  submit_finetuning.sbatch smollm emnlp25_100k_difficulty_v2)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch smollm emnlp25_100k_difficulty_v2

jobid=$(sbatch --parsable  submit_finetuning.sbatch smollm emnlp25_100k_combination_v5)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch smollm emnlp25_100k_combination_v5

jobid=$(sbatch --parsable  submit_finetuning.sbatch smollm emnlp25_100k_combination_v5_cat)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch smollm emnlp25_100k_combination_v5_cat

jobid=$(sbatch --parsable  submit_finetuning.sbatch smollm emnlp25_100k_clustering_cat_combination_v5)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch smollm emnlp25_100k_clustering_cat_combination_v5

jobid=$(sbatch --parsable  submit_finetuning.sbatch smollm emnlp25_all)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch smollm emnlp25_all

# Evaluation of base and instruct models
sbatch submit_evaluation.sbatch falcon base
sbatch submit_evaluation.sbatch falcon instruct

sbatch submit_evaluation.sbatch llama base
sbatch submit_evaluation.sbatch llama instruct

sbatch submit_evaluation.sbatch mistral base
sbatch submit_evaluation.sbatch mistral instruct

sbatch submit_evaluation.sbatch qwen base
sbatch submit_evaluation.sbatch qwen instruct

sbatch submit_evaluation.sbatch smollm base
sbatch submit_evaluation.sbatch smollm instruct


# Ablation experiments with different data sizes for mistral
jobid=$(sbatch --parsable  submit_finetuning.sbatch mistral emnlp25_50k_random)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch mistral emnlp25_50k_random

jobid=$(sbatch --parsable  submit_finetuning.sbatch mistral emnlp25_50k_longest)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch mistral emnlp25_50k_longest

jobid=$(sbatch --parsable  submit_finetuning.sbatch mistral emnlp25_50k_deita)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch mistral emnlp25_50k_deita

jobid=$(sbatch --parsable  submit_finetuning.sbatch mistral emnlp25_50k_clustering_cat_combination_v5)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch mistral emnlp25_50k_clustering_cat_combination_v5

jobid=$(sbatch --parsable  submit_finetuning.sbatch mistral emnlp25_25k_random)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch mistral emnlp25_25k_random

jobid=$(sbatch --parsable  submit_finetuning.sbatch mistral emnlp25_25k_longest)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch mistral emnlp25_25k_longest

jobid=$(sbatch --parsable  submit_finetuning.sbatch mistral emnlp25_25k_deita)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch mistral emnlp25_25k_deita

jobid=$(sbatch --parsable  submit_finetuning.sbatch mistral emnlp25_25k_clustering_cat_combination_v5)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch mistral emnlp25_25k_clustering_cat_combination_v5

jobid=$(sbatch --parsable  submit_finetuning.sbatch mistral emnlp25_10k_random)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch mistral emnlp25_10k_random

jobid=$(sbatch --parsable  submit_finetuning.sbatch mistral emnlp25_10k_longest)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch mistral emnlp25_10k_longest

jobid=$(sbatch --parsable  submit_finetuning.sbatch mistral emnlp25_10k_deita)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch mistral emnlp25_10k_deita

jobid=$(sbatch --parsable  submit_finetuning.sbatch mistral emnlp25_10k_clustering_cat_combination_v5)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch mistral emnlp25_10k_clustering_cat_combination_v5

jobid=$(sbatch --parsable  submit_finetuning.sbatch mistral emnlp25_5k_random)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch mistral emnlp25_5k_random

jobid=$(sbatch --parsable  submit_finetuning.sbatch mistral emnlp25_5k_longest)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch mistral emnlp25_5k_longest

jobid=$(sbatch --parsable  submit_finetuning.sbatch mistral emnlp25_5k_deita)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch mistral emnlp25_5k_deita

jobid=$(sbatch --parsable  submit_finetuning.sbatch mistral emnlp25_5k_clustering_cat_combination_v5)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch mistral emnlp25_5k_clustering_cat_combination_v5

jobid=$(sbatch --parsable  submit_finetuning.sbatch mistral emnlp25_1k_random)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch mistral emnlp25_1k_random

jobid=$(sbatch --parsable  submit_finetuning.sbatch mistral emnlp25_1k_longest)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch mistral emnlp25_1k_longest

jobid=$(sbatch --parsable  submit_finetuning.sbatch mistral emnlp25_1k_deita)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch mistral emnlp25_1k_deita

jobid=$(sbatch --parsable  submit_finetuning.sbatch mistral emnlp25_1k_clustering_cat_combination_v5)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch mistral emnlp25_1k_clustering_cat_combination_v5

# Ablation - sampling efficiency
jobid=$(sbatch --parsable  submit_finetuning.sbatch mistral emnlp25_25k_random_proportional)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch mistral emnlp25_25k_random_proportional

jobid=$(sbatch --parsable  submit_finetuning.sbatch mistral emnlp25_25k_random_proportional_clustering)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch mistral emnlp25_25k_random_proportional_clustering

jobid=$(sbatch --parsable  submit_finetuning.sbatch mistral emnlp25_25k_combination_v5)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch mistral emnlp25_25k_combination_v5

jobid=$(sbatch --parsable  submit_finetuning.sbatch mistral emnlp25_25k_combination_v5_cat)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch mistral emnlp25_25k_combination_v5_cat

# Ablation - quality scorer efficiency
jobid=$(sbatch --parsable  submit_finetuning.sbatch mistral emnlp25_25k_random_code_only)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch mistral emnlp25_25k_random_code_only

jobid=$(sbatch --parsable  submit_finetuning.sbatch mistral emnlp25_25k_random_math_only)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch mistral emnlp25_25k_random_math_only

jobid=$(sbatch --parsable  submit_finetuning.sbatch mistral emnlp25_25k_random_gen_brain_only)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch mistral emnlp25_25k_random_gen_brain_only

jobid=$(sbatch --parsable  submit_finetuning.sbatch mistral emnlp25_25k_random_fqa_extr_reas_only)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch mistral emnlp25_25k_random_fqa_extr_reas_only

jobid=$(sbatch --parsable  submit_finetuning.sbatch mistral emnlp25_25k_dedicated_quality_v2_code_only)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch mistral emnlp25_25k_dedicated_quality_v2_code_only

jobid=$(sbatch --parsable  submit_finetuning.sbatch mistral emnlp25_25k_dedicated_quality_v2_math_only)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch mistral emnlp25_25k_dedicated_quality_v2_math_only

jobid=$(sbatch --parsable  submit_finetuning.sbatch mistral emnlp25_25k_dedicated_quality_v2_gen_brain_only)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch mistral emnlp25_25k_dedicated_quality_v2_gen_brain_only

jobid=$(sbatch --parsable  submit_finetuning.sbatch mistral emnlp25_25k_dedicated_quality_v2_fqa_extr_reas_only)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch mistral emnlp25_25k_dedicated_quality_v2_fqa_extr_reas_only

jobid=$(sbatch --parsable  submit_finetuning.sbatch mistral emnlp25_25k_deita_quality_code_only)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch mistral emnlp25_25k_deita_quality_code_only

jobid=$(sbatch --parsable  submit_finetuning.sbatch mistral emnlp25_25k_deita_quality_math_only)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch mistral emnlp25_25k_deita_quality_math_only

jobid=$(sbatch --parsable  submit_finetuning.sbatch mistral emnlp25_25k_deita_quality_gen_brain_only)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch mistral emnlp25_25k_deita_quality_gen_brain_only

jobid=$(sbatch --parsable  submit_finetuning.sbatch mistral emnlp25_25k_deita_quality_fqa_extr_reas_only)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch mistral emnlp25_25k_deita_quality_fqa_extr_reas_only

jobid=$(sbatch --parsable  submit_finetuning.sbatch mistral emnlp25_25k_dedicated_quality_v2)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch mistral emnlp25_25k_dedicated_quality_v2

jobid=$(sbatch --parsable  submit_finetuning.sbatch mistral emnlp25_25k_deita_quality)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch mistral emnlp25_25k_deita_quality

# Skewed experiments
jobid=$(sbatch --parsable  submit_finetuning.sbatch mistral emnlp25_skewed_1_25k_clustering_combination_v5_cat)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch mistral emnlp25_skewed_1_25k_clustering_combination_v5_cat

jobid=$(sbatch --parsable  submit_finetuning.sbatch mistral emnlp25_skewed_1_25k_random)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch mistral emnlp25_skewed_1_25k_random

jobid=$(sbatch --parsable  submit_finetuning.sbatch mistral emnlp25_skewed_1_25k_longest)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch mistral emnlp25_skewed_1_25k_longest

jobid=$(sbatch --parsable  submit_finetuning.sbatch mistral emnlp25_skewed_1_25k_deita)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch mistral emnlp25_skewed_1_25k_deita

jobid=$(sbatch --parsable  submit_finetuning.sbatch mistral emnlp25_skewed_2_25k_clustering_combination_v5_cat)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch mistral emnlp25_skewed_2_25k_clustering_combination_v5_cat

jobid=$(sbatch --parsable  submit_finetuning.sbatch mistral emnlp25_skewed_2_25k_random)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch mistral emnlp25_skewed_2_25k_random

jobid=$(sbatch --parsable  submit_finetuning.sbatch mistral emnlp25_skewed_2_25k_longest)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch mistral emnlp25_skewed_2_25k_longest

jobid=$(sbatch --parsable  submit_finetuning.sbatch mistral emnlp25_skewed_2_25k_deita)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch mistral emnlp25_skewed_2_25k_deita

jobid=$(sbatch --parsable  submit_finetuning.sbatch mistral emnlp25_skewed_3_25k_clustering_combination_v5_cat)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch mistral emnlp25_skewed_3_25k_clustering_combination_v5_cat

jobid=$(sbatch --parsable  submit_finetuning.sbatch mistral emnlp25_skewed_3_25k_random)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch mistral emnlp25_skewed_3_25k_random

jobid=$(sbatch --parsable  submit_finetuning.sbatch mistral emnlp25_skewed_3_25k_longest)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch mistral emnlp25_skewed_3_25k_longest

jobid=$(sbatch --parsable  submit_finetuning.sbatch mistral emnlp25_skewed_3_25k_deita)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch mistral emnlp25_skewed_3_25k_deita

jobid=$(sbatch --parsable  submit_finetuning.sbatch mistral emnlp25_skewed_4_25k_clustering_combination_v5_cat)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch mistral emnlp25_skewed_4_25k_clustering_combination_v5_cat

jobid=$(sbatch --parsable  submit_finetuning.sbatch mistral emnlp25_skewed_4_25k_random)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch mistral emnlp25_skewed_4_25k_random

jobid=$(sbatch --parsable  submit_finetuning.sbatch mistral emnlp25_skewed_4_25k_longest)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch mistral emnlp25_skewed_4_25k_longest

jobid=$(sbatch --parsable  submit_finetuning.sbatch mistral emnlp25_skewed_4_25k_deita)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch mistral emnlp25_skewed_4_25k_deita

jobid=$(sbatch --parsable  submit_finetuning.sbatch mistral emnlp25_skewed_5_25k_clustering_combination_v5_cat)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch mistral emnlp25_skewed_5_25k_clustering_combination_v5_cat

jobid=$(sbatch --parsable  submit_finetuning.sbatch mistral emnlp25_skewed_5_25k_random)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch mistral emnlp25_skewed_5_25k_random

jobid=$(sbatch --parsable  submit_finetuning.sbatch mistral emnlp25_skewed_5_25k_longest)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch mistral emnlp25_skewed_5_25k_longest

jobid=$(sbatch --parsable  submit_finetuning.sbatch mistral emnlp25_skewed_5_25k_deita)
sbatch --dependency=afterok:${jobid} submit_evaluation.sbatch mistral emnlp25_skewed_5_25k_deita

# Tulu experiments
sbatch submit_evaluation_tulu.sbatch 1k_clustering_combination_v5_cat_bs512
sbatch submit_evaluation_tulu.sbatch 1k_random_bs512
sbatch submit_evaluation_tulu.sbatch 5k_clustering_combination_v5_cat_bs512
sbatch submit_evaluation_tulu.sbatch 5k_random_bs512
sbatch submit_evaluation_tulu.sbatch 10k_clustering_combination_v5_cat_bs1024
sbatch submit_evaluation_tulu.sbatch 10k_random_bs1024
sbatch submit_evaluation_tulu.sbatch 25k_clustering_combination_v5_cat_bs1024
sbatch submit_evaluation_tulu.sbatch 25k_random_bs1024
sbatch submit_evaluation_tulu.sbatch 50k_clustering_combination_v5_cat_bs1024
sbatch submit_evaluation_tulu.sbatch 50k_random_bs1024
sbatch submit_evaluation_tulu.sbatch 100k_clustering_combination_v5_cat_bs1024
sbatch submit_evaluation_tulu.sbatch 100k_random_bs1024
sbatch submit_evaluation_tulu.sbatch 200k_clustering_combination_v5_cat_bs1024
sbatch submit_evaluation_tulu.sbatch 200k_random_bs1024
sbatch submit_evaluation_tulu.sbatch 400k_clustering_combination_v5_cat_bs1024
sbatch submit_evaluation_tulu.sbatch 400k_random_bs1024
sbatch submit_evaluation_tulu.sbatch sft_mixture_0225_bs1024

sbatch submit_evaluation_lq_source.sbatch 1k_clustering_combination_v5_cat_bs512
sbatch submit_evaluation_lq_source.sbatch 1k_random_bs512
sbatch submit_evaluation_lq_source.sbatch 5k_clustering_combination_v5_cat_bs512
sbatch submit_evaluation_lq_source.sbatch 5k_random_bs512
sbatch submit_evaluation_lq_source.sbatch 10k_clustering_combination_v5_cat_bs1024
sbatch submit_evaluation_lq_source.sbatch 10k_random_bs1024
sbatch submit_evaluation_lq_source.sbatch 25k_clustering_combination_v5_cat_bs1024
sbatch submit_evaluation_lq_source.sbatch 25k_random_bs1024
sbatch submit_evaluation_lq_source.sbatch 50k_clustering_combination_v5_cat_bs1024
sbatch submit_evaluation_lq_source.sbatch 50k_random_bs1024
sbatch submit_evaluation_lq_source.sbatch 100k_clustering_combination_v5_cat_bs1024
sbatch submit_evaluation_lq_source.sbatch 100k_random_bs1024
sbatch submit_evaluation_lq_source.sbatch all_bs1024
