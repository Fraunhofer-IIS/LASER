#!/bin/bash

cd dataset_mixer/

# TODO: Add all experimental configs here
DATASET_CONFIGS=(
    "emnlp25/100k_random.yaml"
    "test_random.yaml"
)

for CONFIG in "${DATASET_CONFIGS[@]}"; do
    echo "Sample data for config: $CONFIG"
    python finetuning_data_mixer.py --config "$CONFIG" 
done
