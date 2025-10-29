# LASER: Label-Aware Scoring and clustERing 

This repository contains the code to run the LASER data selection pipeline and scripts to replicate the experiment of our EMNLP 2025 *Findings* paper [[link](https://arxiv.org/abs/2505.22157)].

### Abstract

- Studies show post-training **(instruction tuning) datasets** for LLMs can be substantially down-sampled **without deteriorating performance**. ​

- However, **Instruction Data (ID) selection** often incurs <span style="color:red">high computational costs</span> or is <span style="color:red">limited to narrow domains</span>.​

- In this work, we demonstrate that data selection can be both—**efficient and universal**—by using a multi-step pipeline, <span style="color:teal">**LASER**</span>.​

![Description](assets/LASER_illustration.jpg)

## LASER Classification & Scoring

Please check the [analysis](./analysis/).

## LASER Sampling

### Download datasets
```bash
bash scripts/download_data.sh
```
The datasets will be stored in `data/`


### Setup
Create a virtual environment and install the dependencies with (or use the existing `analysis-env`)
```bash
bash scripts/create_analysis_env.sh
```


### Specify sampling configurations
In a YAML file stored in `dataset_mixer/dataset_configs/`, e.g.,
```yaml
test_random:
  scoring_strategy: random
  sample_size: 10000
  data:
    - name: alpaca_gpt4
      data_path: DATA_PATH/alpaca_gpt4.jsonl
      type: self_instruct
      multi_turn: True
      language: en

    - name: flan_v2_90k
      data_path: DATA_PATH/flan_v2_90k.jsonl
      type: self_instruct
      multi_turn: True
      language: en

    - name: sharegpt_en
      data_path: DATA_PATH/sharegpt_en.jsonl
      type: chat
      multi_turn: True
      language: en
```

### Run sampling
1. Add sampling configurations within the `DATASET_CONFIGS=` list in `sample_data.sh`.
2. Run
```bash
bash scripts/sample_data.sh
```


### Replicate EMNLP'25 Experiments
[...]


## Citation

If you find this project is useful in your own work, please consider citing as follows:

```
@article{mirza2025stratified,
  title={Stratified Selective Sampling for Instruction Tuning with Dedicated Scoring Strategy},
  author={Mirza, Paramita and Weber, Lucas and K{\"u}ch, Fabian},
  journal={arXiv preprint arXiv:2505.22157},
  year={2025}
}
```
