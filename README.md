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

### Setup
[...]


### Add datasets
[...]


### Specify sampling configurations
[...]


### Replicate EMNLP'25 Experiments
[...]


## TODOs
- write READMEs 
- add `requirements.txt`
- add dataset configs for all datasets created in experiments; add them to `scripts/sample_data`
- update script (`data/download_data.py`) with the right preprocessing logic for agent-instruct
- test run repo, as is
- clean up code (especially, `dataset_mixer/finetuning_dataset_mixer.py`)

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
