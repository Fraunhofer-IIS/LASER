# LASER-analysis: A Repository for Analyzing Instructions

This repository contains code and data for analyzing instruction-following data.

- We use [SetFit](https://github.com/huggingface/setfit) for efficient few-shot learning with Sentence Transformers
for classifying instructions into 7 categories with [SIGMA-cls](https://huggingface.co/IIS-NLP-internal/sigma-cls): [Math, Coding, Generation, Brainstorming, Reasoning, Factual QA, Extraction]. We construct the training and validation data by sampling ~250 samples per category. 
Evaluated on the validation split, the classifier gets 96% macro F1-score.
- We use [Deita complexity scorer](https://huggingface.co/hkust-nlp/deita-complexity-scorer) and 
[Deita quality scorer](https://huggingface.co/hkust-nlp/deita-quality-scorer) for scoring (instruction) complexity and 
(response) quality w.r.t a given instruction
- We use a [PRM](https://huggingface.co/Qwen/Qwen2.5-Math-PRM-7B) to score math response data.
- We use our custom IF-Quality scorer (based on [Qwen/Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B)) to score how well a response adheres to constraints given by an instruction.
- We use our custom Code-Quality scorer (based on [Qwen/Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B)) to score the quality of a code-related response to a given instruction.
- We use our custom [difficulty scorer](https://huggingface.co/IIS-NLP-internal/qwen3-8B-difficulty-scorer-v2) to score how challenging given instructions are.
- We use [vllm](https://github.com/vllm-project/vllm) for faster inference

### Running Analysis

#### Prerequisites

1. Create a virtual environment ``python -m venv analysis-env`` then activate `source analysis-env/bin/activate`
2. Install the dependencies with ``pip install -r requirements.txt``

#### Running SIGMA-cls for classifying instructions into 7 categories

1. Run ``python -m run_analysis --analysis categories_v2``, which will run the classifier on all considered datasets. 
Otherwise, run ``python -m run_analysis --analysis categories_v2 --dataset <dataset_name(s)>`` to run the classifier on 
specific dataset(s), e.g., ``python -m run_analysis --analysis categories_v2 --dataset sigma_v2_evol,deita_10k``. 
List of available datasets below.
2. The categories (and pie chart) will be saved in ``categories_v2/``, e.g.,

![Alt text](categories/sigma_v2_evol.png?raw=true "Title")
![Alt text](categories/deita_10k.png?raw=true "Title")

#### Running Complexity Analysis

1. Run ``python -m run_analysis --analysis complexity``, which will run the complexity scorer on all considered datasets. 
Otherwise, run ``python -m run_analysis --analysis complexity --dataset <dataset_name(s)>`` to run the classifier on 
specific dataset(s), e.g., ``python -m run_analysis --analysis complexity --dataset sigma_v2_evol,deita_10k``. 
List of available datasets below.
2. The scores (and histogram) will be saved in ``complexity_scores/``, e.g.,

![Alt text](complexity_scores/sigma_v2_evol.png?raw=true "Title")
![Alt text](complexity_scores/deita_10k.png?raw=true "Title")

#### Running Quality Analysis

1. Run ``python -m run_analysis --analysis quality``, which will run the quality scorer on all considered datasets. 
Otherwise, run ``python -m run_analysis --analysis quality --dataset <dataset_name(s)>`` to run the classifier on 
specific dataset(s), e.g., ``python -m run_analysis --analysis quality --dataset sigma_v2_evol,deita_10k``. 
List of available datasets below.
2. The scores (and histogram) will be saved in ``quality_scores/``, e.g.,

![Alt text](quality_scores/sigma_v2_evol.png?raw=true "Title")
![Alt text](quality_scores/deita_10k.png?raw=true "Title")

#### Complexity/Quality analysis on existing datasets

Run ``python compute_correlations.py``

![Alt text](correlations/avg_complexity_quality.png?raw=true "Title")

#### Running InsTagger

1. Run ``python -m run_analysis --analysis tagging``, which will run the quality scorer on all considered datasets. 
Otherwise, run ``python -m run_analysis --analysis tagging --dataset <dataset_name(s)>`` to run the classifier on 
specific dataset(s), e.g., ``python -m run_analysis --analysis tagging --dataset sigma_v2_evol,deita_10k``. 
List of available datasets below.
2. The scores (and histogram) will be saved in ``instagger/``, e.g.,

![Alt text](instagger/sigma_v2_evol.png?raw=true "Title")
![Alt text](instagger/deita_10k.png?raw=true "Title")

#### Diversity analysis on existing datasets, based on #InsTag and embedding distance

Run ``python compute_correlations.py``

![Alt text](correlations/num_unique_tags_10k.png?raw=true "Title")
![Alt text](correlations/num_unique_tags_10k_avg_embedding.png?raw=true "Title")

#### Running Reward Model Preference analysis on existing datasets

1. Run ``python -m run_analysis --analysis reward_modelling``, which will run the reward model on all considered datasets. 
Otherwise, run ``python -m run_analysis --analysis reward_modelling --dataset <dataset_name(s)>`` to run the classifier on 
specific dataset(s), e.g., ``python -m run_analysis --analysis reward_modelling --dataset sigma_v2_evol,deita_10k``. 
List of available datasets below.
2. The scores (and histogram) will be saved in ``reward_scores/``, e.g.,

![Alt text](reward_scores/sigma_v2_evol_category.png?raw=true "Title")

#### Correlation between scores, token length and #tags

![Alt text](correlations/instlen_complexity.png?raw=true "Title")
![Alt text](correlations/resplen_quality.png?raw=true "Title")
![Alt text](correlations/numtag_complexity.png?raw=true "Title")

#### List of Analyzed Datasets
- sigma_v1
- sigma_v2_evol
- sigma_v3
- deita_10k
- no_robots
- flan_v2_cot
- dolly_15k
- alpaca
- alpaca_gpt4
- lima
- longform
- bactrian-x_en
- wizardlm_evol_instruct
- wizardlm_orca
- sharegpt
- oasst2
- ultrachat

#### Add a new dataset to be analyzed
1. Add a dataset processing in ``utils.py`` with `dataset_name` as the key, making sure that 
`instructions` contains all user requests in the dataset and `responses` contains the corresponding 
system responses. Please add also `dataset_title` to show the dataset name on the charts, e.g., 
```
    elif dataset_name == "longform":
        dataset_title = "LongForm"
        dataset = load_dataset("akoksal/LongForm")  # from Huggingface Hub
        longform_train = dataset["train"]
        instructions = [sample["input"] for sample in longform_train]
        responses = [sample["output"] for sample in longform_train]
```


