# LASER-analysis: A Repository for Analyzing Instructions

![Description](../assets/LASER_illustration_analysis.jpg)

This repository contains the code for analyzing instruction datasets via the following methods:

- **Domain classification**: We use [SetFit](https://github.com/huggingface/setfit) for efficient few-shot learning with Sentence Transformers for classifying instructions into 7 categories: [*Math, Coding, Generation, Brainstorming, Reasoning, Factual QA, Extraction*].
- **Deita scoring**: We use [Deita complexity scorer](https://huggingface.co/hkust-nlp/deita-complexity-scorer) and 
[Deita quality scorer](https://huggingface.co/hkust-nlp/deita-quality-scorer) for scoring (instruction) complexity and 
(response) quality w.r.t a given instruction.
- **Math scoring**: We use a [Process Reward Model (PRM)](https://huggingface.co/Qwen/Qwen2.5-Math-PRM-7B) to score responses to math problems, in terms of reasoning steps.
- **Instruction-following scoring**: We use our custom IF-Quality scorer (with a [Qwen/Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B) backbone) to score how well a response adheres to constraints given by an instruction.
- **Code scoring**: We use our custom Code-Quality scorer (with a [Qwen/Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B) backbone) to score the quality of a code-related response to a given instruction.
- **Difficulty scoring**: We use our custom [difficulty scorer](https://huggingface.co/IIS-NLP-internal/qwen3-8B-difficulty-scorer-v2) to score how challenging given instructions are.

The majority of the listed scorers are implemented to employ [vllm](https://github.com/vllm-project/vllm) for faster inference.

## Running Analysis

Make sure that you have the necessary depenendencies installed in your environment.
### Classification
To take full advantage of routed scoring, make sure to run instruction classification before running sample scoring. You can run instruction classification like this:
```bash
# Activate the your environment
source laser-env/bin/activate

# Run classifier, e.g. for alpaca_gpt4 and flan_v2_90k
python -m run_analysis --analysis categories_v2 --dataset alpaca_gpt4,flan_v2_90k
```
The categories will be saved in ``../data/analysis/categories_v2/``.

### Scoring
```bash
# Activate the your environment
source laser-env/bin/activate

# Run scorers; e.g. if-quality scoring; the available analysis types are:
# if_quality,code_quality,process_reward_modelling,difficulty_v2,complexity,quality,tokens
python -m run_analysis --analysis if_quality --dataset alpaca_gpt4,flan_v2_90k
```
The scores will be saved in ``../data/analysis/<ANALYSIS_TYPE>_scores/``.


### Add own datasets
To analyse new datasets, simply add it as a Tuple in ``utils.py`` to the `single_turn_jsonl_dataset` (when dataset-format
follows ["instruction", optional: "input", "output"]) or to `multi_turn_jsonl_dataset`
(when dataset-format follows conversational json format ("messages")):
```python
# Datasets that are registered (format: "instruction", [optional: "input"], "output")
single_turn_jsonl_dataset = [
    # (dataset-name,        pretty-name,   data location),
    ("open_math_instruct",  "MATH",        "open_math_instruct.jsonl"),
]

# Datasets that are registered (format: "messages")
multi_turn_jsonl_dataset = [
    # (dataset-name,   pretty-name,    data location,        kwargs),
    ("flan_v2_90k",    "Flan V2 90k",  "flan_v2_90k.jsonl",  None),
...
]
```


