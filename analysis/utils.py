import json
import os
import argparse


from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple


PATH_TO_DATA = Path(f"../data")

##########################
## Dataset loading
##########################
# How to use: 
# You can add a dataset by adding a tuple to either single_turn_jsonl_dataset, multi_turn_jsonl_dataset or add a "wildcard" style dataset to patterns
# For multi_turn_jsonl_dataset, there is an additional element that give the possibility to add kwargs for the dataloading

@dataclass
class DatasetSpec:
    title:  str
    loader: Callable[[], Tuple[List[str], List[str], List[int]]]

REGISTRY: Dict[str, DatasetSpec] = {}

def register(name: str, title: str, loader: Callable):
    REGISTRY[name] = DatasetSpec(title, loader)

def process_conversational_data(samples, messages_str: str="messages", role_str: str="role", content_str: str="content",
                                user_str: str="user", assistant_str: str="assistant"):
    instructions = []
    responses = []
    num_exchanges = []

    for i, sample in enumerate(samples):
        if len(sample[messages_str]) < 2:  # Erroneous sample
            print(i, sample['id'])
        else:
            t = 0
            num = 0
            while sample[messages_str][t][role_str] != user_str and t < len(sample[messages_str]):
                t += 1
            while t < len(sample[messages_str]):
                if sample[messages_str][t][role_str] == user_str:
                    instructions.append(sample[messages_str][t][content_str])
                    if t + 1 < len(sample[messages_str]) and \
                            sample[messages_str][t + 1][role_str] == assistant_str:
                        responses.append(sample[messages_str][t + 1][content_str])
                    else:
                        responses.append("")
                    num += 1
                t += 2
            num_exchanges.append(num)

    return instructions, responses, num_exchanges

def _join_instruction(sample: dict) -> str:
    """join instruction with input if it is non-empty / non-missing."""
    if sample.get("instruction") and sample.get("input"):
        return f"{sample['instruction']}\n\n{sample['input']}"
    elif not sample.get("instruction") and sample.get("input"):
        return sample["input"]
    elif sample.get("instruction"):
        return sample["instruction"]
    else:
        return ""


def _jsonl_loader(filename: str) -> Callable[[], Tuple[List[str], List[str], List[int]]]:
    """Returns loader for single-turn JSONL datasets."""
    def _load():
        fp = PATH_TO_DATA / filename
        samples = [json.loads(line) for line in fp.open()]
        instructions = [_join_instruction(s) for s in samples]
        responses    = [s["output"]          for s in samples]
        return instructions, responses, []     
    return _load


def _conv_jsonl_loader(filename: str,
                       conv_kwargs: dict | None = None) -> Callable[[], Tuple[
                           List[str], List[str], List[int]]]:
    """Returns loader for conversational JSONL datasets."""
    conv_kwargs = conv_kwargs or {}
    def _load():
        fp = PATH_TO_DATA / filename
        samples = [json.loads(line) for line in fp.open()]
        return process_conversational_data(samples, **conv_kwargs)
    return _load

def _match_pattern(name: str):
    return DatasetSpec(title=name.replace("_", " "),
                       loader=_conv_jsonl_loader(f"{name}.jsonl"))

# Datasets that are registered (format: "instruction", [optional: "input"], "output")
single_turn_jsonl_dataset = [
    # (dataset-name,                                    pretty-name,                                        data location),
    ("open_math_instruct",                              "MATH",                                             "open_math_instruct.jsonl"),
]

# Datasets that are registered (format: "messages")
multi_turn_jsonl_dataset = [
    # (dataset-name,                 pretty-name,                                   data location,                          kwargs),
    ("flan_v2_90k",                  "Flan V2 90k",                                 "flan_v2_90k.jsonl",                        None),
    ("sharegpt_en",                  "ShareGPT (EN)",                               "sharegpt_en.jsonl",                        None),
    ("daring_anteater",              "Daring Anteater",                             "daring_anteater.jsonl",                    None),
    ("wizardlm_evol_instruct",       "WizardLM Evol Instruct V2",                   "wizardlm_evol_instruct.jsonl",             None),
    ("ifeval_like_5k",               "IFEval-like",                                 "ifeval_like_5k.jsonl",                     None),
    ("emnlp25_200k_agentinst_random","AgentInstruct V1 (200k, random)",             "emnlp25_200k_agentinst_random.jsonl",      None),
    ("numina_math_cot_v1",           "NuminaMath CoT",                              "numina_math_cot_v1.jsonl",                 None),
    ("conifer_v1",                   "Conifer",                                     "conifer_v1.jsonl",                         None),
    ("open_math_instruct_2",         "OpenMathInstruct-2",                          "open_math_instruct_2.jsonl",   None),
    ("ultrainteract_coding",         "UltraInteract (coding)",                      "ultrainteract_coding.jsonl",               None),
    ("dolly_15k",                    "Databricks Dolly 15k",                        "dolly_15k.jsonl",                          None),
    ("alpaca",                       "Stanford Alpaca",                             "alpaca.jsonl",                             None),
    ("alpaca_gpt4",                  "Stanford Alpaca (with GPT4)",                  "alpaca_gpt4.jsonl",                       None),
]


all_datasets = single_turn_jsonl_dataset + multi_turn_jsonl_dataset

# Wildcard patterns; can be used instead of adding stuff to the above lists
patterns = ["emnlp25_"]

# Registering dataset configs from list above
for dataset_info in all_datasets:
    if len(dataset_info) == 3:
        _name, _title, _file = dataset_info
    elif len(dataset_info) == 4:
        _name, _title, _file, _kwargs = dataset_info
    else:
        raise AssertionError
    
    if dataset_info in single_turn_jsonl_dataset:
        register(_name, _title, _jsonl_loader(_file))
    elif dataset_info in multi_turn_jsonl_dataset:
        register(_name, _title, _conv_jsonl_loader(_file, conv_kwargs=_kwargs))


def load_sft_dataset(dataset_name: str):
    spec = REGISTRY.get(dataset_name)
    if spec is None and any(dataset_name.startswith(p) for p in patterns):
        spec = _match_pattern(dataset_name)

    if spec is None:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    instructions, responses, num_exchanges = spec.loader()
    return spec.title, instructions, responses, num_exchanges

##########################
## Other utils
##########################

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        help="The dataset name. The default is 'all', all datasets will be processed.",
    )
    parser.add_argument(
        "--analysis",
        type=str,
        default="all",
        help="The analysis type. The default is 'all', all analysis will be conducted.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.3",
        help="The tokenizer used to tokenize the samples.",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="NovaSearch/stella_en_400M_v5",
        help="The embedding model for embedding instructions and/or responses.",
    )
    parser.add_argument(
        "--request_batch_size",
        type=int,
        default=512,
        help="The number of requests to send to LLM at a time."
    )
    parser.add_argument(
        "--num_devices",
        type=int,
        default=1,
        help="The number of GPUs to use."
    )
    parser.add_argument(
        "--repeat_analysis",
        action="store_true",
        help="Whether to force repeat running the analyzer.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Whether to plot the analysis.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=f"{PATH_TO_DATA}/analysis",
        help="Where to store the analysis results."
    )
    parser.add_argument(
        "--model_deployment",
        type=str,
        default="vllm",
        help="Which framework to use for model deployment. Hf option currently only for quality, reward and code_edu"
    )
    parser.add_argument(
        "--sep_tok_reason",
        type=str,
        default="\n\n",
        help="Sequence along which to split reasoning traces. Only relevant for PRM scoring."
    )

    parser.set_defaults(repeat_analysis=False)
    parser.set_defaults(plot=False)
    return parser.parse_args()


def get_analyzer(analysis_type: str, args):
    if analysis_type == "complexity":
        from analyzer.complexity_scorer import ComplexityScorer
        analyzer = ComplexityScorer(deployment=args.model_deployment, num_devices=args.num_devices)
        output_dir = os.path.join(args.output_dir, "./{analysis_type}_scores")

    elif analysis_type == "quality":
        from analyzer.quality_scorer import QualityScorer
        analyzer = QualityScorer(deployment=args.model_deployment, num_devices=args.num_devices)
        output_dir = os.path.join(args.output_dir, "./{analysis_type}_scores")

    elif analysis_type == "tokens":
        from analyzer.token_counter import TokenCounter
        analyzer = TokenCounter(args.tokenizer)
        output_dir = os.path.join(args.output_dir, "./{analysis_type}_scores")

    elif analysis_type == "categories_v2":
        from analyzer.task_classifier import TaskClassifier
        analyzer = TaskClassifier()
        output_dir = os.path.join(args.output_dir, "./categories_v2")

    elif analysis_type == "difficulty_v2":
        from analyzer.difficulty_scorer_v2 import DifficultyScorer
        analyzer = DifficultyScorer()
        output_dir = os.path.join(args.output_dir, f"./{analysis_type}_scores")
        
    elif analysis_type == "process_reward_modelling":
        from analyzer.process_reward_modeller import ProcessRewardModeller
        analyzer = ProcessRewardModeller(sep_tok_reason=args.sep_tok_reason)
        output_dir = os.path.join(args.output_dir, "./{analysis_type}_scores")

    elif analysis_type == "if_quality":
        from analyzer.if_quality_scorer import IFQualityScorer
        analyzer = IFQualityScorer(num_devices=args.num_devices)
        output_dir = os.path.join(args.output_dir, "./{analysis_type}_scores")

    elif analysis_type == "code_quality":
        from analyzer.code_quality_scorer import CodeQualityScorer
        analyzer = CodeQualityScorer(num_devices=args.num_devices)
        output_dir = os.path.join(args.output_dir, "./{analysis_type}_scores")
    
    elif analysis_type == "dataset_stats":
        analyzer = None
        output_dir = os.path.join(args.output_dir, "./dataset_stats")

    else:
        print("No analysis type found!")
        exit(0)
    
    return analyzer, output_dir
