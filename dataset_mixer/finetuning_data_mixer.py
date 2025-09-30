import argparse
import json
import random
import os
import torch
import gc
import math
from chromadb_utils import create_collection, populate_collection, add_to_collection
from tqdm import tqdm
import yaml
from time import time
from collections import Counter
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from copy import copy

from sentence_transformers import SentenceTransformer
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA

use_fp16 = True
device = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(1234)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        help="Path to yaml file containing dataset configs.",
    )
    parser.add_argument(
        "--shuffle",
        type=bool,
        default=True,
        help="Whether to shuffle the final dataset.",
    )
    parser.add_argument(
        "--train_split",
        type=float,
        default=0.0,
        help="Fraction of train split (the rest as val).",
    )
    parser.add_argument(
        "--analysis_dir",
        type=str,
        default="/data/horse/ws/luwe911g-nemotron_ws/data/analysis",
        help="Where to store the analysis results."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data/horse/ws/luwe911g-nemotron_ws/data",
        help="Where to store the generated data."
    )
    parser.add_argument(
        "--num_devices",
        type=int,
        default=1,
        help="Number of GPUs"
    )
    return parser.parse_args()


setfit_categories = ['Math', 'Generation', 'Coding', 'Extraction', 'Reasoning', 'Factual QA', 'Brainstorming']
colors = ['C0','C1','C2','C3','C4','C5','C6']

COLOR_MAP = {category: color for category, color in zip(setfit_categories + [np.nan,        'no_setfit_label'], 
                                                        colors            + ['lightgrey',   'lightgrey'])}


class BaseSampler:
    def __init__(self, dataset_name, config):
        self.all_data = []
        self.all_tags = set()
        self.all_token_size = 0
        self.data_ids = set()
        self.dataset_name = dataset_name
        self.metric_names = {'embedding_distance': 'embedding_distance', # how the folder is called vs. how the key is called
                             'complexity_scores': 'complexity', 
                             'quality_scores': 'quality', 
                             'instagger': 'ins_tags', 
                             'reward_scores': 'reward',
                             'categories_v2': 'setfit_label',
                             'reward_diff': 'reward_diff',
                             'token_length/instructions': 'inst_length',
                             'token_length/responses': 'resp_length',
                             'if_quality_scores': 'if_quality',
                             'prm_scores': 'process_reward',
                             'code_edu_scores': 'code_edu',
                             'difficulty_scores_2': 'difficulty',
                             'difficulty_v2_scores': 'difficulty_v2',
                             'code_quality_scores': 'code_quality',
                             }
        self.metric_scores_dir = {val: key for key, val in self.metric_names.items()}

        self.config = config
        self._check_data_presence(self.config['data'])
        self.p_upper_metrics, self.p_lower_metrics = self._get_metric_normalisation_factors(self.config['data'],
                                                                                  ['complexity', 'quality', 'reward',
                                                                                   'if_quality', 'process_reward', 'code_edu',
                                                                                   'code_quality',
                                                                                   'difficulty', 'difficulty_v2'])
        self.chromadb_collection = create_collection(collection_name="instruction_embeddings",
                                                     embedding_model=self.config.pop("default_embedding_model"),
                                                     device=device)
        self.max_seq_length = 2048
        self.num_devices = args.num_devices

    def set_sampling_strategy(self):
        self.scoring_strategy = 'default' if 'scoring_strategy' not in self.config else self.config['scoring_strategy']

        if self.scoring_strategy == 'default':
            self.reward_diff_weight = 0.0
            self.rewards_weight = 0.4
            self.quality_weight = 0.4
            self.complexity_weight = 0.2
            self.difficulty_weight = 0.0
            self.token_length_weight = 0.0
        elif self.scoring_strategy.startswith('longest'):
            self.reward_diff_weight = 0.0
            self.rewards_weight = 0.0
            self.quality_weight = 0.0
            self.complexity_weight = 0.0
            self.difficulty_weight = 0.0
            self.token_length_weight = 1.0
        elif self.scoring_strategy == 'rewards':
            self.reward_diff_weight = 0.0
            self.rewards_weight = 1.0
            self.quality_weight = 0.0
            self.complexity_weight = 0.0
            self.difficulty_weight = 0.0
            self.token_length_weight = 0.0
        elif self.scoring_strategy == 'rewards+quality':
            self.reward_diff_weight = 0.0
            self.rewards_weight = 0.5
            self.quality_weight = 0.5
            self.complexity_weight = 0.0
            self.difficulty_weight = 0.0
            self.token_length_weight = 0.0
        elif self.scoring_strategy == 'reward_diff':
            self.reward_diff_weight = 1.0
            self.rewards_weight = 0.0
            self.quality_weight = 0.0
            self.complexity_weight = 0.0
            self.difficulty_weight = 0.0
            self.token_length_weight = 0.0
        elif self.scoring_strategy == 'difficulty' or self.scoring_strategy == 'difficulty_v2':
            self.reward_diff_weight = 0.0
            self.rewards_weight = 0.0
            self.quality_weight = 0.0
            self.complexity_weight = 0.0
            self.difficulty_weight = 1.0
            self.token_length_weight = 0.0
        elif self.scoring_strategy.startswith('no_rewards'):
            self.reward_diff_weight = 0.0
            self.rewards_weight = 0.0
            self.quality_weight = 0.7
            self.complexity_weight = 0.3
            self.difficulty_weight = 0.0
            self.token_length_weight = 0.0
        elif self.scoring_strategy.startswith('deita'):
            self.reward_diff_weight = 0.0
            self.rewards_weight = 0.0
            self.quality_weight = 1.0
            self.complexity_weight = 1.0
            self.difficulty_weight = 0.0
            self.token_length_weight = 0.0
        elif self.scoring_strategy.startswith('random'):
            self.reward_diff_weight = 0.0
            self.rewards_weight = 0.0
            self.quality_weight = 0.0
            self.complexity_weight = 0.0
            self.difficulty_weight = 0.0
            self.token_length_weight = 0.0
        elif self.scoring_strategy.startswith('quality'):
            self.reward_diff_weight = 0.0
            self.rewards_weight = 0.0
            self.quality_weight = 1.0
            self.complexity_weight = 0.0
            self.difficulty_weight = 0.0
            self.token_length_weight = 0.0
        elif self.scoring_strategy.startswith('combination_v'):
            self.reward_diff_weight = 0.0
            self.rewards_weight = 0.0
            self.quality_weight = 1.0
            self.complexity_weight = 0.0
            self.difficulty_weight = 1.0
            self.token_length_weight = 0.0
        elif self.scoring_strategy.startswith('combination'):
            self.reward_diff_weight = 0.0
            self.rewards_weight = 0.0
            self.quality_weight = 0.5
            self.complexity_weight = 0.0
            self.difficulty_weight = 0.5
            self.token_length_weight = 0.0
        elif self.scoring_strategy.startswith('dedicated_quality'):
            self.reward_diff_weight = 0.0
            self.rewards_weight = 0.0
            self.quality_weight = 1.0
            self.complexity_weight = 0.0
            self.difficulty_weight = 0.0
            self.token_length_weight = 0.0



    def load_dataset(self, dataset_config):
        with open(dataset_config['data_path']) as finst:
            samples = [json.loads(line) for line in finst.readlines()]
        samples = self.preprocess_data(samples, dataset_config=dataset_config)
        samples = self._add_metrics(samples, dataset_config['name'])
        samples = self._add_origin(samples, dataset_config['name'])
        if 'language' in dataset_config:
            samples = self._add_language(samples, dataset_config['language'])
        return samples

    @staticmethod
    def _identify_language(model, text):
        # Use the model to predict the language of the text
        predictions = model.predict(text, k=1)  # k=1 means returning top 1 prediction
        language_code = predictions[0][0].replace("__label__", "")  # Extract the language code
        confidence = predictions[1][0]  # Extract the confidence score
        return language_code, confidence


    def _add_conversation(self, samples):
        conversational_samples = []
        for sample in samples:
            if 'input' in sample:
                sample['messages'] = [
                    {"role": "user", "content": f"{sample['instruction']}\n\n{sample['input']}"},
                    {"role": "assistant", "content": sample['output']}
                ]
                del sample['input']
            else:
                sample['messages'] = [
                    {"role": "user", "content": sample['instruction']},
                    {"role": "assistant", "content": sample['output']}
                ]
            del sample['instruction']
            del sample['output']
            conversational_samples.append(sample)
        return conversational_samples
    
    def _add_origin(self, samples, origin):
        return [{**sample, 'origin': origin} if 'origin' not in sample else {**sample} for sample in samples]

    def _add_language(self, samples, lang):
        return [{**sample, 'language': lang.lower()} if 'language' not in sample else {**sample} for sample in samples]
        
    def _add_metrics(self, samples, dataset_name):
        for metric_name, metric_clean_name in self.metric_names.items():
            try:
                if metric_name == "instagger":
                    with open(f"{args.analysis_dir}/{metric_name}/{dataset_name}.jsonl") as f:
                        for sample, line in zip(samples, f.readlines()): 
                            sample.update({metric_clean_name: json.loads(line) if line.strip() != 'None' else None}) 
                elif metric_name == "categories_v2":
                    with open(f"{args.analysis_dir}/{metric_name}/{dataset_name}_aggr.csv") as f:
                        for sample, line in zip(samples, f.readlines()):
                            sample.update({metric_clean_name: line.strip() if line.strip() != 'None' else None})
                else:
                    with open(f"{args.analysis_dir}/{metric_name}/{dataset_name}.csv") as f:
                        for sample, line in zip(samples, f.readlines()): 
                            if metric_name.startswith("categories"): #if line.strip() != 'None':
                                sample.update({metric_clean_name: line.strip() if line.strip() != 'None' else None})
                            else:
                                sample.update({metric_clean_name: float(line.strip()) if line.strip() != 'None' else None})
            except FileNotFoundError:
                print(f"Could not find {metric_clean_name} for {dataset_name} at default path.")
        return samples
    
    def save_metrics(self):
        """Save collective metrics of constructed dataset"""
        for metric_name, metric_clean_name in self.metric_names.items():
            if metric_name == "instagger":
                with open(f"{args.analysis_dir}/{metric_name}/{self.dataset_name}.jsonl", "w") as f:
                    metric_values = [inst.get(metric_clean_name, None) for inst in self.sampled_data]
                    for value in metric_values:
                        f.write(json.dumps(value) + "\n")
            elif metric_name == "categories_v2":
                with open(f"{args.analysis_dir}/{metric_name}/{dataset_name}_aggr.csv", "w") as f:
                    metric_values = [inst.get(metric_clean_name, None) for inst in self.sampled_data]
                    for value in metric_values:
                        f.write(f"{value}\n")
            else:
                try:
                    with open(f"{args.analysis_dir}/{metric_name}/{self.dataset_name}.csv", "w") as f:
                        metric_values = [inst.get(metric_clean_name, None) for inst in self.sampled_data]
                        for value in metric_values:
                            f.write(f"{value}\n")
                except:
                    print(f"Access denied for {args.analysis_dir}/{metric_name}/{self.dataset_name}.csv. Pass..")
                    pass

    def save_dataset(self, train_split=None):
        """Save constructed dataset"""

        with open(f"{args.output_dir}/{self.dataset_name}.jsonl", "w") as f:
            for inst in self.sampled_data:
                f.write(json.dumps(inst)+"\n")

        if train_split >= 0.0:
            random.shuffle(self.sampled_data)
            path = f'{args.output_dir}/{self.dataset_name}'
            os.makedirs(path, exist_ok=True)

            if train_split > 0.0 and train_split < 1.0:
                train_split_size = int(train_split * len(self.sampled_data))
                with open(path + '/train.json', 'w') as file:
                    json.dump(self.sampled_data[:train_split_size], file, indent=4)
                with open(path + '/val.json', 'w') as file:
                    json.dump(self.sampled_data[train_split_size:], file, indent=4)
            elif train_split == 1.0:
                with open(path + '/train.json', 'w') as file:
                    json.dump(self.sampled_data, file, indent=4)
            else:
                with open(path + '/val.json', 'w') as file:
                    json.dump(self.sampled_data, file, indent=4)
                
    def plot_final_composition(self, by='origin'):
        """ TODO: plot exact amounts of samples, not just percentages"""
        used_keys = [by] 
        results = {key: [instruction.get(key, None) for instruction in self.sampled_data] for key in used_keys}
        results_df = pd.DataFrame(results)
        counts = Counter(results_df[by])
        plt.figure()
        plt.pie(counts.values(), labels=counts.keys(), autopct='%1.1f%%', wedgeprops={'linewidth': 0.5, 'edgecolor': 'white'}, textprops={'size': 'small'})
        plt.title({self.dataset_name}, fontsize=12)
        plt.tight_layout()
        plt.savefig(f"dataset_configs/{self.dataset_name}_{by}.png", bbox_inches="tight")
             
    def plot_metrics(self, by='origin'):
        """Plot metrics of constructed dataset"""
        used_keys = ['origin', 'overall_preference'] + list(self.metric_names.values())
        results = {key: [instruction.get(key, None) for instruction in self.sampled_data] for key in used_keys}
        results['categories'] = [inst.get('setfit_label', 'no_setfit_label') for inst in self.sampled_data]
        results_df = pd.DataFrame(results)

        # Scores
        for metric_name, metric_name_clean in self.metric_names.items():
            if metric_name in ["instagger", "categories", "categories_v2"]:
                continue
            try:
                plt.figure(figsize=(8, 6))
                sns.histplot(data=results_df, x=metric_name_clean, hue=by, multiple="stack")
                plt.axvline(results_df[metric_name_clean].mean(), color='grey', linestyle='--')
                plt.text(results_df[metric_name_clean].mean(), 0, f"mean: {results_df[metric_name_clean].mean():.2f}", rotation=90)
                plt.title(self.dataset_name)
                plt.savefig(f"{args.analysis_dir}/{metric_name}/{self.dataset_name}_by_{by}.png")
            except:
                pass
        
        # Overall preference
        plt.figure(figsize=(8, 6))
        # plt.xlim(0, 3)
        plt.yscale("log")
        sns.histplot(data=results_df, x='overall_preference', hue=by, multiple="stack")
        plt.title(self.dataset_name)
        os.makedirs(f"{args.analysis_dir}/overall_preferences", exist_ok=True)
        plt.savefig(f"{args.analysis_dir}/overall_preferences/{self.dataset_name}_by_{by}.png")
        
        # Categories
        counts = Counter(results_df['categories'])
        plt.figure()
        plt.pie(counts.values(), labels=counts.keys(), autopct='%1.1f%%', wedgeprops={'linewidth': 0.5, 'edgecolor': 'white'},
            textprops={'size': 'small'}, colors=[COLOR_MAP[key] for key in counts.keys()])
        plt.title({self.dataset_name}, fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{args.analysis_dir}/categories_v2/{self.dataset_name}.png", bbox_inches="tight")
        

    def format_prompt(self, sample):
        return sample['instruction']
    
    def init_chromadb(self, embedding_model):
        documents = []
        for sample in self.all_data:
            documents.append(self.format_prompt(sample))
        self.chromadb_collection = create_collection(collection_name="instruction_embeddings",
                                            embedding_model=embedding_model, device=device)
        populate_collection(collection=self.chromadb_collection, documents=documents, batch_size=100)

    def empty_collection(self):
        doc_ids = self.chromadb_collection.get()["ids"]
        if doc_ids:
            self.chromadb_collection.delete(ids=doc_ids)

    def set_embedders(self, config):
        default_embedder = config.pop("default_embedding_model") if "default_embedding_model" in config else "NovaSearch/stella_en_400M_v5"
        if "custom_embedder" in config:
            self.embedder_map = {category: config["custom_embedder"].get(category, default_embedder) for category in setfit_categories}
        else:
            self.embedder_map = {category: default_embedder for category in setfit_categories}
        self.embedder_map["all"] = default_embedder
        print(f"Embedder map: {self.embedder_map}")

    def min_max_scaling(self, score, p_upper, p_lower):
        return (score - p_lower) / (p_upper - p_lower)

    def calculate_overall_preference(self, samples, scores_exist = True):
        """Calculate a simple preference score TODO: this can use some refactoring"""
        for sample in samples:
            sample['overall_preference'] = 0

            if scores_exist and self.scoring_strategy != "random":
                if "edu_score" in sample:
                    self.edu_score_weight = 1.0
                    self.rewards_weight = 0.0
                    self.quality_weight = 0.0
                    self.complexity_weight = 0.0
                else:
                    self.edu_score_weight = 0.0

                # Dedicated quality scores per category
                dedicated_quality_score = None
                if "dedicated_quality" in self.scoring_strategy or "combination" in self.scoring_strategy:
                    if sample['setfit_label'] == "Generation" or sample['setfit_label'] == "Brainstorming":
                        dedicated_quality_score = self.min_max_scaling(sample['if_quality'],
                                                                       self.p_upper_metrics['if_quality'],
                                                                       self.p_lower_metrics['if_quality'])

                    elif sample['setfit_label'] == "Math":
                        dedicated_quality_score = self.min_max_scaling(sample['process_reward'],
                                                                       self.p_upper_metrics['process_reward'],
                                                                       self.p_lower_metrics['process_reward'])

                    elif sample['setfit_label'] == "Coding":
                        if "code_quality" in self.scoring_strategy:
                            dedicated_quality_score = self.min_max_scaling(sample['code_quality'],
                                                                       self.p_upper_metrics['code_quality'],
                                                                       self.p_lower_metrics['code_quality'])
                        else:
                            dedicated_quality_score = self.min_max_scaling(sample['code_edu'],
                                                                           self.p_upper_metrics['code_edu'],
                                                                           self.p_lower_metrics['code_edu'])

                    else:   # deita-quality scores as default
                        dedicated_quality_score = self.min_max_scaling(sample['quality'],
                                                                       self.p_upper_metrics['quality'],
                                                                       self.p_lower_metrics['quality'])

                if self.quality_weight == 1.0 and self.complexity_weight == 1.0:    # deita-style
                    sample.update({'overall_preference': sample['quality'] * sample['complexity']})

                elif self.quality_weight == 1.0 and self.difficulty_weight == 1.0:    # deita-style
                    difficulty_score = self.min_max_scaling(sample['difficulty_v2'],
                                                           self.p_upper_metrics['difficulty_v2'],
                                                           self.p_lower_metrics['difficulty_v2'])
                    if dedicated_quality_score < 0 and difficulty_score < 0:
                        final_score = 0.0
                    else:
                        final_score = dedicated_quality_score * difficulty_score
                    sample.update({'overall_preference': final_score})

                else:
                    if self.rewards_weight > 0.0:
                        sample.update({'overall_preference': sample['overall_preference'] + (self.rewards_weight * self.min_max_scaling(sample['reward'],
                                                                           self.p_upper_metrics['reward'],
                                                                           self.p_lower_metrics['reward']))})

                    if self.quality_weight > 0.0:
                        if dedicated_quality_score:
                            sample.update({'overall_preference': sample['overall_preference'] + (
                                        self.quality_weight * dedicated_quality_score)})
                        else:   # deita-quality scores as default
                            sample.update({'overall_preference': sample['overall_preference'] + (
                                        self.rewards_weight * self.min_max_scaling(sample['quality'],
                                                                                   self.p_upper_metrics['quality'],
                                                                                   self.p_lower_metrics['quality']))})

                    if self.complexity_weight > 0.0:
                        sample.update({'overall_preference': sample['overall_preference'] + (
                                    self.rewards_weight * self.min_max_scaling(sample['complexity'],
                                                                               self.p_upper_metrics['complexity'],
                                                                               self.p_lower_metrics['complexity']))})
                    if self.difficulty_weight > 0.0:
                        sample.update({'overall_preference': sample['overall_preference'] + (
                                    self.difficulty_weight * self.min_max_scaling(sample['difficulty_v2'],
                                                                                  self.p_upper_metrics['difficulty_v2'],
                                                                                  self.p_lower_metrics['difficulty_v2']))})

                    if self.edu_score_weight > 0.0:
                        sample.update({'overall_preference': sample['overall_preference'] + (
                                self.rewards_weight * self.min_max_scaling(sample['edu_score'],
                                                                           self.p_upper_metrics['edu_score'],
                                                                           self.p_lower_metrics['edu_score']))})

                    if self.reward_diff_weight == 1.0:
                        sample['overall_preference'] += self.reward_diff_weight * sample['reward_diff']

                    if self.token_length_weight > 0.0:
                        sample['overall_preference'] += sample['resp_length']

            else:
                # raise ValueError(f"Please calculate missing dataset metrics. See sample: \n{sample}")
                sample['overall_preference'] = 1
        return samples
    
    def preprocess_data(self, data, **kwargs):
        """Preprocess data; this is a passthrough function"""
        return data

    def add_gsm8k_formatting(self, messages):
        formatted_messages = []
        for turn in messages:
            if turn['role'] == "user":
                format = random.randint(0,2)
                if format == 0:     # No template
                    formatted_messages.append({'role': turn['role'],
                                               'content': turn['content']})
                elif format == 1:
                    formatted_messages.append({'role': turn['role'],
                                               'content': "Question: "+turn['content']+"\nAnswer:"})
                elif format == 2:
                    formatted_messages.append({'role': turn['role'],
                                               'content': "Q: " + turn['content'] + "\nA:"})
            else:
                formatted_messages.append(turn)
        return formatted_messages

    def postprocess_data(self, ordering, **kwargs):
        """Postprocess data; this is a passthrough function"""

        # Clean empty system messages
        for sample in self.sampled_data:
            if sample['messages'][0]['role'] == "system" and sample['messages'][0]['content'] == "":
                sample['messages'] = sample['messages'][1:]

        # Formatting
        if "fmath" in ordering:
            math_data = [sample for sample in self.sampled_data if sample.get('setfit_label') == "Math"]
            others_data = [sample for sample in self.sampled_data if sample.get('setfit_label') != "Math"]
            self.sampled_data = others_data
            while len(math_data) > 0:
                sample = math_data.pop()
                sample['messages'] = self.add_gsm8k_formatting(sample['messages'])
                self.sampled_data.append(sample)

            random.shuffle(self.sampled_data)

        if ordering == "shuffle":
            random.shuffle(self.sampled_data)

        elif "difficulty" in ordering:
            print("Ordering data based on difficulty scores...")
            self.sampled_data = sorted(self.sampled_data, key=lambda x: x['difficulty_v2'])

        elif "length" in ordering:
            print("Ordering data based on length...")
            self.sampled_data = sorted(self.sampled_data, key=lambda x: x['inst_length'] + x['resp_length'])

        if "category" in ordering:
            print("Ordering data based on categories...")
            ordered_data = []
            for category in setfit_categories:
                ordered_data += [sample for sample in self.sampled_data if sample.get('setfit_label') == category]
            self.sampled_data = ordered_data

        elif "packing_math" in ordering:
            math_data = [sample for sample in self.sampled_data if sample.get('setfit_label') == "Math"]
            others_data = [sample for sample in self.sampled_data if sample.get('setfit_label') != "Math"]
            math_data_single_turn = [sample for sample in math_data if len(sample['messages']) == 2]
            math_data_multi_turn = [sample for sample in math_data if len(sample['messages']) > 2]

            self.sampled_data = others_data + math_data_multi_turn
            half_of_math = len(math_data_single_turn) / 2
            k = 5
            while len(math_data_single_turn) > half_of_math:   # Apply this only on 50% of data
                new_messages = []
                if len(math_data_single_turn) > (k + 1):
                    for i in range(k):
                        new_messages += math_data_single_turn.pop()['messages']
                else:
                    for i in range(len(math_data_single_turn) - 1):
                        new_messages += math_data_single_turn.pop()['messages']
                last_one = math_data_single_turn.pop()
                last_one['messages'] = new_messages + last_one['messages']

                self.sampled_data.append(last_one)

            self.sampled_data += math_data_single_turn

            random.shuffle(self.sampled_data)

        prompt_completion = False
        if prompt_completion:
            for sample in self.sampled_data:
                sample['prompt'] = sample['messages'][:-1]
                sample['completion'] = sample['messages'][-1:]
                sample.pop('messages')


    def remove_language_from_id(self, data_id) -> str:
        """ Remove language code from data id, to avoid cross-lingual duplicates. data_id should have pattern: {dataset_id}_v{version}_{language}_{datapoint_id}"""
        if data_id is None: 
            return None
        else: 
            id_split = data_id.split("_")
            return "_".join(id_split[:2] + id_split[3:])

    def check_within_budget(self, dataset_config, num_sampled_data, num_sampled_tokens):
        if 'sample_size' in dataset_config:
            return num_sampled_data < dataset_config['sample_size']
        elif 'token_size' in dataset_config:
            return num_sampled_tokens < dataset_config['token_size']
        else:   # by default, consider all samples
            return True
         
    def process_data(self, dataset_config):
        """Sample data and add them to the all_data"""
        data = self.load_dataset(dataset_config)
        dataset_name = dataset_config['name']
        if 'subset_name' in dataset_config: dataset_name += f"_{dataset_config['subset_name']}"
        
        allow_x_lang_duplicates = True
        if 'allow_x_lang_duplicates' in dataset_config: allow_x_lang_duplicates = dataset_config['allow_x_lang_duplicates']

        sample_scored = True
        if "sample_scored" in dataset_config: sample_scored = dataset_config['sample_scored']
        
        oversampling = 1.0
        if 'oversampling' in dataset_config: oversampling = dataset_config['oversampling']

        self.set_sampling_strategy()
        data = self.calculate_overall_preference(data, sample_scored)

        threshold = 0.1
        if 'min_distance' in dataset_config: threshold = dataset_config['min_distance']

        if 'filter_key' in dataset_config:
            # in case of subsets, filter the data according to subset name
            filter_key = dataset_config['filter_key']
            data = [instruction for instruction in data if instruction.get(filter_key) == dataset_config['subset_name']]

        self.init_setfit_limits(dataset_config)

        if self.scoring_strategy == "random":
            sorted_samples = data
            random.Random(42).shuffle(sorted_samples)
        else:
            sorted_samples = data
            random.Random(42).shuffle(sorted_samples)
            sorted_samples = sorted(data, key=lambda x: x['overall_preference'], reverse=True)
        sorted_samples = [copy(item) for item in sorted_samples for _ in range(int(oversampling))] + copy(sorted_samples[:int(len(sorted_samples) * (oversampling - int(oversampling)))])


        if ('sample_size' in dataset_config and dataset_config['sample_size'] < len(sorted_samples)) \
                or 'setfit_limits' in dataset_config:
            num_sampled_data = 0
            num_sampled_tokens = 0
            pbar = tqdm(total=dataset_config['sample_size'])
            self.empty_collection()
            i = -1
            while self.check_within_budget(dataset_config, num_sampled_data, num_sampled_tokens) and i < len(sorted_samples) - 1:
                i += 1
                sample = sorted_samples[i]

                # Filter out samples longer than max_seq_length
                if sample["inst_length"] + sample["resp_length"] > self.max_seq_length:
                    pass

                # Check setfit limits
                if 'setfit_limits' in dataset_config:
                    if not self.check_setfit_limits(sample.get('setfit_label')):
                        continue

                query_text = self.format_prompt(sample)
                if self.scoring_strategy == "deita" and self.chromadb_collection.get()["ids"]:
                    # Query ChromaDB for similar instructions
                    results = self.chromadb_collection.query(
                        query_texts=[query_text],
                        n_results=1)
                    distance = results['distances'][0][0]
                    if distance < threshold:
                        continue

                data_id_no_lang = self.remove_language_from_id(sample.get('data_id', None))
                x_lang_duplicate = data_id_no_lang in self.data_ids

                if not x_lang_duplicate or allow_x_lang_duplicates or x_lang_duplicate is None:
                    self.all_data.append(sample)
                    num_sampled_data += 1
                    num_sampled_tokens += sample["inst_length"] + sample["resp_length"]
                    pbar.update(1)
                    if self.scoring_strategy == "deita":
                        add_to_collection(
                            collection=self.chromadb_collection,
                            documents=[query_text])
                    self.data_ids.add(data_id_no_lang)
                    if 'setfit_limits' in dataset_config:
                        self.update_setfit_limits(sample.get('setfit_label'))
        else:
            self.all_data += sorted_samples
            num_sampled_data = len(sorted_samples)
            num_sampled_tokens = 0
            for i in range(num_sampled_data):
                num_sampled_tokens += sorted_samples[i]["inst_length"] + sorted_samples[i]["resp_length"]

        # print("(after) top 20: ", [sample['overall_preference'] for sample in sorted_samples[:20]])
        # print("(after) bottom 20: ", [sample['overall_preference'] for sample in sorted_samples[-20:]])

        del sorted_samples  # just to be safe

        print(f"+{dataset_name}; Sampled: {num_sampled_data}; Total: {len(self.all_data)}; "
            f"Tokens: {num_sampled_tokens}")
            

    def get_embeddings(self, data, emb_name, multi_process=True):
        inputs = [self.format_prompt(sample) for sample in data]
        emb_model = SentenceTransformer(emb_name, trust_remote_code=True)
        if use_fp16:
            emb_model = emb_model.half()

        if multi_process:
            # Start the multi-process pool on all available CUDA devices
            pool = emb_model.start_multi_process_pool()

            # Compute the embeddings using the multi-process pool
            emb = emb_model.encode_multi_process(inputs, pool, normalize_embeddings=True, batch_size=512, show_progress_bar=True)

            # Optional: Stop the processes in the pool
            emb_model.stop_multi_process_pool(pool)
        else:
            emb = emb_model.encode(inputs, convert_to_tensor=True, normalize_embeddings=True, batch_size=512,  device=device)

        print("Embeddings computed. Shape:", emb.shape)

        del emb_model
        gc.collect()
        torch.cuda.empty_cache()
        return emb

    def run_clustering(self, data, config):
        type_clustering, sample_size = config['clustering'], config['sample_size']
        if "final_setfit_proportions" in config:
            final_setfit_proportions = config['final_setfit_proportions']
        else:
            final_setfit_proportions = {}
        self.sampled_data = []
        categories = setfit_categories if type_clustering == "categories" else ["all"] if type_clustering == "all" else None
        assert categories is not None, "Please choose a valid clustering strategy [Options: None, all, categories]"

        # This is the percentile of leftover data to consider for diversity (1.0 -> Only consider highest scoring datapoints, no clustering; 0.0 -> cluster with all available datapoints)
        threshold_percentile_clustering = 0.0
        if 'threshold_percentile_clustering' in config: threshold_percentile_clustering = config["threshold_percentile_clustering"]

        for category in tqdm(categories):
            # number of clusters is equivalent to numbers of samples in category
            proportion = 1.0 if category == "all" else final_setfit_proportions[category]
            num_clusters = n_target_samples = int(proportion * sample_size)
            print(category, num_clusters)

            # get data
            if category == "all":
                category_data = data
            else:
                category_data = [sample for sample in data if sample.get('setfit_label') == category]
            category_data = sorted(category_data, key=lambda x: x['overall_preference'], reverse=True)
                
            if len(category_data) < num_clusters:
                print(f"WARNING! Less samples than required for category {category} (required: {sample_size}x{proportion}={num_clusters} / available: {len(category_data)}). Sampling all available.")
                self.sampled_data.extend(category_data)
                print(f"Sampled: {len(category_data)} for category {category}; Total: {len(self.sampled_data)}")
                continue
            
            # get threshold for preference score
            # percent_leftover = 1 - (num_clusters / len(category_data))
            # threshold_percentile = threshold_percentile_clustering * percent_leftover
            # threshold = np.percentile([dp["overall_preference"] for dp in category_data], int(threshold_percentile * 100))
            # print(f"{category} threshold. total_samples - target_samples = {percent_leftover}; absolute threshold for overall preference: {threshold}; Percentage of all data below threshold {threshold_percentile}.")

            threshold = np.percentile([dp["overall_preference"] for dp in category_data],
                                      int(threshold_percentile_clustering * 100))

            # get embeddings
            start_embeddings = time()
            embedder_name = self.embedder_map[category]
            embeddings = self.get_embeddings(category_data, embedder_name)
            print(f"Time ellapsed for getting embeddings: {time() - start_embeddings}s.")
            
            # do dim reduction
            pca = PCA(n_components=0.99, svd_solver='full')
            embeddings = pca.fit_transform(embeddings)
            
            if embeddings.shape[0] < embeddings.shape[1]:
                print(f"WARNING! To few datapoints for category {category}. Clustering will not be meaningful.")
            
            # cluster
            start_clustering = time()
            if embeddings.shape[0] > 25_000:
                kmeans = MiniBatchKMeans(n_clusters=num_clusters,
                                         init='k-means++',
                                         max_iter=10, #50,
                                         batch_size=1024, #2048,
                                         random_state=42, n_init='auto',
                                         verbose=0)
                clusters = kmeans.fit_predict(embeddings)

            else:
                kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10, verbose=5)
                clusters = kmeans.fit_predict(embeddings)
            print(f"Time ellapsed for clustering: {time() - start_clustering}s.")
            
            # calculate metrics
            sil_score, db_index, ch_index = silhouette_score(embeddings, clusters), davies_bouldin_score(embeddings, clusters), calinski_harabasz_score(embeddings, clusters)
            print("#"* 50)
            print(f"Seperability metrics for clustering of category {category}:")
            print(f"-"*50)
            print(  f"Silhouette Score: {sil_score} (↑Higher is better)\n" \
                    f"Davies-Bouldin Index: {db_index} (↓Lower is better)\n" \
                    f"Calinski-Harabasz Index: {ch_index} (↑Higher is better)")
            print(f"-"*50)
            print(f"Datapoints per cluster: {len(category_data) / num_clusters}")
            print("#"* 50)

            print("#clusters", len(set(clusters)))
            num_sample_from_cluster = 0
            for cluster_id in set(clusters):
                indices = np.where(clusters == cluster_id)[0]
                if self.scoring_strategy == "random":
                    selected_index = random.choice(indices)
                    self.sampled_data.append(category_data[selected_index])
                    num_sample_from_cluster += 1
                else:
                    scores = [category_data[i]['overall_preference'] for i in indices]
                    if max(scores) > threshold:
                        selected_index = indices[np.argmax(scores)]
                        self.sampled_data.append(category_data[selected_index])
                        num_sample_from_cluster += 1

            print("from clustering", category, threshold, "sampled from clusters:", num_sample_from_cluster)
            data_ids = [sample['data_id'] for sample in self.sampled_data]

            for sample in category_data:
                if num_sample_from_cluster >= n_target_samples:
                    break
                if sample['data_id'] not in data_ids:
                    self.sampled_data.append(sample)
                    num_sample_from_cluster += 1

            print("final sampling", category, threshold, "num sampled:", num_sample_from_cluster, "total sampled:", len(self.sampled_data))

            # Free up memory
            del kmeans
            del clusters
            del embeddings

            # print(f"Sampled: {num_sampled_data} for category {category}; Total: {len(self.sampled_data)}")
            print("#"* 50)

    def proportional_sampling(self, sorted_samples, sample_size, setfit_proportions):
        filtered_samples = []
        # sort data by setfit label
        samples_by_setfit = {}
        for setfit_label in setfit_proportions.keys():
            samples_by_setfit[setfit_label] = [sample for sample in sorted_samples if sample['setfit_label'] == setfit_label]
            num_samples = math.ceil(setfit_proportions[setfit_label] * sample_size)
            filtered_samples += samples_by_setfit[setfit_label][:num_samples]
            print("- ", setfit_label, num_samples)

        return filtered_samples[:sample_size]

    def embedding_based_sampling(self, sorted_samples, sample_size, threshold=0.1):

        if 'min_distance' in dataset_config: threshold = dataset_config['min_distance']
        filtered_samples = []
        num_sampled_data = 0
        num_sampled_tokens = 0
        pbar = tqdm(total=len(sorted_samples))
        self.empty_collection()
        print("###", self.chromadb_collection.count())
        i = -1
        while i < sample_size and i < len(sorted_samples):
            i += 1
            sample = sorted_samples[i]

            # Filter out samples longer than max_seq_length
            if sample["inst_length"] + sample["resp_length"] > self.max_seq_length:
                pass

            # Query ChromaDB for similar instructions
            query_text = self.format_prompt(sample)
            if self.chromadb_collection.get()["ids"]:
                results = self.chromadb_collection.query(
                    query_texts=[query_text],
                    n_results=1)
                distance = results['distances'][0][0]
            else:
                distance = threshold+1  #First instance , always in

            if distance > threshold:
                filtered_samples.append(sample)
                num_sampled_data += 1
                num_sampled_tokens += sample["inst_length"] + sample["resp_length"]
                pbar.update(1)
                add_to_collection(
                    collection=self.chromadb_collection,
                    documents=[query_text])

        return filtered_samples


    def run_sampling(self, config):
        sample_size = config['sample_size']

        # Sort data points based on selected scoring function
        if self.scoring_strategy == "random":
            sorted_samples = self.all_data
            random.Random(math.floor(sample_size/1000)).shuffle(sorted_samples)
        else:
            sorted_samples = sorted(self.all_data, key=lambda x: x['overall_preference'], reverse=True)

        if "clustering" in config and config["clustering"] is not None:
            self.run_clustering(self.all_data, config) # Options: None, all, categories
        elif "embedding" in self.scoring_strategy:
            self.sampled_data = self.embedding_based_sampling(sorted_samples, sample_size)
        else:
            if "final_setfit_proportions" in config:
                self.sampled_data = self.proportional_sampling(sorted_samples, sample_size, config['final_setfit_proportions'])
            else:
                self.sampled_data = sorted_samples[:sample_size]
        
        del self.all_data # Delete to free memory and avoid accidental use

    
    def _check_data_presence(self, data_configs):
        # check that data exists
        for dataset in data_configs:
            dataset_name = dataset['name']
            assert os.path.exists(dataset['data_path']), f"Data path {dataset['data_path']} does not exist."
        
            # check that metrics exist
            for metric_name in self.metric_names.keys():
                    if metric_name == "instagger":
                        if not os.path.exists(f"{args.analysis_dir}/{metric_name}/{dataset_name}.jsonl"):
                            print(f"Could not find {metric_name} for {dataset_name} at default path.")
                    else:
                        if not os.path.exists(f"{args.analysis_dir}/{metric_name}/{dataset_name}.csv"):
                            print(f"Could not find {metric_name} for {dataset_name} at default path.")

    def _get_metric_normalisation_factors(self, data_configs, metric_names):
        """loads metrics from all datasets and calculates 95th percentile for normalisation"""
        
        all_metric_values = {metric: [] for metric in metric_names}
        # Load metrics from all datasets
        for dataset in data_configs:
            dataset_name = dataset['name']
            for metric_name in metric_names:
                try:
                    with open(f"{args.analysis_dir}/{self.metric_scores_dir[metric_name]}/{dataset_name}.csv") as f:
                        all_metric_values[metric_name] += [float(line.strip()) for line in f.readlines() if line.strip() != 'None' and line.strip() != "NaN"]
                except FileNotFoundError:
                    pass

        p_upper_all_metrics = {}
        p_lower_all_metrics = {}
        # Calculate n-th percentile for normalisation
        for metric_name in metric_names:
            if all_metric_values[metric_name]:
                p_upper_all_metrics[metric_name] = np.percentile(all_metric_values[metric_name], 99)
                p_lower_all_metrics[metric_name] = np.percentile(all_metric_values[metric_name], 1)

        print(p_upper_all_metrics, p_lower_all_metrics)
        return (p_upper_all_metrics, p_lower_all_metrics)
        
    def _final_setfit_filtering(self, data, setfit_proportions):
        """ Take dict of setfit labels and proportions, determine maximal possible dataset-size and filter the dataset accordingly""" 
        # sort data by setfit label
        samples_by_setfit = {}
        for setfit_label in setfit_proportions.keys():
            samples_by_setfit[setfit_label] = [sample for sample in data if sample['setfit_label'] == setfit_label]

        # determine maximally possible final size based on fixed proportions
        max_possible_size = int(min([len(samples_by_setfit[setfit_label]) / proportion if proportion > 0 else 10**9 for setfit_label, proportion in setfit_proportions.items()]))

        print(f"Shrinking dataset from {len(data)} to {max_possible_size} samples.")

        # filter data based on overall preference following setfit labels
        orig_data_size = len(data)
        old_size_variable_proportions = sum([len(samples) for setfit_label, samples in samples_by_setfit.items() if setfit_proportions[setfit_label] < 0]) / orig_data_size
        new_size_variable_proportions = (1 - sum([proportion for proportion in setfit_proportions.values() if proportion > 0]))
                        
        filtered_data = []
        for setfit_label, proportion in setfit_proportions.items():
            if proportion < 0:
                # if proportion is not specified, keep original proportion of samples
                proportion = (len(samples_by_setfit[setfit_label]) / orig_data_size / old_size_variable_proportions) * new_size_variable_proportions

            samples_by_setfit[setfit_label] = sorted(samples_by_setfit[setfit_label], key=lambda x: x['overall_preference'], reverse=True)
            filtered_data.extend(samples_by_setfit[setfit_label][:int(max_possible_size  * proportion)])

        self.all_tags = set()
        for sample in filtered_data:
            if "ins_tags" in sample and sample["ins_tags"] is not None:
                tags = set([tag['tag'] for tag in sample["ins_tags"] if "tag" in tag])
                self.all_tags.update(tags)

        print(f"Total: {len(filtered_data)}; InsTags: {len(self.all_tags)}")
        return filtered_data
    
    def init_setfit_limits(self, dataset_config):
        """Set max numbers to be added to dataset per setfit label"""
        if 'setfit_limits' in dataset_config:
            self.setfit_limits = {category: limit * dataset_config['sample_size'] 
                                        for setfit_label in dataset_config['setfit_limits'] 
                                        for category, limit in setfit_label.items()}
            print(self.setfit_limits)

    def check_setfit_limits(self, setfit_label):
        """Check if there are more samples that can be added to the dataset for a given setfit label"""
        return self.setfit_limits[setfit_label] > 0
    
    def update_setfit_limits(self, setfit_label):
        """Update the number of samples that can be added to the dataset for a given setfit label"""
        self.setfit_limits[setfit_label] -= 1

class SFTSampler(BaseSampler):
    def __init__(self, dataset_name, config):
        super().__init__(dataset_name, config)

    def format_prompt(self, sample):
        # user_messages = ""
        # for turn in sample['messages']:
        #     if turn['role'] == "user":
        #         user_messages += f"{turn['content']} "
        # return user_messages.strip()
        for turn in sample['messages']:
            if turn['role'] == "assistant":
                return turn['content']

    def preprocess_data(self, data, dataset_config, **kwargs):
        if 'multi_turn' not in dataset_config or dataset_config['multi_turn'] == False:
            data = self._add_conversation(data)
        return data


class POSampler(BaseSampler):
    def __init__(self, dataset_name, config):
        super().__init__(dataset_name, config)
        self.type_po = config['type_dataset']

    def format_prompt(self, sample):
        if self.type_po == 'kto':
            return sample['prompt'][0]['content'].strip()
        else:
            user_messages = ""
            for turn in sample['chosen']:
                if turn['role'] == "user":
                    user_messages += f"{turn['content']} "
            return user_messages.strip()

    def preprocess_data(self, data, **kwargs):
        if self.type_po == 'kto':
            return self.package_data(data)
        else:
            return data

    def postprocess_data(self, data, shuffle=True):
        if shuffle:
            random.shuffle(data)
        if self.type_po == 'kto':
            return self.unpackage_data(data)
        return data

    def package_data(self, data):
        """Package positive and negative example into single datapoint for sampling"""
        repackaged_data = []
        for i in range(0, len(data), 2):
            assert data[i]['label'] == True and data[i+1]['label'] == False, "Asserting positive and negative example alternating."
            repackaged_data.append({**data[i], 'negative_example': data[i+1]})
        return repackaged_data

    def unpackage_data(self, data):
        """Unpackage positive and negative example from single datapoint"""
        unpackaged_data = []
        for sample in data:
            negative_example = sample.pop('negative_example')
            unpackaged_data.append(sample)
            unpackaged_data.append(negative_example)
        return unpackaged_data

if __name__ == "__main__":
    args = parse_args()
    
    with open(os.path.join('dataset_configs', args.config), 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)

    dataset_name, config = config.popitem()
    sampler = POSampler(dataset_name, config) if 'type_dataset' in config and config['type_dataset'] in ["kto", "dpo"] else SFTSampler(dataset_name, config)

    start = time()

    sampler.set_embedders(config)

    for dataset_config in tqdm(config['data'], desc="Processed datasets"):
        # Check if there are subsets to handle
        if 'subsets' in dataset_config:
            for subset_config in tqdm(dataset_config['subsets'], desc="Processed subsets"):
                subset_dataset_config = {
                    **dataset_config,  # inherit base config for the dataset
                    **subset_config    # override with subset-specific values
                }
                sampler.process_data(subset_dataset_config)
        else:
            sampler.process_data(dataset_config)

    if 'sample_size' in config:
        sampler.run_sampling(config)
    else:
        sampler.sampled_data = sampler.all_data

    ordering_strategy = "shuffle"
    if 'ordering_strategy' in config: ordering_strategy = config['ordering_strategy']
    sampler.postprocess_data(ordering=ordering_strategy)

    end = time()
    print(f"### Execution time for {args.config.replace('.yaml', '')}: {end - start:.5f} seconds.")

    sampler.save_dataset(train_split=args.train_split)
    sampler.save_metrics()

    # sampler.plot_final_composition(by='origin')
    # sampler.plot_final_composition(by='language')
    # sampler.plot_metrics(by='origin')
    # sampler.plot_metrics(by='categories')


