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

DATA_PATH = "../data/"

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
        default=f"{DATA_PATH}/analysis",
        help="Where to store the analysis results."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=f"{DATA_PATH}",
        help="Where the source data is stored."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=f"{DATA_PATH}",
        help="Where to store the generated data."
    )
    parser.add_argument(
        "--num_devices",
        type=int,
        default=1,
        help="Number of GPUs"
    )
    return parser.parse_args()


setfit_categories = ['Math', 'Coding', 'Generation', 'Reasoning', 'Brainstorming', 'Factual QA', 'Extraction']
colors = ['#112977','#25657D','#368770','#44994B','#81A854','#B0B55D','#C4A46F']

COLOR_MAP = {category: color for category, color in zip(setfit_categories + [np.nan,        'no_setfit_label'], 
                                                        colors            + ['lightgrey',   'lightgrey'])}

dataset_names = ['orca_agentinstruct_v1', 'alpaca_gpt4', 'flan_v2_90k', 'ifeval_like_5k', 
                 'open_math_instruct_2', 'sharegpt_en', 'ultrainteract_coding', 'wizardlm_evol_instruct',
                 'alpaca', 'longform', 'numina_math_cot_v1_250k', 'conifer_v1', 'dolly_15k', 'daring_anteater',
                 'tulu_3_sft_mixture_0225']

COLOR_MAP['orca_agentinstruct_v1'] = '#FBB4AE'
COLOR_MAP['alpaca_gpt4'] = "#9DC1DF"
COLOR_MAP['flan_v2_90k'] = '#CCEBC5'
COLOR_MAP['ifeval_like_5k'] = "#7D5B86"
COLOR_MAP['open_math_instruct_2'] = "#DFDF92"
COLOR_MAP['sharegpt_en'] = "#CAAF7A"
COLOR_MAP['ultrainteract_coding'] = "#F8C3DE"
COLOR_MAP['wizardlm_evol_instruct'] = "#D8CACA"

COLOR_MAP['alpaca'] = "#B7A7DB"
COLOR_MAP['longform'] = "#85CCC6"
COLOR_MAP['numina_math_cot_v1_250k'] = "#D3E29C"
COLOR_MAP['conifer_v1'] = "#9899B1"
COLOR_MAP['dolly_15k'] = "#CE99B6"
COLOR_MAP['daring_anteater'] = "#C9B39A"

COLOR_MAP['tulu_3_sft_mixture_0225'] = '#F2F2F2'

class BaseSampler:
    def __init__(self, dataset_name, config):
        self.all_data = []
        self.all_tags = set()
        self.all_token_size = 0
        self.data_ids = set()
        self.dataset_name = dataset_name
        self.metric_names = {# how the folder is called vs. how the key is called
                             'complexity_scores': 'complexity', 
                             'quality_scores': 'quality', 
                             'categories_v2': 'setfit_label',
                             'tokens_scores/instructions': 'inst_length',
                             'tokens_scores/responses': 'resp_length',
                             'tokens_scores/last_responses': 'last_resp_length',
                             'code_quality_scores': 'code_quality',
                             'if_quality_scores': 'if_quality',
                             'prm_scores': 'process_reward',
                             'difficulty_v2_scores': 'difficulty_v2',
                             }
        self.metric_scores_dir = {val: key for key, val in self.metric_names.items()}

        self.config = config
        self._check_data_presence(self.config['data'])
        self.norm_models = self._get_metric_normalisation_models(self.config['data'], [
                                                                                        'complexity', 'quality',
                                                                                        'if_quality', 'process_reward', 'code_quality',
                                                                                        'difficulty_v2'
                                                                                        ])
        
        self.set_sampling_strategy()
        if "embedding" in self.scoring_strategy:
            self.chromadb_collection = create_collection(collection_name="instruction_embeddings",
                                                        embedding_model=self.config.pop("default_embedding_model"),
                                                        device=device)
        self.max_seq_length = 4096 # 2048
        self.num_devices = args.num_devices

    def set_sampling_strategy(self):
        self.scoring_strategy = 'default' if 'scoring_strategy' not in self.config else self.config['scoring_strategy']

        if self.scoring_strategy == 'default':
            self.quality_weight = 1.0
            self.complexity_weight = 0.0
            self.difficulty_weight = 1.0
            self.tokens_scores_weight = 0.0
        elif self.scoring_strategy.startswith('random'):
            self.quality_weight = 0.0
            self.complexity_weight = 0.0
            self.difficulty_weight = 0.0
            self.tokens_scores_weight = 0.0
        elif self.scoring_strategy.startswith('longest'):
            self.quality_weight = 0.0
            self.complexity_weight = 0.0
            self.difficulty_weight = 0.0
            self.tokens_scores_weight = 1.0
        elif self.scoring_strategy.startswith('deita'):
            self.quality_weight = 1.0
            self.complexity_weight = 1.0
            self.difficulty_weight = 0.0
            self.tokens_scores_weight = 0.0
        elif self.scoring_strategy == 'difficulty' or self.scoring_strategy == 'difficulty_v2':
            self.quality_weight = 0.0
            self.complexity_weight = 0.0
            self.difficulty_weight = 1.0
            self.tokens_scores_weight = 0.0
        elif self.scoring_strategy.startswith('quality') or self.scoring_strategy.startswith('dedicated_quality'):
            self.quality_weight = 1.0
            self.complexity_weight = 0.0
            self.difficulty_weight = 0.0
            self.tokens_scores_weight = 0.0
        elif self.scoring_strategy.startswith('combination'):
            self.quality_weight = 1.0
            self.complexity_weight = 0.0
            self.difficulty_weight = 1.0
            self.tokens_scores_weight = 0.0

    def load_dataset(self, dataset_config):
        data_dir = dataset_config['data_path']
        data_dir = data_dir.replace("DATA_PATH", args.input_dir)
        with open(data_dir) as finst:
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

    def filter_dataset_by_language(self, samples, language_codes = ["en"]):
        import fasttext
        model = fasttext.load_model('models/lid.176.bin')
        language_codes = [l.lower() for l in language_codes]

        filtered_samples = []
        for sample in samples:
            if 'language' not in sample:
                if 'instruction' in sample:
                    language_code, confidence = self._identify_language(model, sample['instruction'].replace("\n", " "))
                else:
                    language_code, confidence = self._identify_language(model, sample['messages'][0]['content'].replace("\n", " "))
                sample['language'] = language_code
                sample['lang_confidence'] = confidence
            if sample['language'] in language_codes:
                filtered_samples.append(sample)
        return filtered_samples

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
                if metric_name == "categories_v2":
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
            try:
                os.makedirs(f"{args.analysis_dir}/{metric_name}/")
            except FileExistsError:
                # directory already exists
                pass

            if metric_name == "categories_v2":
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
                
    def plot_final_composition(self, by='origin'):
        """Plot copmositions of constructed dataset"""
        used_keys = [by] 
        results = {key: [instruction.get(key, None) for instruction in self.sampled_data] for key in used_keys}
        results_df = pd.DataFrame(results)
        counts = Counter(results_df[by])
        plt.figure()
        plt.pie(counts.values(), labels=counts.keys(), autopct='%1.1f%%', wedgeprops={'linewidth': 0.5, 'edgecolor': 'white'}, textprops={'size': 'small'})
        plt.title({self.dataset_name}, fontsize=12)
        plt.tight_layout()
        os.makedirs(f"{args.analysis_dir}/final_compositions", exist_ok=True)
        plt.savefig(f"{args.analysis_dir}/final_compositions/{self.dataset_name}_{by}.png", bbox_inches="tight")
        plt.close()

    def plot_metrics(self, by='origin'):
        """Plot metrics of constructed dataset"""
        used_keys = ['origin', 'overall_preference'] + list(self.metric_names.values())
        results = {key: [instruction.get(key, None) for instruction in self.sampled_data] for key in used_keys}
        results['categories'] = [inst.get('setfit_label', 'no_setfit_label') for inst in self.sampled_data]
        results_df = pd.DataFrame(results)
        hue_order=[item for item in setfit_categories+dataset_names if item in results_df[by].unique()]

        # Scores
        for metric_name, metric_name_clean in self.metric_names.items():
            if metric_name in ["categories_v2"]:
                continue
            try:
                plt.figure(figsize=(8, 6))
                sns.histplot(data=results_df, x=metric_name_clean, hue=by, palette=COLOR_MAP, hue_order=hue_order, multiple="stack")
                plt.axvline(results_df[metric_name_clean].mean(), color='black', linewidth=2)
                plt.text(results_df[metric_name_clean].mean(), 0.5, f"mean: {results_df[metric_name_clean].mean():.2f}", rotation=90, verticalalignment='center')
                plt.title(self.dataset_name)
                plt.savefig(f"{args.analysis_dir}/{metric_name}/{self.dataset_name}_by_{by}.png")
            except:
                pass
        
        # Overall preference
        plt.figure(figsize=(8, 6))
        if not self.scoring_strategy.startswith('longest') and not self.scoring_strategy.startswith('random'):
            plt.xlim(0.0, 1.0)
        ax = sns.histplot(data=results_df, x='overall_preference', hue=by, palette=COLOR_MAP, hue_order=hue_order, multiple="stack")
        legend = ax.get_legend()
        handles = legend.legend_handles

        ax.axvline(results_df['overall_preference'].mean(), color='grey', linewidth=1.5, label=f"mean: {results_df['overall_preference'].mean():.2f}")
        ax.axvline(results_df['overall_preference'].median(), color='grey', linewidth=1.5, linestyle='--', label=f"median: {results_df['overall_preference'].median():.2f}")
        h, l = ax.get_legend_handles_labels()

        ax.legend(handles=handles+h, labels=hue_order+l, loc="upper right")
        plt.title(self.dataset_name)
        os.makedirs(f"{args.analysis_dir}/overall_preferences", exist_ok=True)
        plt.savefig(f"{args.analysis_dir}/overall_preferences/{self.dataset_name}_by_{by}.png")
        plt.close()
        
        # Categories
        counts = Counter(results_df['categories'])
        plt.figure()
        plt.pie(counts.values(), labels=counts.keys(), autopct='%1.1f%%', wedgeprops={'linewidth': 0.5, 'edgecolor': 'white'},
            textprops={'size': 'small'}, colors=[COLOR_MAP[key] for key in counts.keys()])
        plt.title({self.dataset_name}, fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{args.analysis_dir}/categories_v2/{self.dataset_name}.png", bbox_inches="tight")
        plt.close()
        

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

        # Normalize scores with RobustScaler models
        print("Normalizing scores with RobustScaler models...")   
        for metric_clean_name in ["quality", "complexity", "if_quality", "process_reward", "code_quality"]:
            if metric_clean_name in self.norm_models:
                print(f"   - {metric_clean_name} scores...")
                scores = [sample[metric_clean_name] for sample in samples]
                norm_scores = self.norm_models[metric_clean_name].transform(np.array(scores).reshape(-1, 1)).flatten()
                for i, sample in enumerate(samples):
                    if metric_clean_name in sample and sample[metric_clean_name] is not None:
                        sample[metric_clean_name] = norm_scores[i]

        # Dedicated quality scores per category
        if "dedicated_quality" in self.scoring_strategy or "combination" in self.scoring_strategy:
            for sample in samples:
                if "dedicated_quality" in self.scoring_strategy or "combination" in self.scoring_strategy:
                    if sample['setfit_label'] == "Generation" or sample['setfit_label'] == "Brainstorming":
                        sample.update({'quality': sample['if_quality']})

                    elif sample['setfit_label'] == "Math":
                        sample.update({'quality': sample['process_reward']})

                    elif sample['setfit_label'] == "Coding":
                        sample.update({'quality': sample['code_quality']})

                    else:   # deita-quality scores as default
                        sample.update({'quality': sample['quality']})

        print("Scaling scores with min-max...")
        # Scaling quality, complexity and difficulty scores to [0, 1]
        for metric_clean_name in ["quality", "complexity", "difficulty_v2"]:
            print(f"   - {metric_clean_name} scores...")
            scores = [sample[metric_clean_name] for sample in samples if metric_clean_name in sample and sample[metric_clean_name] is not None]
            if scores:
                min_score = min(scores)
                max_score = max(scores)
                for i, sample in enumerate(samples):
                    if metric_clean_name in sample and sample[metric_clean_name] is not None:
                        curr_score = sample[metric_clean_name]
                        sample[metric_clean_name] = (curr_score - min_score) / (max_score - min_score)

        print("Updating the overall preference scores...")
        for sample in samples:
            sample['overall_preference'] = 0

            if scores_exist and self.scoring_strategy != "random":
                if self.quality_weight == 1.0 and self.complexity_weight == 1.0:    # deita-style
                    sample['overall_preference'] += sample['quality'] * sample['complexity']

                elif self.quality_weight == 1.0 and self.difficulty_weight == 1.0:    # deita-style     
                    sample['overall_preference'] += sample['quality'] * sample['difficulty_v2']

                else:

                    if self.quality_weight > 0.0:
                        sample['overall_preference'] += self.quality_weight * sample['quality']

                    if self.complexity_weight > 0.0:
                        sample['overall_preference'] += self.complexity_weight * sample['complexity']

                    if self.difficulty_weight > 0.0:
                        sample['overall_preference'] += self.difficulty_weight * sample['difficulty_v2']

                    if self.tokens_scores_weight > 0.0:
                        sample['overall_preference'] += sample['last_resp_length']

            else:
                # raise ValueError(f"Please calculate missing dataset metrics. See sample: \n{sample}")
                sample['overall_preference'] = 1

        return samples
    
    def preprocess_data(self, data, **kwargs):
        """Preprocess data; this is a passthrough function"""
        return data

    def postprocess_data(self, ordering, **kwargs):
        """Postprocess data; this is a passthrough function"""

        # Clean empty system messages
        for sample in self.sampled_data:
            if sample['messages'][0]['role'] == "system" and sample['messages'][0]['content'] == "":
                sample['messages'] = sample['messages'][1:]

        # Ordering
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

        if 'language_filter' in dataset_config:
            sorted_samples = self.filter_dataset_by_language(sorted_samples, dataset_config['language_filter'])

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
            if "inst_length" in sorted_samples[0] and "resp_length" in sorted_samples[0]:
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
                                         max_iter=50, #10,
                                         batch_size=2048, #1024,
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
            print("#"* 50)

            # Free up memory
            del kmeans
            del clusters
            del embeddings

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

        # Filter data based on token length (less than max_seq_length)
        if "inst_length" in self.all_data[0] and "resp_length" in self.all_data[0]:
            self.all_data = [sample for sample in self.all_data if (sample["inst_length"] + sample["resp_length"]) <= self.max_seq_length]
            print(f"After filtering based on token length, {len(self.all_data)} samples remain.")

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
            data_dir = dataset['data_path']
            data_dir = data_dir.replace("DATA_PATH", args.input_dir)
            assert os.path.exists(data_dir), f"Data path {data_dir} does not exist."
        
            # check that metrics exist
            for metric_name in self.metric_names.keys():
                    if metric_name == "instagger":
                        if not os.path.exists(f"{args.analysis_dir}/{metric_name}/{dataset_name}.jsonl"):
                            print(f"Could not find {metric_name} for {dataset_name} at default path.")
                    else:
                        if not os.path.exists(f"{args.analysis_dir}/{metric_name}/{dataset_name}.csv"):
                            print(f"Could not find {metric_name} for {dataset_name} at default path.")
    
    def _relevant_metric(self, category, metric_name):
        """Check if a metric is relevant for a given category"""
        if category == "Generation" or category == "Brainstorming":
            return metric_name in ["quality", "complexity", "difficulty_v2", "if_quality", "reward", "edu_score"]
        elif category == "Math":
            return metric_name in ["quality", "complexity", "difficulty_v2", "process_reward", "reward", "edu_score"]
        elif category == "Coding":
            return metric_name in ["quality", "complexity", "difficulty_v2", "code_quality", "code_edu", "reward", "edu_score"]
        else:
            return True  # Default case, all metrics are relevant

    def _get_metric_normalisation_models(self, data_configs, metric_names):
        """loads metrics from all datasets and fits PowerTransformer for normalisation"""

        all_metric_values = {metric: [] for metric in metric_names}
        # Load metrics from all datasets
        for dataset in data_configs:
            dataset_name = dataset['name']
            for metric_name in metric_names:
                try:
                    with open(f"{args.analysis_dir}/{self.metric_scores_dir[metric_name]}/{dataset_name}.csv") as f, open(f"{args.analysis_dir}/categories_v2/{dataset_name}_aggr.csv") as f_cat:
                        categories = [line.strip() for line in f_cat.readlines()]
                        scores = [float(line.strip()) if line.strip() != 'None' and line.strip() != "NaN" else 0.0 for line in f.readlines()]
                        all_metric_values[metric_name] += [score for i, score in enumerate(scores) if self._relevant_metric(categories[i], metric_name)]
                except FileNotFoundError:
                    pass
        
        normalisation_models = {}
        # Fit normalisation models (RobustScaler) for each metric
        from sklearn.preprocessing import RobustScaler
        for metric_name in metric_names:
            if all_metric_values[metric_name]:
                normalisation_models[metric_name] = RobustScaler(quantile_range=(1.0, 99.0)).fit(np.array(all_metric_values[metric_name]).reshape(-1, 1))
            else:
                print(f"No data for metric {metric_name}.")

        return (normalisation_models)
    
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

    try:
        sampler.plot_final_composition(by='origin')
        sampler.plot_final_composition(by='language')
        sampler.plot_metrics(by='origin')
        sampler.plot_metrics(by='categories')
    except Exception as e:
        print(f"Could not plot final composition or metrics: {e}")
