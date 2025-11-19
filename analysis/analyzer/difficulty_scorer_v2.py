import logging
import sys
import torch
import gc
from typing import List
from tqdm import tqdm
from analyzer.utils import plot_histogram, plot_histogram_per_category

from transformers import AutoModelForCausalLM

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


CATEGORIES = ["Math", 
            "Coding", 
            "Brainstorming",
            "Extraction",
            "Generation",
            "Reasoning",
            "Factual QA"]

category_model_map = {"all": "IIS-NLP/difficulty-scorer-8B-v2",
                     # **{category: "IIS-NLP/difficulty-scorer-8B-v2" for category in CATEGORIES} # Currently, category specific scorer are not implemented, if we do, add them here
                     } 


class DifficultyScorer(object):
    def __init__(self, deployment: str = "hf", num_devices: int = 1, max_tokens: int = 2, category_specific: bool = False):
        self.category_specific = category_specific
        self.categories_to_analyze = CATEGORIES if category_specific else ["all"]
        self.deployment = deployment
        self.difficulty_template = "You are an expert of {category} data. You judge problems for their difficulty."
        self.llm = None
        self.current_category = None
        self.num_devices = num_devices

    def init_llm(self, num_devices=1):
        if self.llm is not None:
            self.unload_llm()
            self.llm = None
        if self.deployment == "hf":
            self.llm = AutoModelForCausalLM.from_pretrained(self.model_name, trust_remote_code=True).to("cuda")
            self.tokenizer = self.llm.get_tokenizer()
        else:
            raise NotImplementedError


    def score(self, instructions: List[str], responses: List[str]):
        """ This is a wrapper to get a method with a unified name across all scores to do scoring"""
        return self.infer_difficulty(instructions, responses)

    def infer_difficulty(self, instructions: List[str], responses: List[str], categories: List[str]):
        if not responses:
            # create placeholder
            responses = [None] * len(instructions)
        scores = []
        for instruction, _, category in tqdm(zip(instructions, responses, categories), total=len(instructions)):
            if self.current_category in [category, "all"]:
                instruction = instruction if type(instruction) == str else instruction[0]
                conv = [{"role": "system", "content": self.difficulty_template.format(category=self.current_category)},
                        {"role": "user", "content": instruction}]
                
                # encode sequence and truncate too long sequences
                conv_tokenized = self.tokenizer.apply_chat_template(conv, tokenize=True, return_tensors="pt")[:, -1024:].to(self.llm.model.device)
                with torch.no_grad():
                    scores.append(self.llm(conv_tokenized)['logits'].item())
            else:
                scores.append("NaN")
        return scores

    def run(self, instructions: List[str], responses: List[str], num_exchanges: List[int],
            dataset_name: str, dataset_title: str, output_dir: str,
            request_batch_size: int=512, negative_examples=None,
            categories: List[str] = None):
        
        file_path = f"{output_dir}/{dataset_name}.csv"
        
        # Generate difficulty scores from given scorer model

        # Init empty scores
        difficulty_scores = ["NaN"] * len(instructions)

        from collections import Counter
        category_counts = Counter(categories)

        for category in self.categories_to_analyze:
            if category_counts[category] or category == "all":
                self.model_name = category_model_map[category]
                self.init_llm() # init category-specific llm (or generic if category == "all")
                self.current_category = category

                logger.info(f"Running difficulty v2 on {self.current_category} using {self.model_name} as scorer.")
                logger.info(f"Analysing {category_counts[category] if category != 'all' else len(instructions)}/{len(instructions)} samples.")
                for i in tqdm(range(0, len(instructions), request_batch_size)):
                    batch_instructions = instructions[i: i + request_batch_size]
                    batch_responses = responses[i: i + request_batch_size]
                    if categories:
                        batch_categories = categories[i: i + request_batch_size]
                    else:  # Set all to category "all"
                        batch_categories = ["all"] * len(batch_instructions)
                    scores = self.infer_difficulty(batch_instructions, batch_responses, batch_categories)

                    for j, score in enumerate(scores):
                        if score != "NaN":
                            assert difficulty_scores[j+i] == "NaN"
                            difficulty_scores[j+i] = score
            else:
                logger.info(f"No data of type {category} in dataset. Skip!")

        # Conversational dataset, aggregate metrics per sample
        if num_exchanges:
            difficulty_scores_aggr = []
            user_msg_idx = 0
            for sample_idx in tqdm(range(0, len(num_exchanges))):
                difficulty_score_per_sample = 0
                num_responses = 0
                for i in range(num_exchanges[sample_idx]):
                    if responses[user_msg_idx] != "":
                        difficulty_score_per_sample += difficulty_scores[user_msg_idx]
                        num_responses += 1
                    else:
                        logger.debug("###", instructions[user_msg_idx], "###", responses[user_msg_idx])
                    user_msg_idx += 1
                if num_responses > 0:
                    difficulty_score_per_sample = difficulty_score_per_sample / num_responses
                else:
                    difficulty_score_per_sample = 0
                difficulty_scores_aggr.append(difficulty_score_per_sample)
        
        # Write scores to file
        with open(f"{output_dir}/{dataset_name}.csv", "w") as fout:
            if num_exchanges:
                for score in difficulty_scores_aggr:
                    fout.write(f"{score}\n")
            else:
                for score in difficulty_scores:
                    fout.write(f"{score}\n")
        
        logger.info(f"### Difficulty v2 scores are written to {output_dir}/{dataset_name}.csv")
        
        if num_exchanges:
            return difficulty_scores_aggr
        else:
            return difficulty_scores


    def plot(self, scores: List[float], dataset_name: str, dataset_title: str, output_dir: str, categories: List[str]):
        # Plot the histogram
        min_ylim = 0.0
        max_ylim = 2.0
        plot_histogram(f"{output_dir}/{dataset_name}.png", dataset_title + " (difficulty v2)", scores,
                        min_ylim, max_ylim)
        if categories:
            plot_histogram_per_category(f"{output_dir}/{dataset_name}_category.png",
                                        dataset_title + " (difficulty v2 per category)",
                                        scores, categories)

        logger.info(f"### Difficulty scores plot is written to {output_dir}/{dataset_name}.png")

    def unload_llm(self):
        if self.deployment == "vllm":
            self.llm.unload_llm()
        del self.llm
        gc.collect()
        torch.cuda.empty_cache()

