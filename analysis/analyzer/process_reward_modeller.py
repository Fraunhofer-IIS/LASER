import logging
import sys
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from tqdm import tqdm
from typing import List
from transformers import AutoTokenizer, AutoModel
import warnings
import torch.nn.functional as F
import re
import nltk

from analyzer.utils import plot_histogram, plot_histogram_per_category

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


class ProcessRewardModeller(object):
    def __init__(self,
                 deployment: str = "hf",
                 num_devices: int = 1,
                 sep_tok_reason: str = "\n\n",
                ):
        self.model_name = "Qwen/Qwen2.5-Math-PRM-7B"
        self.deployment = deployment
        self.init_llm(num_devices=num_devices)
        self.categories_to_analyze = ["Math"]
        self.aggregation = "min"
        self.sep_tok_reason = sep_tok_reason
        nltk.download('punkt')
        
    def init_llm(self, num_devices=1):
        if self.deployment == "hf":
            self.sampling_params = {}
            self.llm = AutoModel.from_pretrained(self.model_name,
                                                torch_dtype=torch.bfloat16,
                                                device_map="auto", 
                                                # attn_implementation="flash_attention_2",
                                                trust_remote_code=True,
                                            ).eval()
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True
                                                            )
            self.tokenizer.truncation_side = "left"
        else:
            raise NotImplementedError

    def split_responses(self, responses: List[str], sep_tok_reason: str):
        processed_responses = []
        if sep_tok_reason == '\n\n':
            for response in responses:
                if sep_tok_reason in response:
                    response = re.sub(r':\s+', ':\n', response)
                    processed_responses.append(response.split(sep_tok_reason))
                elif '\n' in response:
                    processed_responses.append(response.split('\n'))
                else:
                    sentences = nltk.sent_tokenize(response)
                    processed_responses.append(sentences)

        else:
            processed_responses = [response.split(sep_tok_reason) for response in responses]
        return processed_responses

    def score(self, instructions: List[str], responses: List[str]):
        """ This is a wrapper to get a method with a unified name across all scores to do scoring"""
        return self.generate_rewards(instructions, responses)

    def generate_rewards(self, instructions: List[str], responses: List[List[str]], categories: List[str]):
        """
        Generate rewards from a given reward model
        !! Note !!: Reponses here are not List[str], but List[List[str]] -> each response is a list of reasoning steps
        """
        scores = []
        for instruction, response, category in tqdm(zip(instructions, responses, categories), total=len(instructions)):
            if category in self.categories_to_analyze:
                messages = [
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": "<extra_0>".join(response) + "<extra_0>"},
                ]

                conversation_str = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False
                )
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    with torch.no_grad():
                        input_ids = self.tokenizer.encode(
                            conversation_str,
                            return_tensors="pt",
                        ).to(self.llm.device)
                        outputs = self.llm(input_ids=input_ids)
                        step_sep_id = self.tokenizer.encode("<extra_0>")[0]
                        token_masks = (input_ids == step_sep_id)
                        step_reward = self.make_step_rewards(outputs[0], token_masks)
                        scores.append(step_reward)
            else:
                scores.append([['NaN']])
        return scores
    
    @staticmethod
    def make_step_rewards(logits, token_masks):
        probabilities = F.softmax(logits, dim=-1)
        probabilities = probabilities * token_masks.unsqueeze(-1) # bs, seq_len, num_labels
        
        all_scores_res = []
        for i in range(probabilities.size(0)):
            sample = probabilities[i] # seq_len, num_labels
            positive_probs = sample[sample != 0].view(-1, 2)[:, 1] # valid_tokens, num_labels
            non_zero_elements_list = positive_probs.cpu().tolist()
            all_scores_res.append(non_zero_elements_list)
        return all_scores_res
    
    def run(self, instructions: List[str], responses: List[str], num_exchanges: List[int],
            dataset_name: str, dataset_title: str, output_dir: str,
            request_batch_size: int=512, negative_examples=None,
            categories: List[str] = None):
        
        responses = self.split_responses(responses, self.sep_tok_reason)
        
        if negative_examples is not None:
            warnings.warn("! Reward scoring for negative examples is currently not implemented. Use reward_diff instead!")
        
        file_path = f"{output_dir}/{dataset_name}.csv"

        
        # Generate rewards from given reward model
        reward_scores = []
        for i in tqdm(range(0, len(instructions), request_batch_size)):
            batch_instructions = instructions[i: i + request_batch_size]
            batch_responses = responses[i: i + request_batch_size]

            if categories:
                batch_categories = categories[i: i + request_batch_size]
            else:  # Set all to 'Math'
                batch_categories = [self.categories_to_analyze[0]] * len(batch_instructions)

            scores = self.generate_rewards(batch_instructions, batch_responses, batch_categories)
            if self.aggregation == "mean":
                reward_scores += [sum(score[0])/len(score[0]) for score in scores]
            elif self.aggregation == "min":
                reward_scores += [min(score[0]) for score in scores]
            else:
                reward_scores += scores

        # Conversational dataset, aggregate metrics per sample
        if num_exchanges:
            reward_scores_aggr = []
            user_msg_idx = 0
            for sample_idx in tqdm(range(0, len(num_exchanges))):
                reward_score_per_sample = 0
                num_responses = 0
                for i in range(num_exchanges[sample_idx]):
                    if responses[user_msg_idx] != "":
                        try:
                            reward_score_per_sample += reward_scores[user_msg_idx]
                            num_responses += 1
                        except:
                            pass
                    else:
                        logger.debug("###", instructions[user_msg_idx], "###", responses[user_msg_idx])
                    user_msg_idx += 1
                if num_responses > 0:
                    reward_score_per_sample = reward_score_per_sample / num_responses
                else:
                    reward_score_per_sample = 0
                reward_scores_aggr.append(reward_score_per_sample)

        # Write scores to file
        with open(f"{output_dir}/{dataset_name}.csv", "w") as fout:
            if num_exchanges:
                for score in reward_scores_aggr:
                    fout.write(f"{score}\n")
            else:
                for score in reward_scores:
                    fout.write(f"{score}\n")
        
        logger.info(f"### Reward scores are written to {output_dir}/{dataset_name}.csv")
                
        if num_exchanges:
            return reward_scores_aggr
        else:
            return reward_scores

    def plot(self, scores: List[float], dataset_name: str, dataset_title: str, output_dir: str, categories: List[str]):
        # # Plot the histogram
        # min_ylim = 0
        # max_ylim = 0
        # plot_histogram(f"{output_dir}/{dataset_name}.png", dataset_title + " (reward model preferences)", scores,
        #                min_ylim, max_ylim)
        # if categories:
        #     plot_histogram_per_category(f"{output_dir}/{dataset_name}_category.png",
        #                                 dataset_title + " (reward model preferences per category)",
        #                                 scores, categories)
        #
        # logger.info(f"### Reward scores plot is written to {output_dir}/{dataset_name}.png")
        pass

