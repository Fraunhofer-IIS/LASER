import logging
import sys

# setting path
sys.path.append('..')
sys.path.append('../..')

from llm.vllm_api import VLLM
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import gc
from typing import List
import numpy as np
from scipy.special import softmax
from tqdm import tqdm
from analyzer.utils import plot_histogram, plot_histogram_per_category

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

class QualityScorer(object):

    def __init__(self, categories_to_analyze: List[str], deployment: str = "vllm",
                 num_devices: str = 1, max_model_len: int = 2048):
        self.model_name = "hkust-nlp/deita-quality-scorer"
        self.deployment = deployment
        self.init_llm(num_devices=num_devices, max_model_len=max_model_len)

        self.quality_template = (
            "You are a helpful assistant. Please identify the quality score of the Response corresponding to the Question. \n #Question#:\n{instruction}\n#Response#:\n{output} \n##Quality: ")

        self.id2score = {
                    29896: "1",
                    29906: "2",
                    29941: "3",
                    29946: "4",
                    29945: "5",
                    29953: "6"
                }

        self.categories_to_analyze = categories_to_analyze

    def init_llm(self, num_devices, max_model_len):
        self.max_model_len = max_model_len
        self.max_tokens = 2
        self.logprobs = 20
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, model_max_length=self.max_model_len)
        self.tokenizer.truncation_side = "left"

        if self.deployment == "vllm":
            self.llm = VLLM(model_name=self.model_name, num_devices=num_devices, max_model_len=max_model_len)

        elif self.deployment == "hf":
            self.sampling_params = {"max_new_tokens": self.max_tokens, 
                                    "output_scores": True,
                                    "num_return_sequences": 1, 
                                    "return_dict_in_generate": True}
            with torch.no_grad():
                self.llm = AutoModelForCausalLM.from_pretrained(self.model_name,
                                                                trust_remote_code=True,
                                                                device_map="auto",
                                                                )
        else:
            raise NotImplementedError

    def infer_quality(self, user_requests: List[str], system_responses: List[str]):
        # Truncate input text
        template_input_ids = self.tokenizer(self.quality_template).input_ids
        num_token_complexity_template = len(template_input_ids)
        max_prompt_len = self.max_model_len - self.max_tokens
        num_avail_token = (max_prompt_len - num_token_complexity_template)
        truncated_user_requests = []
        truncated_system_responses = []
        for i, system_response in enumerate(system_responses):
            user_request_input_ids = self.tokenizer(user_requests[i]).input_ids
            num_token_user_request = len(user_request_input_ids)
            num_avail_token_resp = num_avail_token - num_token_user_request
            if num_avail_token_resp > 0:
                response_input_ids = self.tokenizer(system_response, truncation=True,
                                                    max_length=num_avail_token_resp).input_ids
                system_response = self.tokenizer.decode(response_input_ids, skip_special_tokens=True)
                truncated_user_requests.append(user_requests[i])
                truncated_system_responses.append(system_response)
            else:
                request_input_ids = self.tokenizer(user_requests[i], truncation=True,
                                                   max_length=num_avail_token).input_ids
                user_request = self.tokenizer.decode(request_input_ids, skip_special_tokens=True)
                truncated_user_requests.append(user_request)
                truncated_system_responses.append("")

        prompts = [self.quality_template.format(instruction=truncated_user_requests[i], output=resp) for i, resp in enumerate(truncated_system_responses)]
        
        if self.deployment == "vllm":
            outputs = self.llm.run_get_probs(prompts=prompts,
                                             max_tokens=self.max_tokens,
                                             logprobs=self.logprobs,
                                             )
        else:
            input_ids = self.tokenizer(prompts, padding='longest', truncation=True, return_tensors="pt").to('cuda')
            outputs = self.llm.generate(**input_ids, **self.sampling_params).scores[0]
        
        scores = []
        for output in outputs:
            try:
                score_logits = []
                score_template = np.array([1, 2, 3, 4, 5, 6])
                for k in self.id2score:
                    if self.deployment == "vllm":
                        logprobs_list = output.outputs[0].logprobs[0]
                        if k in logprobs_list:
                            logprob = logprobs_list[k].logprob
                            score_logits.append(logprob)
                        else:
                            score_logits.append(-20.0)
                    else:
                        score_logits.append(output[k].cpu())
                score_logits = np.array(score_logits)
                score_npy = softmax(score_logits, axis=0)
                score_npy = score_npy * score_template

                score_npy = np.sum(score_npy, axis=0)
                scores.append(score_npy)
            except:
                scores.append(3.0)
        
        return scores

    def run(self, instructions: List[str], responses: List[str], num_exchanges: List[int],
            dataset_name: str, dataset_title: str, output_dir: str,
            request_batch_size: int=512, negative_examples=None,
            categories: List[str] = None):

        to_analyze = zip(instructions, responses)
        to_analyze_idx = [idx for idx, instruction in enumerate(instructions)]
        if categories:
            to_analyze = [(instruction, responses[idx]) for idx, instruction in enumerate(instructions) if
                          categories[idx] in self.categories_to_analyze]
            to_analyze_idx = [idx for idx, instruction in enumerate(instructions) if
                              categories[idx] in self.categories_to_analyze]

        if self.deployment == "hf":
            request_batch_size = 20 # With HF we cannot have high batch size

        _quality_scores = []
        for i in tqdm(range(0, len(to_analyze), request_batch_size)):
            batch_to_analyze = to_analyze[i: i + request_batch_size]
            batch_instructions, batch_responses = list(zip(*batch_to_analyze))

            scores = self.infer_quality(batch_instructions, batch_responses)
            _quality_scores += scores

        # Postprocessing
        quality_scores = []
        for idx in range(len(instructions)):
            if idx in to_analyze_idx:
                quality_scores.append(_quality_scores[to_analyze_idx.index(idx)])
            else:
                quality_scores.append("NaN")

        # Conversational dataset, aggregate metrics per sample
        if num_exchanges:
            quality_scores_aggr = []
            user_msg_idx = 0
            for sample_idx in tqdm(range(0, len(num_exchanges))):
                quality_score_per_sample = 0
                num_responses = 0
                for i in range(num_exchanges[sample_idx]):
                    if responses[user_msg_idx] != "":
                        try:
                            quality_score_per_sample += quality_scores[user_msg_idx]
                            num_responses += 1
                        except:
                            pass
                    else:
                        logger.debug("###", instructions[user_msg_idx], "###", responses[user_msg_idx])
                    user_msg_idx += 1
                if num_responses > 0:
                    quality_score_per_sample = quality_score_per_sample / num_responses
                else:
                    quality_score_per_sample = 0
                quality_scores_aggr.append(quality_score_per_sample)

        # Write scores to file
        with open(f"{output_dir}/{dataset_name}.csv", "w") as fout:
            if num_exchanges:
                for score in quality_scores_aggr:
                    fout.write(f"{score}\n")
            else:
                for score in quality_scores:
                    fout.write(f"{score}\n")

        logger.info(f"### Quality scores are written to {output_dir}/{dataset_name}.csv")

        if num_exchanges:
            return quality_scores_aggr
        else:
            return quality_scores

    def plot(self, scores: List[float], dataset_name: str, dataset_title: str, output_dir: str, categories: List[str]):
        # Plot the histogram
        min_ylim = 1.0
        max_ylim = 6.0
        plot_histogram(f"{output_dir}/{dataset_name}.png", dataset_title + " (quality)", scores,
                        min_ylim, max_ylim)
        if categories:
            plot_histogram_per_category(f"{output_dir}/{dataset_name}_category.png",
                                        dataset_title + " (quality per category)",
                                        scores, categories)

        logger.info(f"### Quality scores plot is written to {output_dir}/{dataset_name}.png")

    def unload_llm(self):
        if self.deployment == "vllm":
            self.llm.unload_llm()
        elif self.deployment == "hf":
            del self.llm
        del self.tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        logger.info(f"### Quality model is unloaded.")

