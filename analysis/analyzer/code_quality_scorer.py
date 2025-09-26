import logging
import sys

# setting path
sys.path.append('..')
sys.path.append('../..')

from llm.vllm_api import VLLM
from vllm.sampling_params import GuidedDecodingParams
from transformers import AutoTokenizer
from analyzer.utils import plot_histogram, plot_histogram_per_category
from typing import List, Dict
from tqdm import tqdm
from pydantic import BaseModel
import regex
import json
import nltk
import math
import gc
import torch

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

class Answer(BaseModel):
    review: str
    final_verdict: str
    code_original: str
    code_revision: str

class CodeQualityScorer(object):

    def __init__(self, llm_annotator: str = "Qwen/Qwen2.5-Coder-14B-Instruct", deployment: str = "vllm",
                 num_devices: int = 1, max_model_len: int = 4096):
        self.model_name = llm_annotator
        self.deployment = deployment
        self.init_llm(num_devices=num_devices, max_model_len=max_model_len)

        self.review_template = open("template/code_revision_template.txt").read()
        self.guided_decoding_params = GuidedDecodingParams(json=Answer.model_json_schema())
        self.categories_to_analyze = ["Coding"]

    def init_llm(self, num_devices, max_model_len):
        self.max_prompt_len = math.floor(max_model_len / 2)
        self.max_model_len = max_model_len
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, model_max_length=self.max_model_len)

        if self.deployment == "hf":
            raise NotImplementedError
        elif self.deployment == "vllm":
            self.llm = VLLM(model_name=self.model_name, num_devices=num_devices, max_model_len=max_model_len)
        else:
            raise NotImplementedError

    def token_edit_levenstein_similarity_normalized(self, text1: List, text2: List) -> float:
        """
        Compute the normalized levenstein distance between two texts.
        """
        import nltk
        # return 1 - nltk.edit_distance(text1, text2) / max(len(text1), len(text2))
        max_len = max(len(text1), len(text2))
        return (max_len - nltk.edit_distance(text1, text2)) / max_len

    def clean_code(self, code_snippet: str):
        cleaned_code = code_snippet.replace('\\n', '\n')
        cleaned_code = cleaned_code.replace('```cpp', '')
        cleaned_code = cleaned_code.replace('```python', '')
        cleaned_code = cleaned_code.replace('```java', '')
        cleaned_code = cleaned_code.replace('```c++', '')
        cleaned_code = cleaned_code.replace('```json', '')
        cleaned_code = cleaned_code.replace('```xml', '')
        cleaned_code = cleaned_code.replace('```', '')

        return cleaned_code

    def code_review(self, user_requests: List[str], system_responses: List[str],
                    review_file_path: str):
        # Truncate input text
        template_input_ids = self.tokenizer(self.review_template).input_ids
        num_token_complexity_template = len(template_input_ids)
        max_prompt_len = self.max_prompt_len
        num_avail_token = (max_prompt_len - num_token_complexity_template)
        truncated_user_requests = []
        truncated_system_responses = []
        request_too_long = []
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
                request_too_long.append(i)
                truncated_user_requests.append("request too long!")
                truncated_system_responses.append("request too long!")

        instructions_to_analyze = [i for i in range(len(user_requests)) if i not in request_too_long]

        prompts = []
        for i, instruction in enumerate(user_requests):
            if i in instructions_to_analyze:
                prompts.append(self.tokenizer.apply_chat_template([{"role": "user",
                                                                  "content": self.review_template.format(instruction=truncated_user_requests[i], output=truncated_system_responses[i])}],
                                                                  tokenize=False))

        code_quality_scores = []
        if prompts:
            outputs = self.llm.run(prompts=prompts,
                               guided_decoding=self.guided_decoding_params,
                               max_tokens=self.max_model_len-self.max_prompt_len,
                               temperature=0.0,
                               top_p=0.5
                               )

            json_pattern = regex.compile(r'\{(?:[^{}]|(?R))*\}')

            for i, instruction in enumerate(user_requests):
                if i in instructions_to_analyze:
                    prompt_idx = instructions_to_analyze.index(i)
                    try:
                        # completion = json_pattern.findall(outputs[i])[0]
                        completion = json.loads(outputs[prompt_idx])

                        code_original = self.clean_code(completion['code_original'])
                        code_revision = self.clean_code(completion['code_revision'])

                        code_original_list = code_original.strip().splitlines()
                        code_revision_list = code_revision.strip().splitlines()

                        functional_correctness = completion['final_verdict'] == "correct"
                        code_review = completion['review']

                        if code_original_list[0] != "no code":
                            distance = nltk.edit_distance(code_original_list, code_revision_list)
                            final_score = self.token_edit_levenstein_similarity_normalized(code_original_list,
                                                                                           code_revision_list)
                            if functional_correctness:
                                if len(code_original_list) > 3 and code_revision_list[0] == "no revision":
                                    final_score = 1.0
                                    code_revision = code_original
                            else:
                                final_score = final_score * 0.5
                        else:
                            distance = 0.0
                            if functional_correctness:
                                final_score = 0.5
                            else:
                                final_score = 0.0

                    except:
                        functional_correctness = True
                        distance = 0.0
                        final_score = 1.0
                        code_review = "no review"
                        code_original = "no code"
                        code_revision = "no revision"

                    code_quality_scores.append(final_score)
                    with open(review_file_path, "a") as fout:
                        fout.write(json.dumps({
                            "code_review": code_review,
                            "code_original": code_original,
                            "code_revision": code_revision,
                            "functional_correctness": functional_correctness,
                            "edit_similarity": distance,
                        }) + '\n')

                else:
                    code_quality_scores.append(0.0)
                    with open(review_file_path, "a") as fout:
                        fout.write(json.dumps({}) + '\n')

            del outputs

        else:
            for i in range(len(user_requests)):
                code_quality_scores.append(0.0)
                with open(review_file_path, "a") as fout:
                    fout.write(json.dumps({}) + '\n')

        del truncated_system_responses
        del prompts
        gc.collect()

        return code_quality_scores

    def run(self, instructions: List[str], responses: List[str], num_exchanges: List[int],
            dataset_name: str, dataset_title: str, output_dir: str,
            num_devices: int=1, request_batch_size: int=512, negative_examples=None,
            categories: List[str] = None):

        to_analyze = zip(instructions, responses)
        to_analyze_idx = [idx for idx, instruction in enumerate(instructions)]
        if categories:
            to_analyze = [(instruction, responses[idx]) for idx, instruction in enumerate(instructions) if
                          categories[idx] in self.categories_to_analyze]
            to_analyze_idx = [idx for idx, instruction in enumerate(instructions) if
                              categories[idx] in self.categories_to_analyze]

        # Rewrite json file as a fresh file
        with open(f"{output_dir}/{dataset_name}.jsonl", "w") as fout:
            fout.write("")

        _code_quality_scores = []
        for i in tqdm(range(0, len(to_analyze), request_batch_size)):
            batch_to_analyze = to_analyze[i: i + request_batch_size]
            batch_instructions, batch_responses = list(zip(*batch_to_analyze))

            # Get code quality scores
            scores = self.code_review(batch_instructions, batch_responses,
                                      f"{output_dir}/{dataset_name}.jsonl")
            _code_quality_scores += scores

        # Postprocessing
        code_quality_scores = []
        for idx in range(len(instructions)):
            if idx in to_analyze_idx:
                code_quality_scores.append(_code_quality_scores[to_analyze_idx.index(idx)])
            else:
                code_quality_scores.append("NaN")

        # Conversational dataset, aggregate metrics per sample
        if num_exchanges:

            code_quality_scores_aggr = []
            user_msg_idx = 0
            for sample_idx in tqdm(range(0, len(num_exchanges))):
                code_quality_score_per_sample = 0
                num_responses = 0
                for i in range(num_exchanges[sample_idx]):
                    if responses[user_msg_idx] != "":
                        try:
                            code_quality_score_per_sample += code_quality_scores[user_msg_idx]
                            num_responses += 1
                        except:
                            pass
                    else:
                        logger.debug(f"### {instructions[user_msg_idx]} ### {responses[user_msg_idx]}")
                    user_msg_idx += 1
                if num_responses > 0:
                    code_quality_score_per_sample = code_quality_score_per_sample / num_responses
                else:
                    code_quality_score_per_sample = 0
                code_quality_scores_aggr.append(code_quality_score_per_sample)

        # Write scores to file
        with open(f"{output_dir}/{dataset_name}.csv", "w") as fout:
            if num_exchanges:
                for score in code_quality_scores_aggr:
                    fout.write(f"{score}\n")
            else:
                for score in code_quality_scores:
                    fout.write(f"{score}\n")

        logger.info(f"### Code Quality scores are written to {output_dir}/{dataset_name}.csv")

        if num_exchanges:
            return code_quality_scores_aggr
        else:
            return code_quality_scores

    def plot(self, scores: List[float], dataset_name: str, dataset_title: str, output_dir: str, categories: List[str]):
        # # Plot the histogram
        # min_ylim = 0.0
        # max_ylim = 1.0
        # plot_histogram(f"{output_dir}/{dataset_name}.png", dataset_title + " (quality)", scores,
        #                 min_ylim, max_ylim)
        #
        # logger.info(f"### IF-Quality scores plot is written to {output_dir}/{dataset_name}.png")
        pass

    def unload_llm(self):
        self.llm.unload_llm()
        del self.tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        logger.info(f"### LLM annotator is unloaded.")
