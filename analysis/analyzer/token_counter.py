import logging
import sys
import os
from transformers import AutoTokenizer
from typing import List, Dict
from analyzer.utils import plot_histogram, plot_histogram_per_category
from tqdm import tqdm

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

class TokenCounter(object):

    def __init__(self, tokenizer_path):
        self.tokenizer_path = tokenizer_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, trust_remote_code=True)
        self.split_dataset = []

    def infer_length(self, user_request: str, system_response: str):
        return (len(self.tokenizer(user_request, return_tensors="pt")['input_ids'][0]),
                len(self.tokenizer(system_response, return_tensors="pt")['input_ids'][0]))

    def run(self, instructions: List[str],  responses: List[str], num_exchanges: List[int],
            dataset_name: str, dataset_title: str, output_dir: str,
            request_batch_size: int=16, negative_examples=None,
            categories: List[str] = None):
        instruction_length = []
        response_length = []
        total_size = 0
        for i in tqdm(range(0, len(instructions))):
            inst_length, resp_length = self.infer_length(instructions[i], responses[i])
            instruction_length.append(inst_length)
            response_length.append(resp_length)
            total_size += inst_length + resp_length

        # Conversational dataset, aggregate metrics per sample
        if num_exchanges:
            instruction_length_aggr = []
            response_length_aggr = []
            user_msg_idx = 0
            for sample_idx in tqdm(range(0, len(num_exchanges))):
                instruction_length_per_sample = 0
                response_length_per_sample = 0
                for i in range(num_exchanges[sample_idx]):
                    instruction_length_per_sample += instruction_length[user_msg_idx]
                    response_length_per_sample += response_length[user_msg_idx]
                    user_msg_idx += 1
                instruction_length_aggr.append(instruction_length_per_sample)
                response_length_aggr.append(response_length_per_sample)

        # logger.info(len(instruction_length_aggr), len(response_length_aggr))

        os.makedirs(f"{output_dir}/instructions", exist_ok=True)
        os.makedirs(f"{output_dir}/responses", exist_ok=True)

        # Write token counts to file
        with open(f"{output_dir}/instructions/{dataset_name}.csv", "w") as fout:
            if num_exchanges:
                for length in instruction_length_aggr:
                    fout.write(f"{length}\n")
            else:
                for length in instruction_length:
                    fout.write(f"{length}\n")

        with open(f"{output_dir}/responses/{dataset_name}.csv", "w") as fout:
            if num_exchanges:
                for length in response_length_aggr:
                    fout.write(f"{length}\n")
            else:
                for length in response_length:
                    fout.write(f"{length}\n")

        with open(f"{output_dir}/{dataset_name}.csv", "w") as fout:
            fout.write(f"{total_size}\n")

        logger.info(f"### Total dataset size: {total_size} tokens ({self.tokenizer_path}) ###")

        if num_exchanges:
            return {"instruction_length": instruction_length_aggr, "response_length": response_length_aggr,
                    "total_size": total_size}
        else:
            return {"instruction_length": instruction_length, "response_length": response_length,
                    "total_size": total_size}

    def plot(self, scores: Dict, dataset_name: str, dataset_title: str, output_dir: str, categories: List[str]):
        # Plot the histogram
        min_ylim = 0.0
        max_ylim = 0.0
        instruction_length = scores["instruction_length"]
        response_length = scores["response_length"]
        plot_histogram(f"{output_dir}/instructions/{dataset_name}.png", dataset_title + " (instruction length)", instruction_length, min_ylim, max_ylim)
        plot_histogram(f"{output_dir}/responses/{dataset_name}.png", dataset_title + " (response length)", response_length, min_ylim, max_ylim)
        if categories:
            plot_histogram_per_category(f"{output_dir}/instructions/{dataset_name}_category.png",
                                        dataset_title + " (instruction length per category)",
                                        instruction_length, categories)
            plot_histogram_per_category(f"{output_dir}/responses/{dataset_name}_category.png",
                                        dataset_title + " (response length per category)",
                                        response_length, categories)


