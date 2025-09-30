import os
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from vllm.distributed.parallel_state import destroy_model_parallel
from transformers import AutoTokenizer
import argparse
import logging
from typing import List, Dict
import json
from tqdm import tqdm
from pydantic import BaseModel
#import regex
import gc
import torch

class VLLM():
    def __init__(
            self,
            model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
            num_devices: int = 1,
            max_model_len: int = None,
            **kwargs
    ):
        self.__logger = logging.getLogger(self.__class__.__name__)
        self.__logger.info("Initiating HF Model Loading (via vllm)")

        self.model_path = model_name
        if num_devices > 1:
            self.llm = LLM(model=self.model_path, tensor_parallel_size=num_devices,
                       trust_remote_code=True, max_model_len=max_model_len, **kwargs
                       )  # Create an LLM, multiple devices.
        else:
            self.llm = LLM(model=self.model_path, trust_remote_code=True,
                       max_model_len=max_model_len)  # Create an LLM.

        # self.tokenizer = self.llm.get_tokenizer()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model_name = self.model_path.split("/")[-1]

    def run_chat(
        self, messages: List[Dict],
        **kwargs,
        ) -> str:
        sampling_params = SamplingParams(**kwargs)
        outputs = self.llm.chat(
            messages,
            sampling_params=sampling_params,
        )

        # print(outputs[0].outputs[0].text)
        return outputs[0].outputs[0].text

    def run(self, prompts: List[str],
            **kwargs) -> List[str]:
        sampling_params = SamplingParams(**kwargs)
        outputs = self.llm.generate(prompts, sampling_params=sampling_params)

        results = []
        for i, output in enumerate(outputs):
            try:
                output_str = output.outputs[0].text
                results.append(output_str)
            except:
                results.append(None)

        return results

    def run_get_probs(self, prompts: List[str],
                      **kwargs) -> List[str]:
        sampling_params = SamplingParams(**kwargs)
        outputs = self.llm.generate(prompts, sampling_params=sampling_params)

        return outputs
    
    def make_requests(self, prompt_template: str, samples: List[str]):
        pass

    def unload_llm(self):
        # avoid huggingface/tokenizers process dead lock
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        destroy_model_parallel()
        del self.llm.llm_engine
        del self.llm  # Isn't necessary for releasing memory, but why not
        gc.collect()
        torch.cuda.empty_cache()
        import ray
        ray.shutdown()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt_template",
        type=str,
        help="The input file that contains the prompt template.",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        help="The input file that contains the data samples.",
    )
    parser.add_argument(
        "--model_name",
        default="meta-llama/Llama-3.2-3B-Instruct",
        type=str,
        help="The HuggingFace chat/instruct model name.",
    )
    return parser.parse_args()

def extract_first_json(text: str):
    """Extract the first valid JSON object substring from text without regex."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return text[start:i+1]
                except:
                    return None
    return None


if __name__ == "__main__":
    args = parse_args()

    llm = VLLM(model_name=args.model_name)
    input_file = args.input_file

    ######### As a test case, LLM inference for classifying tasks/instructions #########
    input_file = "../analysis/data/mtbench.jsonl"
    samples = [json.loads(line) for line in open(input_file).readlines()]

    #pattern = regex.compile(r'\{(?:[^{}]|(?R))*\}')
    prompt_template = open("../analysis/template/classify_instruction_template.txt").read()

    class Answer(BaseModel):
        answer: str
        explanation: str
    json_schema = Answer.model_json_schema()
    guided_decoding_params = GuidedDecodingParams(json=json_schema)

    prompts = []
    for sample in tqdm(samples):
        messages = [{"role": "user",
                     "content": prompt_template.format(input=sample['input'])}]
        prompt = llm.tokenizer.apply_chat_template(messages, tokenize=False)
        prompts.append(prompt)

    outputs = llm.run(prompts=prompts, guided_decoding=guided_decoding_params,
                      max_tokens=1024, temperature=0.0, top_p=0.5)
    categories = []
    for completion in outputs:
        try:
            completion = extract_first_json(completion)
            #completion = pattern.findall(completion)[0]
            completion = json.loads(completion)
            category = completion['answer']
            print(">>>", completion, "###", category)
        except:
            category = ""
        categories.append(category)

    output_file = f"{input_file.replace('.jsonl', '')}_{llm.model_name}.jsonl"
    with open(output_file, "w") as fout:
        for i, sample in enumerate(samples):
            sample['category'] = categories[i]
            sample['annotator'] = args.model_name
            fout.write(json.dumps(sample)+'\n')

    ####################################################################################