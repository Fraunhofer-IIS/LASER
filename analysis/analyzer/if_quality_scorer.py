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
import re
import gc
import torch

from analyzer.utils import count_relation_mapping, number_mapping, frequency_mapping
from analyzer.utils import check_word_length, check_sentence_length, check_paragraph_length, negation_exists

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


class ConstraintAnswer(BaseModel):
    answer: Dict[str, str]

class QuestionAnswer(BaseModel):
    answer: Dict[str, bool]

class AnalyzeAnswer(BaseModel):
    score: int

class IFQualityScorer(object):

    def __init__(self, llm_annotator: str = "Qwen/Qwen3-14B", deployment: str = "vllm", num_devices: int = 1, max_model_len: int = 8192):
        self.model_name = llm_annotator
        self.deployment = deployment
        self.max_model_len = max_model_len
        self.llm = VLLM(model_name=self.model_name, num_devices=num_devices, max_model_len=max_model_len,
                        gpu_memory_utilization=0.5)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.constraint_template = open("template/if_quality_constraint_template.txt").read()
        self.question_template = open("template/if_quality_question_template.txt").read()
        self.analyze_template = open("template/if_quality_analyze_template.txt").read()

        self.guided_decoding_params_constraint = GuidedDecodingParams(json=ConstraintAnswer.model_json_schema())
        self.guided_decoding_params_question = GuidedDecodingParams(json=QuestionAnswer.model_json_schema())
        self.guided_decoding_params_analyze = GuidedDecodingParams(json=AnalyzeAnswer.model_json_schema())

        self.categories_to_analyze = ["Generation", "Brainstorming"]

    def clean_constraints(self, completion):
        considered_constraints = ["letter_case", "placeholder_and_postscript", "repeat_prompt",
                                  "output_combination", "choose_output", "output_format",
                                  "keyword_included", "keyword_avoided", "keyword_frequency",
                                  "language", "length", "punctuation", "start_and_ending",
                                  # "writing_style", "writing_type", "topic"
                                  ]
        constraints_expressed = [(completion[const], const) for const in completion.keys()
                                 if const != "" and const != "no" and const != "yes"
                                 and completion[const] in considered_constraints
                                 ]
        constraints_filtered = []
        constraint_expressions = [exp for (const, exp) in constraints_expressed]
        for (const, exp) in constraints_expressed:
            # Identify constraints that are subsets of another, and filter them out
            is_subset = [(exp in expr and exp != expr) for expr in constraint_expressions]
            if True not in is_subset:
                constraints_filtered.append((const, exp))

        return constraints_filtered

    def postprocess_constraints(self, constraints_expressed):
        constraints_filtered = []
        output_options = []
        for (const, exp) in constraints_expressed:
            if const == "keyword_included" or const == "keyword_avoided":
                if "\"" in exp or "'" in exp:
                    exp = exp.replace("n't ", "nt ").replace("'s ", "s ")
                    keywords = re.findall('"([^"]*)"', exp) + re.findall("'([^']*)'", exp)
                    keywords = [key for key in keywords if key != ""]
                    if keywords: constraints_filtered.append((const, "; ".join(keywords)))
            elif const == "choose_output":
                output_options.append(exp)
            elif const == "start_and_ending":
                if (" end " in exp or "End" in exp or " finish " in exp or "Finish" in exp) and "\"" in exp:
                    keywords = re.findall('"([^"]*)"', exp)
                    if keywords and keywords[0].count(" ") > 1:
                        constraints_filtered.append((const, "; ".join(["ending", keywords[0]])))
                elif (" start " in exp or "Start" in exp) and "\"" in exp:
                    keywords = re.findall('"([^"]*)"', exp)
                    if keywords and keywords[0].count(" ") > 1:
                        constraints_filtered.append((const, "; ".join(["start", keywords[0]])))
                else:
                    constraints_filtered.append((const, exp))
            else:
                constraints_filtered.append((const, exp))

        if output_options:
            constraints_filtered.append(("choose_output", "; ".join(output_options)))

        return constraints_filtered

    def extract_constraints(self, user_requests: List[str]):
        max_new_tokens = 1024
        prompts = []
        for i, instruction in enumerate(user_requests):
            content = self.constraint_template.format(instruction=instruction)
            # Truncate input text
            content_input_ids = self.tokenizer(content, truncation=True,
                                                max_length=self.max_model_len-max_new_tokens-100).input_ids
            content = self.tokenizer.decode(content_input_ids, skip_special_tokens=True)

            messages = [{"role": "user",
                         "content": content}]
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False,
                                                        add_generation_prompt=True)
            prompts.append(prompt)

        expressed_constraints = []
        if prompts:
            outputs = self.llm.run(prompts=prompts, guided_decoding=self.guided_decoding_params_constraint,
                                   max_tokens=max_new_tokens, temperature=0.0, top_p=0.5)

            json_pattern = regex.compile(r'\{(?:[^{}]|(?R))*\}')

            for i, instruction in enumerate(user_requests):
                try:
                    completion = outputs[i]
                    completion = json_pattern.findall(completion)[0]
                    completion = json.loads(completion)['answer']
                    constraints_found = self.clean_constraints(completion)
                    constraints_found = self.postprocess_constraints(constraints_found)
                except:
                    constraints_found = []
                expressed_constraints.append(constraints_found)
        else:
            for i in range(len(user_requests)):
                expressed_constraints.append([])

        return expressed_constraints

    def analyze_constraints(self, user_requests: List[str], system_responses: List[str], found_constraints: List[List]):
        max_new_tokens = 1024
        prompts = []
        prompts_with_constraints = [i for i, response in enumerate(system_responses) if found_constraints[i]]
        numbered_constraints = []
        for i, response in enumerate(system_responses):
            if found_constraints[i]:
                questions = []
                constraints_expressed = {}
                for i, (cons, exp) in enumerate(found_constraints[i]):
                    if cons == "choose_output":
                        questions.append(f"{i+1}. {cons}: Does the following text (after 'ASSISTANT:') contain one of the following phrase: {exp}?")
                    else:
                        questions.append(f"{i+1}. {cons}: Does the following text (after 'ASSISTANT:') follow the {cons.replace('_', ' ')} constraint of '{exp}'?")
                    constraints_expressed[f"{i+1}"] = (cons, exp)

                content = self.question_template.format(output=response, questions='\n'.join(questions))
                # Truncate input text
                content_input_ids = self.tokenizer(content, truncation=True,
                                                   max_length=self.max_model_len - max_new_tokens - 100).input_ids
                content = self.tokenizer.decode(content_input_ids, skip_special_tokens=True)

                messages = [{"role": "user",
                             "content": content}]
                prompt = self.tokenizer.apply_chat_template(messages, tokenize=False,
                                                            add_generation_prompt=True)
                prompts.append(prompt)
                numbered_constraints.append(constraints_expressed)

        all_checked_constraints = []
        all_evaluated_constraints = []
        if prompts:
            outputs = self.llm.run(prompts=prompts, guided_decoding=self.guided_decoding_params_question,
                                   max_tokens=max_new_tokens, temperature=0.0, top_p=0.5)
            json_pattern = regex.compile(r'\{(?:[^{}]|(?R))*\}')

            for i, response in enumerate(system_responses):
                if i in prompts_with_constraints:
                    prompt_idx = prompts_with_constraints.index(i)
                    try:
                        completion = outputs[prompt_idx]
                        completion = json_pattern.findall(completion)[0]
                        completion = json.loads(completion)['answer']
                        constraints_evaluated = completion
                    except:
                        constraints_evaluated = {}

                    checked_constraints = [numbered_constraints[prompt_idx][key]
                                           for key in numbered_constraints[prompt_idx] if key in constraints_evaluated]
                    evaluated_constraints = [constraints_evaluated[key] for key in numbered_constraints[prompt_idx]
                                             if key in constraints_evaluated]
                    all_checked_constraints.append(checked_constraints)
                    all_evaluated_constraints.append(evaluated_constraints)
                else:
                    all_checked_constraints.append([])
                    all_evaluated_constraints.append([])
        else:
            for i in range(len(system_responses)):
                all_checked_constraints.append([])
                all_evaluated_constraints.append([])

        return all_checked_constraints, all_evaluated_constraints

    def analyze_responses(self, user_requests: List[str], system_responses: List[str],
                          found_constraints: List[List]):
        max_new_tokens = 100
        prompts = []
        prompts_to_analyze = [i for i, response in enumerate(system_responses)
                              if found_constraints[i] == []]
        for i, instruction in enumerate(user_requests):
            if i in prompts_to_analyze:
                content = self.analyze_template.format(instruction=user_requests[i], output=system_responses[i])
                # Truncate input text
                content_input_ids = self.tokenizer(content, truncation=True,
                                                   max_length=self.max_model_len - max_new_tokens - 100).input_ids
                content = self.tokenizer.decode(content_input_ids, skip_special_tokens=True)

                messages = [{"role": "user",
                            "content": content}]
                prompt = self.tokenizer.apply_chat_template(messages, tokenize=False,
                                                            add_generation_prompt=True)
                prompts.append(prompt)

        all_scores = []
        if prompts:
            outputs = self.llm.run(prompts=prompts, guided_decoding=self.guided_decoding_params_analyze,
                                   max_tokens=max_new_tokens, temperature=0.0, top_p=0.5)
            json_pattern = regex.compile(r'\{(?:[^{}]|(?R))*\}')

            for i, instruction in enumerate(user_requests):
                if i in prompts_to_analyze:
                    prompt_idx = prompts_to_analyze.index(i)
                    try:
                        completion = outputs[prompt_idx]
                        completion = json_pattern.findall(completion)[0]
                        score = json.loads(completion)['score'] / 10
                    except:
                        score = 0.5
                else:
                    score = 0.0
                all_scores.append(score)
        else:
            for i in range(len(user_requests)):
                all_scores.append(0.0)

        return all_scores


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

        _quality_scores = []
        _constraints_found = []
        _constraints_evaluated = []
        for i in tqdm(range(0, len(to_analyze), request_batch_size)):
            batch_to_analyze = to_analyze[i: i + request_batch_size]
            batch_instructions, batch_responses = list(zip(*batch_to_analyze))

            # Extract constraints from user prompt
            expressed_constraints = self.extract_constraints(batch_instructions)

            # Manually evaluate the constraints
            all_checked = []
            all_unchecked = []
            all_evaluated = []
            for j in range(len(batch_instructions)):
                checked, unchecked, evaluated = self.evaluate_constraints(prompt=batch_instructions[j],
                                                                          response=batch_responses[j],
                                                                          constraints_found=expressed_constraints[j])
                all_checked.append(checked)
                all_unchecked.append(unchecked)
                all_evaluated.append(evaluated)

            # LLM evaluation of the constraints
            checked, evaluated = self.analyze_constraints(batch_instructions, batch_responses, all_unchecked)
            all_checked_constraints = [x[0] + x[1] for x in zip(all_checked, checked)]
            all_evaluated_constraints = [x[0] + x[1] for x in zip(all_evaluated, evaluated)]

            num_followed_constraints = [all_evaluated_constraints[j].count(True) for j, checked in enumerate(all_checked_constraints)]
            if_quality_scores = [num_followed_constraints[j] / len(checked)
                                 if len(checked) > 0 else 0.0
                                 for j, checked in enumerate(all_checked_constraints)]

            # Incorporating number of followed constraints in the score
            if_quality_scores_2 = [if_quality_scores[j] * num_followed_constraints[j]
                                 if num_followed_constraints[j] > 0 and num_followed_constraints[j] < 10
                                   else if_quality_scores[j]
                                 for j, checked in enumerate(all_checked_constraints)]

            # LLM evaluation of the responses
            response_quality_scores = self.analyze_responses(batch_instructions, batch_responses,
                                                             all_checked_constraints)
            scores = [sum(x) for idx, x in enumerate(zip(if_quality_scores_2, response_quality_scores))]

            _quality_scores += scores
            _constraints_found += all_checked_constraints
            _constraints_evaluated += all_evaluated_constraints

        # Postprocessing
        quality_scores = []
        constraints_found = []
        constraints_evaluated = []
        for idx in range(len(instructions)):
            if idx in to_analyze_idx:
                quality_scores.append(_quality_scores[to_analyze_idx.index(idx)])
                constraints_found.append(_constraints_found[to_analyze_idx.index(idx)])
                constraints_evaluated.append(_constraints_evaluated[to_analyze_idx.index(idx)])
            else:
                quality_scores.append("NaN")
                constraints_found.append([])
                constraints_evaluated.append([])

        # Conversational dataset, aggregate metrics per sample
        if num_exchanges:
            quality_scores_aggr = []
            constraints_found_aggr = []
            constraints_evaluated_aggr = []
            user_msg_idx = 0
            for sample_idx in tqdm(range(0, len(num_exchanges))):
                quality_score_per_sample = 0
                constraints_found_per_sample = []
                constraints_evaluated_per_sample = []
                num_responses = 0
                for i in range(num_exchanges[sample_idx]):
                    if responses[user_msg_idx] != "":
                        try:
                            quality_score_per_sample += quality_scores[user_msg_idx]
                            constraints_found_per_sample += constraints_found[user_msg_idx]
                            constraints_evaluated_per_sample += constraints_evaluated[user_msg_idx]
                            num_responses += 1
                        except:
                            pass
                    else:
                        logger.debug(f"### {instructions[user_msg_idx]} ### {responses[user_msg_idx]}")
                    user_msg_idx += 1
                if num_responses > 0:
                    quality_score_per_sample = quality_score_per_sample / num_responses
                else:
                    quality_score_per_sample = 0
                quality_scores_aggr.append(quality_score_per_sample)
                constraints_found_aggr.append(constraints_found_per_sample)
                constraints_evaluated_aggr.append(constraints_evaluated_per_sample)

        # Write scores to file
        with open(f"{output_dir}/{dataset_name}.csv", "w") as fout:
            if num_exchanges:
                for score in quality_scores_aggr:
                    fout.write(f"{score}\n")
            else:
                for score in quality_scores:
                    fout.write(f"{score}\n")

        # Write constraints to file
        with open(f"{output_dir}/{dataset_name}.jsonl", "w") as fout:
            if num_exchanges:
                for i, const in enumerate(constraints_found_aggr):
                    constraints = {'constraints_expressed': constraints_found_aggr[i],
                                   'constraints_followed': constraints_evaluated_aggr[i]}
                    fout.write(f"{json.dumps(constraints)}\n")
            else:
                for i, const in enumerate(constraints_found):
                    constraints = {'constraints_expressed': constraints_found[i],
                                   'constraints_followed': constraints_evaluated[i]}
                    fout.write(f"{json.dumps(constraints)}\n")

        logger.info(f"### IF Quality scores are written to {output_dir}/{dataset_name}.csv")

        if num_exchanges:
            return quality_scores_aggr
        else:
            return quality_scores

    def plot(self, scores: List[float], dataset_name: str, dataset_title: str, output_dir: str, categories: List[str]):
        # # Plot the histogram
        # min_ylim = 0.0
        # max_ylim = 1.0
        # plot_histogram(f"{output_dir}/{dataset_name}.png", dataset_title + " (quality)", scores,
        #                 min_ylim, max_ylim)
        #
        # print(f"### IF-Quality scores plot is written to {output_dir}/{dataset_name}.png")
        pass

    def unload_llm(self):
        self.llm.unload_llm()
        del self.tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        logger.info(f"### LLM annotator is unloaded.")

    def evaluate_constraints(self, prompt: str, response: str, constraints_found: List):
        checked_constraints = []
        unchecked_constraints = []
        checked_constraints_evaluated = []

        for (const, exp) in constraints_found:
            if const == "punctuation":
                if negation_exists(exp) and "comma" in exp:
                    checked_constraints.append((const, exp))
                    checked_constraints_evaluated.append("," not in response)
                else:
                    unchecked_constraints.append((const, exp))

            elif const == "keyword_included":
                keywords = exp.split("; ")
                keywords_exist = [keyword in response for keyword in keywords]
                checked_constraints.append((const, exp))
                checked_constraints_evaluated.append(False not in keywords_exist)

            elif const == "keyword_avoided":
                keywords = exp.split("; ")
                keywords_exist = [keyword in response for keyword in keywords]
                checked_constraints.append((const, exp))
                checked_constraints_evaluated.append(True not in keywords_exist)

            elif const == "keyword_frequency":
                words = [w.strip().replace("\"", "").replace("'", "").replace(".", "").replace(",", "")
                         for w in exp.split(" ")]
                frequency = 0
                if "once" in words:
                    frequency = frequency_mapping["once"]
                elif "twice" in words:
                    frequency = frequency_mapping["twice"]
                elif "times" in words:
                    freq_idx = words.index("times")
                    try:
                        frequency = int(words[freq_idx - 1])
                    except:
                        frequency = 0

                relation = ""
                for rel in count_relation_mapping:
                    if rel in exp:
                        relation = count_relation_mapping[rel]
                        break
                if relation == "": relation = "="

                if frequency > 0:
                    if "word" in words or "name" in words:
                        if "word" in words:
                            key_idx = words.index("word")
                        elif "name" in words:
                            key_idx = words.index("name")
                        if key_idx+1 < len(words):
                            repeated_word = words[key_idx + 1]
                            keyword_count = response.count(f"{repeated_word}") + response.count(
                                f"{repeated_word.title()}")
                            if relation == "<":
                                followed = keyword_count < frequency
                            elif relation == ">":
                                followed = keyword_count > frequency
                            if relation == "<=":
                                followed = keyword_count <= frequency
                            elif relation == ">=":
                                followed = keyword_count >= frequency
                            else:  # relation == "=":
                                followed = keyword_count == frequency

                            checked_constraints.append((const, exp))
                            checked_constraints_evaluated.append(followed)
                        else:
                            unchecked_constraints.append((const, exp))
                    elif "letter" in words:
                        letter_idx = words.index("letter")
                        if letter_idx+1 < len(words):
                            repeated_letter = words[letter_idx + 1]
                            letter_count = response.count(repeated_letter)
                            if relation == "<":
                                followed = letter_count < frequency
                            elif relation == ">":
                                followed = letter_count > frequency
                            if relation == "<=":
                                followed = letter_count <= frequency
                            elif relation == ">=":
                                followed = letter_count >= frequency
                            else:  # relation == "=":
                                followed = letter_count == frequency

                            checked_constraints.append((const, exp))
                            checked_constraints_evaluated.append(followed)
                        else:
                            unchecked_constraints.append((const, exp))
                else:
                    unchecked_constraints.append((const, exp))

            elif const == "choose_output":
                keywords = exp.split("; ")
                keywords_exist = [keyword in response for keyword in keywords]
                checked_constraints.append((const, exp))
                checked_constraints_evaluated.append(keywords_exist.count(True) == 1)

            elif const == "length":
                if "paragraph" in exp:
                    check_length = check_paragraph_length(exp, response)
                    if check_length:
                        checked_constraints.append((const, exp))
                        checked_constraints_evaluated.append(check_length)
                    else:
                        unchecked_constraints.append((const, exp))
                elif "sentence" in exp:
                    check_length = check_sentence_length(exp, response)
                    if check_length:
                        checked_constraints.append((const, exp))
                        checked_constraints_evaluated.append(check_length)
                    else:
                        unchecked_constraints.append((const, exp))
                elif "word" in exp:
                    check_length = check_word_length(exp, response)
                    if check_length:
                        checked_constraints.append((const, exp))
                        checked_constraints_evaluated.append(check_length)
                    else:
                        unchecked_constraints.append((const, exp))
                else:
                    unchecked_constraints.append((const, exp))

            elif const == "repeat_prompt":
                checked_constraints.append((const, exp))
                contain_prompt = [sent in response for sent in re.split(r'[\.?!]', prompt)]
                checked_constraints_evaluated.append(True in contain_prompt)

            elif const == "letter_case":
                if "times" in exp or "twice" in exp:
                    unchecked_constraints.append((const, exp))
                elif (negation_exists(exp)) and "capital" in exp:
                    checked_constraints.append((const, exp))
                    checked_constraints_evaluated.append(response.islower())
                elif "lowercase" in exp or "lower case" in exp:
                    checked_constraints.append((const, exp))
                    checked_constraints_evaluated.append(response.islower())
                elif "capital" in exp:
                    checked_constraints.append((const, exp))
                    checked_constraints_evaluated.append(response.isupper())
                else:
                    unchecked_constraints.append((const, exp))

            elif const == "start_and_ending":
                if ("wrap" in exp or "Wrap" in exp) and "double quot" in exp:
                    checked_constraints.append((const, exp))
                    checked_constraints_evaluated.append(response.startswith("\"") and response.endswith("\""))
                elif exp.startswith("start;"):
                    starting_phrase = exp.split("; ")[1]
                    checked_constraints.append((const, exp))
                    checked_constraints_evaluated.append(response.startswith(starting_phrase))
                elif exp.startswith("ending;"):
                    ending_phrase = exp.split("; ")[1]
                    checked_constraints.append((const, exp))
                    checked_constraints_evaluated.append(response.endswith(ending_phrase))
                else:
                    unchecked_constraints.append((const, exp))
            else:
                unchecked_constraints.append((const, exp))

        return checked_constraints, unchecked_constraints, checked_constraints_evaluated

