# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import torch
import pandas as pd
import datetime
from typing import TYPE_CHECKING, Literal, Optional, Tuple, Union

from datasets import Dataset, load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, PreTrainedModel, PreTrainedTokenizer
from dataclasses import dataclass
from accelerate import PartialState
from multiprocess import set_start_method

from trl import (
    SFTConfig,
    SFTTrainer,
    ModelConfig,
    ScriptArguments,
    get_peft_config,
    get_kbit_device_map,
    get_quantization_config,
    extract_prompt
)

set_start_method("spawn")
torch.distributed.init_process_group(timeout=datetime.timedelta(seconds=9000))

def add_columns(example):
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(rank % torch.cuda.device_count())
    msg = example.pop("messages")
    example["chosen"] = msg
    example["rejected"] = msg
    return example

def rename_columns(example):
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(rank % torch.cuda.device_count())
    completion = example.pop("chosen")
    example["completion"] = completion
    example.pop("rejected")
    return example

def load_data(dataset_name, split=None, load_hf_data_from_disk=False):
    if load_hf_data_from_disk:
        return load_from_disk(dataset_name)
    if os.path.isfile(dataset_name):
        df = pd.read_json(dataset_name, lines=True)
        # dataset should be in one of the supported formats
        # https://huggingface.co/docs/trl/en/sft_trainer#dataset-format-support
        if 'messages' in df.columns:
            df = df[['messages']]
        elif 'prompt' in df.columns and 'completion' in df.columns:
            df = df[['prompt', 'completion']]
        else:
            raise ValueError('Unsupported data format, see https://huggingface.co/docs/trl/en/sft_trainer#dataset-format-support')
        if split:
            dataset = Dataset.from_pandas(df).train_test_split(test_size=0.03, seed=42)
            return dataset[split]
        else:
            return Dataset.from_pandas(df)
    else:
        return load_dataset(dataset_name, split=split)


@dataclass
class CustomArgs:
    """
    Custom arguments for our personal use case.

    load_data_from_disk (`bool`, *optional*, defaults to `False`):
        Whether to load hf dataset from disk.

    """

    load_hf_data_from_disk: bool = False
    dataset_name_eval: str = None
    spectrum_parameters: str = None
    tokenizer_name_or_path: str = None
    completion_only: bool = False

@dataclass
class ChatMlSpecialTokens:
    """Dataclass for special tokens used in ChatML, including system, user, assistant, bos, eos, and pad tokens."""

    bos_token: str = "<|im_start|>"
    eos_token: str = "<|im_end|>"
    pad_token: str = "<|im_end|>"

    @property
    def system(self):
        return f"{self.bos_token}system"

    @property
    def user(self):
        return f"{self.bos_token}user"

    @property
    def assistant(self):
        return f"{self.bos_token}assistant"

    @property
    def chat_template(self):
        return (
            "{% for message in messages %}"
            f"{{{{'{self.bos_token}' + message['role'] + '\n' + message['content'] + '{self.eos_token}' + '\n'}}}}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            f"{{{{ '{self.assistant}\n' }}}}"
            "{% endif %}"
        )

@dataclass
class LLamaSpecialTokens:
    """Dataclass for special tokens used in LLama, including system, user, assistant, bos, eos, and pad tokens."""

    bos_token: str = "<s>"
    eos_token: str = "</s>"
    pad_token: str = "<pad>"

    @property
    def chat_template(self):
        return (
            "{% if messages[0]['role'] == 'system' %}"
            "{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}"
            "{% set loop_messages = messages %}{% set system_message = false %}{% endif %}"
            "{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
            "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}"
            "{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\n' + system_message + '\n<</SYS>>\n\n' + message['content'] %}"
            "{% else %}{% set content = message['content'] %}{% endif %}"
            "{% if message['role'] == 'user' %}"
            f"{{{{'{self.bos_token}' + '[INST] ' + content.strip() + ' [/INST]'}}}}"
            "{% elif message['role'] == 'assistant' %}"
            f"{{{{' ' + content.strip() + ' ' + '{self.eos_token}'}}}}"
            "{% endif %}{% endfor %}"
        )

FORMAT_MAPPING = {"chatml": ChatMlSpecialTokens,
                  "llama": LLamaSpecialTokens,}

def setup_chat_format(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    format: Optional[Literal["chatml", "llama"]] = "chatml",
    resize_to_multiple_of: Optional[int] = None,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Setup chat format by adding special tokens to the tokenizer, setting the correct format, and extending the embedding layer of the model based on the new special tokens.

    If the model already has a chat template, this will throw an error. If you want to overwrite it, please set `tokenizer.chat_template` to `None`.

    Args:
        model (`~transformers.PreTrainedModel`): The model to be modified.
        tokenizer (`~transformers.PreTrainedTokenizer`): The tokenizer to be modified.
        format (`Optional[Literal["chatml"]]`): The format to be set. Defaults to "chatml".
        resize_to_multiple_of (`Optional[int]`): Number to resize the embedding layer to. Defaults to None.

    Returns:
        model (`~transformers.PreTrainedModel`): The modified model.
        tokenizer (`~transformers.PreTrainedTokenizer`): The modified tokenizer.
    """
    # check if model already had a chat template
    if tokenizer.chat_template is not None:
        raise ValueError(
            "Chat template is already added to the tokenizer. If you want to overwrite it, please set it to None"
        )

    # check if format available and retrieve
    if format not in FORMAT_MAPPING:
        raise ValueError(f"Format {format} not available. Please use one of {FORMAT_MAPPING.keys()}")

    chat_format = FORMAT_MAPPING[format]()

    # set special tokens and them
    tokenizer.eos_token = chat_format.eos_token
    tokenizer.pad_token = chat_format.pad_token
    # tokenizer.bos_token = chat_format.bos_token   # ChatML bos_token will not be regarded as a token

    if "additional_special_tokens" in tokenizer.special_tokens_map_extended:
        tokens_to_add = [special_token for special_token in [chat_format.eos_token, chat_format.pad_token]
                         if special_token not in tokenizer.special_tokens_map_extended["additional_special_tokens"]]
        added_tokens = tokenizer.add_special_tokens(
            {
                "additional_special_tokens": tokenizer.special_tokens_map_extended["additional_special_tokens"] +
                                             tokens_to_add
            }
        )
    else:
        tokens_to_add = [special_token for special_token in [chat_format.eos_token, chat_format.pad_token]]
        added_tokens = tokenizer.add_special_tokens(
            {
                "additional_special_tokens": tokens_to_add
            }
        )

    # set chat format for tokenizer
    tokenizer.chat_template = chat_format.chat_template

    model.resize_token_embeddings(
        len(tokenizer), pad_to_multiple_of=resize_to_multiple_of if resize_to_multiple_of is not None else None
    )

    # Update the model config to use the new eos & bos tokens
    if getattr(model, "config", None) is not None:
        # model.config.bos_token_id = tokenizer.bos_token_id    # ChatML bos_token will not be regarded as a token
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
    # Update the generation config to use the new eos & bos token
    if getattr(model, "generation_config", None) is not None:
        # model.generation_config.bos_token_id = tokenizer.bos_token_id     # ChatML bos_token will not be regarded as a token
        model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, SFTConfig, ModelConfig, CustomArgs))
    script_args, training_args, model_args, custom_args = parser.parse_args_into_dataclasses()
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)

    device_string = PartialState().process_index
    device_map = {'': device_string}

    model_kwargs = dict(
        revision=model_args.model_revision,
        torch_dtype=torch.bfloat16,
        attn_implementation=model_args.attn_implementation,
        use_cache=False, # if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    # Load a pretrained model
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, **model_kwargs
    )

    if custom_args.tokenizer_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            custom_args.tokenizer_name_or_path, trust_remote_code=model_args.trust_remote_code,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code,
        )

    if "Mistral" in model_args.model_name_or_path:
        # Remove original chat template
        tokenizer.chat_template = None
        tokenizer.default_chat_template = None
        model, tokenizer = setup_chat_format(model=model, tokenizer=tokenizer, format="llama")

    print("### Model and tokenizer are loaded.")

    # Load the training and eval dataset
    if custom_args.dataset_name_eval:
        train_data = load_data(script_args.dataset_name, split=None,
                               load_hf_data_from_disk=custom_args.load_hf_data_from_disk)
        eval_data = load_data(custom_args.dataset_name_eval, split=None,
                              load_hf_data_from_disk=custom_args.load_hf_data_from_disk)
    else:
        train_data = load_data(script_args.dataset_name, split=script_args.dataset_train_split,
                               load_hf_data_from_disk=custom_args.load_hf_data_from_disk)
        eval_data = load_data(script_args.dataset_name, split=script_args.dataset_test_split,
                           load_hf_data_from_disk=custom_args.load_hf_data_from_disk)
        
    print("### Dataset is loaded.")

    if "Qwen" in model_args.model_name_or_path or "SmolLM" in model_args.model_name_or_path:
        training_args.chat_template_path="HuggingFaceTB/SmolLM3-3B"
    elif "Llama-3" in model_args.model_name_or_path:
        training_args.chat_template_path="meta-llama/Meta-Llama-3-8B-Instruct"
    elif "Falcon3" in model_args.model_name_or_path:
        training_args.chat_template_path="tiiuae/Falcon3-10B-Instruct"

    # Convert dataset into prompt-completion dataset
    print("Converting dataset into prompt-completion format...")
    train_data = train_data.map(add_columns, batched=True)
    eval_data = eval_data.map(add_columns, batched=True)
    train_data = train_data.map(extract_prompt).map(rename_columns, batched=True)
    eval_data = eval_data.map(extract_prompt).map(rename_columns, batched=True)

    # Initialize the SFT trainer
    print("Running the SFT Trainer...")
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        peft_config=get_peft_config(model_args),
    )

    # Train and push the model to the Hub

    # trainer.evaluate()
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)

