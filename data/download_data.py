import json
import requests
import random
from datasets import load_dataset

random.seed(42)

# Information for creating unique data IDs
version = "1"
language = "en"
datapoint_id_pattern = "{dataset_id}_v{version}_{language}_{datapoint_id}"

# Define datasets to download and their specific settings
DATASETS = {
            "alpaca_gpt4":                      {"hf_name": "vicgalle/alpaca-gpt4",
                                                 "input_key": "input"}, 
            
            "open_math_instruct_2":             {"hf_name": "nvidia/OpenMathInstruct-2",
                                                 "hf_kwargs": {"split": "train_2M",
                                                             "streaming": True},
                                                 "preprocessing": lambda x: x.filter(lambda example: "augmented" not in example["problem_source"]),
                                                 "instruction_key": "problem",
                                                 "output_key": "generated_solution"}, 
            
            "flan_v2_90k":                      {"hf_name": "ai2-adapt-dev/flan_v2_converted"},
            
            "sharegpt_en":                      {"hf_name": "theblackcat102/sharegpt-english",
                                                 "messages_key": "conversations",
                                                 "role_key": "user",
                                                 "content_key": "text"}, 
            
            "wizardlm_evol_instruct":           {"hf_name": "WizardLMTeam/WizardLM_evol_instruct_V2_196k",
                                                 "messages_key": "conversations",
                                                 "role_key": "from",
                                                 "content_key": "value"},
            
            "emnlp25_200k_agentinst_random":    {"hf_name": "microsoft/orca-agentinstruct-1M-v1",
                                                 "preprocessing": lambda x: random.sample(x, 200000)}, 
            
            "ifeval_like_5k":                   {"hf_name": "HuggingFaceH4/ifeval-like-data"},

            
            "ultrainteract_coding":             {"hf_name": "openbmb/UltraInteract_sft",
                                                 "output_key": "response",
                                                 "preprocessing": lambda x: x.filter(lambda example: example["task"] == "Coding")},
            
            "tulu_3_sft_mixture_0225":          {"hf_name": "allenai/tulu-3-sft-mixture-0225"},

            "daring_anteater":                  {"hf_name": "nvidia/Daring-Anteater",
                                                 "messages_key": "conversations",
                                                 "role_key": "from",
                                                 "content_key": "value"}, 
            
            "conifer_v1":                       {"hf_name": "ConiferLM/Conifer",
                                                 "hf_kwargs": {"split": "train_sft"}}, 
            
            "numina_math_cot_v1":               {"hf_name": "AI-MO/NuminaMath-CoT",
                                                 "preprocessing": lambda x: x.select(range(250000))}, 
            
            "longform":                         {"hf_name":"akoksal/LongForm",
                                                 "instruction_key": "input",}, 
            
            "dolly_15k":                        {"hf_name": "databricks/databricks-dolly-15k",
                                                 "input_key": "context",
                                                 "output_key": "response"},
            
            "alpaca":                           {"url": "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json",
                                                 "input_key": "input"}, 
}

# Helper function for data conversion
def convert_to_chat_format(data):
    """Convert data to chat format"""
    converted = []
    for sample in data:
        if "messages" in sample:
            sample.pop("messages")
        instruction = sample.pop("instruction")
        response = sample.pop("output")
        converted.append({"messages":
                            [{
                                "content": instruction,
                                "role": "user"
                                },
                                {"content": response,
                                "role": "assistant"
                                }],
                        **sample
                        })
    return converted

# Download and process datasets
for dataset_name, dataset_info in DATASETS.items():
    print(f"Processing dataset: {dataset_name}...")
    
    if "hf_name"  in dataset_info:
        # Load dataset
        if "emnlp25_200k_agentinst_random" in dataset_name:
            data = []
            dataset = load_dataset(dataset_info["hf_name"])
            for key, item in dataset.items():
                data += [{"messages": json.loads(x['messages'])} for x in item]
        else:
            hf_kwargs = dataset_info.get("hf_kwargs", {"split": "train"})
            data = load_dataset(dataset_info["hf_name"], **hf_kwargs)
        
        # Uniformize column names
        if "instruction_key" in dataset_info:
            data = data.rename_column(dataset_info["instruction_key"], "instruction")
        if "output_key" in dataset_info:
            data = data.rename_column(dataset_info["output_key"], "output")
        if "input_key" in dataset_info:
            data = data.map(lambda x: {**x,"instruction": (x["instruction"] + "\n" + x[dataset_info["input_key"]]).strip(),})
        if "messages_key" in dataset_info:
            data = data.rename_column(dataset_info["messages_key"], "messages")
            for sample in data:
                for i, message in enumerate(sample["messages"]):
                    if dataset_info["role_key"]:
                        message["role"] = ["user", "assistant"][i % 2]
                    if dataset_info["content_key"]:
                        message["content"] = message.pop(dataset_info["content_key"])
                        
        # Preprocess
        if "preprocessing" in dataset_info:
            data = dataset_info["preprocessing"](data)
        
        # Convert to list of dicts
        data = [x for x in data]
    
    elif "url" in dataset_info:
        # Download data
        response = requests.get(dataset_info["url"])
    
        # Convert to list of dicts
        if response.status_code == 200:
            data = response.json()
            
        # Uniformize column names
        if "input_key" in dataset_info:
            data = [{**sample,"instruction": (sample["instruction"] + "\n" + sample[dataset_info["input_key"]]).strip()} for sample in data]
       
    # Add unique data IDs
    data = [{**sample, "data_id": datapoint_id_pattern.format(dataset_id=dataset_name.replace("_", "-"), version=version, language=language, datapoint_id=i)} for i, sample in enumerate(data)]
    
    # Convert to chat format
    if "messages" not in data[0].keys():
        data = convert_to_chat_format(data)
        
    # Keep only relevant columns
    data = [{"messages": sample["messages"], "data_id": sample["data_id"]} for sample in data]
        
    
    output_path = f"{dataset_name}.jsonl"
    with open(output_path, 'w') as file:
        for line in data:
            _ = file.write(json.dumps(line) + '\n')
    
