import logging
import sys
import os
import glob
import statistics
from typing import List
import time

from utils import (load_sft_dataset,  
                    get_analyzer,
                    parse_args
                    )

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

analysis_types = ["complexity", 
                  "quality",
                  "tokens", 
                  "categories_v2",
                  "difficulty_v2", 
                  "process_reward_modelling", 
                  "if_quality",
                  "code_quality",
                  ]

def check_results_exist(analysis_type: str, dataset_name: str, output_dir: str):
    if glob.glob(f"{output_dir}/{dataset_name}.*"):
        if analysis_type == "tokens":
            with open(f"{output_dir}/instructions/{dataset_name}.csv") as finst, \
                    open(f"{output_dir}/responses/{dataset_name}.csv") as fresp:
                results = {
                    "instruction_length": [int(line.strip()) for line in finst.readlines()],
                    "response_length": [int(line.strip()) for line in fresp.readlines()]
                }
        else:
            with open(f"{output_dir}/{dataset_name}.csv") as fscores:
                results = [line.strip() for line in fscores.readlines()]
        return True, results
    else:
        return False, []

def get_dataset_stats(instructions: List[str], responses: List[str], num_exchanges: List[int], dataset_name: str,):
    if num_exchanges:
        logger.info(
            f"### Number of samples {len(num_exchanges)}, "
            f"Number of user messages {len(instructions)}, "
            f"Avg number of turns {statistics.mean(num_exchanges)} ###")
        with open(os.path.join(output_dir, f"{dataset_name}.csv"), "w") as fout:
            fout.write(f"#samples\t{len(num_exchanges)}\n")
            fout.write(f"#user_msgs\t{len(instructions)}\n")
            fout.write(f"avg #turns\t{statistics.mean(num_exchanges)}\n")
    else:
        logger.info(
            f"### Number of samples {len(responses)}, Number of user messages {len(instructions)} ###")
        with open(os.path.join(output_dir, f"{dataset_name}.csv"), "w") as fout:
            fout.write(f"#samples\t{len(responses)}\n")
            fout.write(f"#user_msgs\t{len(instructions)}\n")
            fout.write(f"avg #turns\t1\n")

def get_categories(args, num_exchanges: List[int], dataset_name: str,):
    """
    Get categories, if exist
    """
    categories = None
    categories_aggr = None
    if num_exchanges:
        try:
            with open(f"{os.path.join(args.output_dir, './categories_v2')}/{dataset_name}.csv") as fcats, \
                open(f"{os.path.join(args.output_dir, './categories_v2')}/{dataset_name}_aggr.csv") as faggr:
                categories = [line.strip() for line in fcats.readlines()]
                categories_aggr = [line.strip() for line in faggr.readlines()]
        except:
            pass
    else:
        try:
            with open(f"{os.path.join(args.output_dir, './categories_v2')}/{dataset_name}.csv") as fcats:
                categories = [line.strip() for line in fcats.readlines()]
                categories_aggr = categories
        except:
            pass
    if categories is not None:
        print(f"Successfully loaded categories.")
    else:
        print(f"Failed to load categories. Proceeding without dedicated scoring.")
    return categories, categories_aggr


if __name__ == "__main__":
    start = time.time()

    args = parse_args()

    if args.analysis != "all": 
        analysis_types = [a_type.strip() for a_type in args.analysis.split(",")]

    dataset_names = [d_name.strip() for d_name in args.dataset.split(",")]

    for analysis_type in analysis_types:
        
        analyzer, output_dir = get_analyzer(analysis_type, args)
        os.makedirs(output_dir, exist_ok=True)

        for dataset_name in dataset_names:

            start_dataset = time.time()

            # Load Dataset
            (dataset_title, instructions, responses, num_exchanges) = load_sft_dataset(dataset_name)
            negative_examples = None

            # Check if the analysis result exist already
            result_exists, results = check_results_exist(analysis_type, dataset_name, output_dir)

            # Categories, if exists
            categories, categories_aggr = get_categories(args, num_exchanges, dataset_name)
            
            if not result_exists or args.repeat_analysis:
                logger.info(f"Run {analysis_type} on {dataset_title}...")
                if analysis_type == "dataset_stats":
                    get_dataset_stats(instructions=instructions, 
                                    responses=responses, 
                                    num_exchanges=num_exchanges,
                                    dataset_name=dataset_name
                                    )
                else:
                    # Run the analyzer
                    results = analyzer.run(instructions=instructions, responses=responses, num_exchanges=num_exchanges,
                                            dataset_name=dataset_name, dataset_title=dataset_title, output_dir=output_dir,
                                            request_batch_size=args.request_batch_size, negative_examples=negative_examples,
                                            categories=categories
                                        )

            end_dataset = time.time()
            logger.info(f"### Execution time ({dataset_name}): {end_dataset - start_dataset:.5f} seconds.")

            # Plot the charts
            if analyzer and args.plot:
                analyzer.plot(results, dataset_name, dataset_title, output_dir, categories_aggr)

        if analysis_type == "complexity" or analysis_type == "quality" \
                or analysis_type == "if_quality" or analysis_type == "code_quality":
            analyzer.unload_llm()

    end = time.time()
    logger.info(f"### Execution time: {end - start:.5f} seconds.")

