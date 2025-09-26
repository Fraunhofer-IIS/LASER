import logging
import sys
from typing import List
from tqdm import tqdm
from analyzer.utils import plot_categories
from setfit import SetFitModel
from collections import Counter
from transformers import AutoTokenizer

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

class TaskClassifier(object):

    COLOR_MAP = {
        'Brainstorming': 'C0',
        'Coding': 'C1',
        'Extraction': 'C2',
        'Factual QA': 'C3',
        'Generation': 'C4',
        'Math': 'C5',
        'Reasoning': 'C6',
    }

    def __init__(self, optimum=False):
        self.model_path = "IIS-NLP-internal/sigma-cls"
        self.setfit_model = SetFitModel.from_pretrained(self.model_path, trust_remote_code=True)

        if optimum:
            from optimum.onnxruntime import ORTModelForFeatureExtraction
            from setfit.exporters.utils import mean_pooling
            class OnnxSetFitModel:
                def __init__(self, ort_model, tokenizer, model_head):
                    self.ort_model = ort_model
                    self.tokenizer = tokenizer
                    self.model_head = model_head

                def predict(self, inputs):
                    encoded_inputs = self.tokenizer(
                        inputs, padding=True, truncation=True, return_tensors="pt"
                    ).to(self.ort_model.device)

                    outputs = self.ort_model(**encoded_inputs)
                    embeddings = mean_pooling(
                        outputs["last_hidden_state"], encoded_inputs["attention_mask"]
                    )
                    return self.model_head.predict(embeddings.cpu())

                def __call__(self, inputs):
                    return self.predict(inputs)

            # Load model from HuggingFace Hub
            tokenizer = AutoTokenizer.from_pretrained(self.model_path, model_max_length=2048)
            ort_model = ORTModelForFeatureExtraction.from_pretrained(self.model_path,
                                                                     export=True,
                                                                     trust_remote_code=True,
                                                                     provider="CUDAExecutionProvider")
            self.model = OnnxSetFitModel(ort_model, tokenizer, self.setfit_model.model_head)

        else:
            self.model = self.setfit_model


    def run(self, instructions: List[str], responses: List[str], num_exchanges: List[int],
            dataset_name: str, dataset_title: str, output_dir: str,
            request_batch_size: int=512, negative_examples=None,
            categories: List[str] = None):

        predicted_categories = self.model.predict(instructions, show_progress_bar=True)

        # Conversational dataset, aggregate categories per sample
        if num_exchanges:
            predicted_categories_aggr = []
            user_msg_idx = 0
            for sample_idx in tqdm(range(0, len(num_exchanges))):
                categories_per_sample = []
                for i in range(num_exchanges[sample_idx]):
                    categories_per_sample.append(predicted_categories[user_msg_idx])
                    user_msg_idx += 1
                most_frequent_category = [word for word, word_count in Counter(categories_per_sample).most_common(1)][0]
                predicted_categories_aggr.append(most_frequent_category)

        # Write categories to file
        with open(f"{output_dir}/{dataset_name}.csv", "w") as fout, \
                open(f"{output_dir}/{dataset_name}_aggr.csv", "w") as faggr:
            if num_exchanges:
                for category in predicted_categories_aggr:
                    faggr.write(f"{category}\n")
            else:
                for category in predicted_categories:
                    faggr.write(f"{category}\n")
            for category in predicted_categories:
                fout.write(f"{category}\n")

        logger.info(f"### Instruction categories are written to {output_dir}/{dataset_name}.csv")

        if num_exchanges:
            return predicted_categories_aggr
        else:
            return predicted_categories

    def plot(self, predicted_categories: List[str], dataset_name: str, dataset_title: str, output_dir: str, categories: List[str]):
        # Plot the pie chart
        label_stats = {}
        for label in predicted_categories:
            if label not in label_stats: label_stats[label] = 0
            label_stats[label] += 1
        labels = sorted(list(label_stats.keys()))
        x = [label_stats[label] for label in labels]
        plot_categories(f"{output_dir}/{dataset_name}.png", dataset_title + " (category)", x, labels, self.COLOR_MAP)

        logger.info(f"### Instruction categories plot is written to {output_dir}/{dataset_name}.png")
