import logging
import sys
from typing import List
import torch
import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm
from analyzer.utils import plot_histogram, plot_histogram_per_category

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingDistance(object):

    def __init__(self, embedding_model_path):
        self.embedding_model = embedding_model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.client = chromadb.Client()

    def populate_collection(self, collection, documents: List[str], batch_size: int):
        for i in tqdm(range(0, len(documents), batch_size)):
            batch_documents = documents[i: i + batch_size]
            collection.add(
                documents=batch_documents,
                ids=[str(i) for i in range(i, i + len(batch_documents))]
            )

    def create_collection(self, collection_name: str):
        # check whether the collection exists
        try:
            collection = self.client.get_collection(name=collection_name)
            self.client.delete_collection(name=collection_name)
        except Exception as ex:
            pass

        embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.embedding_model,
            device=self.device,
            trust_remote_code=True
        )

        # create collection
        collection = self.client.create_collection(
            name=collection_name,
            embedding_function=embedder,
            metadata={"hnsw:space": "cosine", "hnsw:search_ef": 8000, "hnsw:M": 32}
        )
        return collection

    def run(self, instructions: List[str], responses: List[str], num_exchanges: List[int],
            dataset_name: str, dataset_title: str, output_dir: str,
            request_batch_size: int=16, negative_examples=None,
            categories: List[str] = None):
        logger.info(f"Creating Collection : {dataset_name}")
        self.collection = self.create_collection(collection_name=dataset_name)

        logger.info(f"Populating Collection : {dataset_name}")
        self.populate_collection(collection=self.collection, documents=instructions, batch_size=request_batch_size)

        nearest_distances = []
        for i in tqdm(range(0, len(instructions), request_batch_size)):
            batch_instructions = instructions[i: i + request_batch_size]
            results = self.collection.query(
                query_texts=batch_instructions,  # Chroma will embed this for you
                n_results=2  # how many results to return
            )
            for doc_index, doc in enumerate(batch_instructions):
                distances = [d for d in results['distances'][doc_index]]
                similar_documents = results['documents'][doc_index]
                if doc.strip() == similar_documents[0].strip():
                    nearest_distances.append(distances[1])
                else:
                    nearest_distances.append(distances[0])

        # Conversational dataset, aggregate metrics per sample
        if num_exchanges:
            nearest_distances_aggr = []
            user_msg_idx = 0
            for sample_idx in tqdm(range(0, len(num_exchanges))):
                nearest_distance_per_sample = 0
                for i in range(num_exchanges[sample_idx]):
                    nearest_distance_per_sample += nearest_distances[user_msg_idx]
                    user_msg_idx += 1
                nearest_distance_per_sample = nearest_distance_per_sample / num_exchanges[sample_idx]
                nearest_distances_aggr.append(nearest_distance_per_sample)

        # Write scores to file
        with open(f"{output_dir}/{dataset_name}.csv", "w") as fout:
            if num_exchanges:
                for score in nearest_distances_aggr:
                    fout.write(f"{score}\n")
            else:
                for score in nearest_distances:
                    fout.write(f"{score}\n")

        logger.info(f"### Embedding distances are written to {output_dir}/{dataset_name}.csv")

        self.delete_collection(collection_name=dataset_name)

        if num_exchanges:
            return nearest_distances_aggr
        else:
            return nearest_distances

    def plot(self, scores: List[float], dataset_name: str, dataset_title: str, output_dir: str, categories: List[str]):
        # Plot the histogram
        min_ylim = 0.0
        max_ylim = 0.0
        plot_histogram(f"{output_dir}/{dataset_name}.png", dataset_title + " (embedding distance)", scores,
                       min_ylim, max_ylim)
        if categories:
            plot_histogram_per_category(f"{output_dir}/{dataset_name}_category.png",
                                        dataset_title + " (embedding distance per category)",
                                        scores, categories)

    def delete_collection(self, collection_name: str):
        self.client.delete_collection(name=collection_name)