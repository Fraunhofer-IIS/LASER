import chromadb
from chromadb.utils import embedding_functions
import argparse
import json
from tqdm import tqdm
from typing import List


def add_to_collection(collection, documents: List[str]):
    if not documents: return
    count = collection.count()
    doc_ids = [str(i+count) for i in range(len(documents))]
    collection.add(
        documents=documents,
        ids=doc_ids
    )

def populate_collection(collection, documents: List[str], batch_size: int):
    for i in tqdm(range(0, len(documents), batch_size)):
        batch_documents = documents[i: i + batch_size]
        collection.add(
            documents=batch_documents,
            ids=[str(i) for i in range(i, i + len(batch_documents))]
        )

def create_collection(collection_name: str, embedding_model: str = "sentence-transformers/all-roberta-large-v1", device: str = "cuda"):
    # check whether the collection exists
    client = chromadb.Client()
    try:
        collection = client.get_collection(name=collection_name)
        client.delete_collection(name=collection_name)
    except Exception as ex:
        pass

    embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=embedding_model,
        device=device,
        trust_remote_code=True
    )

    # create collection
    collection = client.create_collection(
        name=collection_name,
        embedding_function=embedder,
        metadata={"hnsw:space": "cosine", "hnsw:search_ef": 8000, "hnsw:M": 32}
    )
    return collection


def query_collection(collection, query_text: str, limit: int):
    results = collection.query(
        query_texts=[query_text],
        n_results=limit,
    )
    distances = results['distances'][0]
    similar_documents = results['documents'][0]
    return similar_documents, distances

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--machine_instructions_path",
        type=str,
        help="The path to the machine generated instructions.",
    )
    parser.add_argument(
        "--seed_tasks_path",
        type=str,
        help="The path to the human written data.",
    )
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()
    seed_tasks = [json.loads(l) for l in open(args.seed_tasks_path, "r")]
    seed_instructions = [t["instruction"] for t in seed_tasks]
    machine_tasks = [json.loads(l) for l in open(args.machine_instructions_path, "r")]
    machine_instructions = [t["instruction"] for t in machine_tasks]
    documents = seed_instructions + machine_instructions

    collection = create_collection(collection_name="test_collection")
    print(len(documents))
    populate_collection(collection=collection, documents=documents, batch_size=10)
    print(collection.count())
    
    similar_documents, distances = query_collection(collection=collection, query_text=machine_instructions[10], limit=10)
    print(similar_documents, distances)
    


