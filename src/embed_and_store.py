"""
Generates OpenAI embeddings for chunked documents and stores them in a FAISS index.
Requires: OPENAI_API_KEY in env, openai>=1.0.0, faiss-cpu, numpy
"""

import os
import json
import faiss
import numpy as np
from tqdm import tqdm
from datetime import datetime
from openai import OpenAI
import pickle

client = OpenAI()

# Initialize OpenAI client (uses OPENAI_API_KEY from environment)
api_key = os.environ.get("OPENAI_API_KEY")
# client = OpenAI(api_key=api_key)
EMBED_MODEL = "text-embedding-3-small"
INDEX_DIR = "vector_index"
INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")
METADATA_PATH = os.path.join(INDEX_DIR, "metadata.json")


def get_embedding(text):
    """
    Generate a single embedding using OpenAI's v1.x client.
    """
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=[text]
    )
    return response.data[0].embedding


def load_chunks(input_dir):
    """
    Load all chunks from JSON files in a given folder.
    """
    chunks = []
    for fname in os.listdir(input_dir):
        if fname.endswith(".json"):
            with open(os.path.join(input_dir, fname), "r", encoding="utf-8") as f:
                data = json.load(f)
                for chunk in data["chunks"]:
                    chunks.append({
                        "text": chunk["text"],
                        "metadata": {
                            **data["document_metadata"],
                            "chunk_id": chunk["chunk_id"]
                        }
                    })
    return chunks


def build_faiss_index(chunks):
    """
    Embed each chunk and build a FAISS index.
    """
    print(f"Embedding {len(chunks)} chunks...")
    embeddings = []
    metadata_list = []

    for chunk in tqdm(chunks):
        emb = get_embedding(chunk["text"])
        embeddings.append(emb)
        metadata_list.append(chunk["metadata"])

    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))

    os.makedirs(INDEX_DIR, exist_ok=True)
    faiss.write_index(index, INDEX_PATH)

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata_list, f, indent=2)

    print(f"Stored FAISS index with {len(embeddings)} vectors.")

# Load FAISS index
def load_faiss_index(index_path, metadata_path):
    index = faiss.read_index(index_path)
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata

# Retrieve top-k most relevant chunks
def get_top_k_chunks(question, index, metadata, top_k=5):
    embedding = get_query_embedding(question)
    D, I = index.search(np.array([embedding]).astype("float32"), top_k)
    return [metadata[i] for i in I[0] if i < len(metadata)]

# Embed question using OpenAI
def get_query_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=[text]
    )
    return response.data[0].embedding


if __name__ == "__main__":
    today = datetime.now().strftime("%Y-%m-%d")
    input_dir = os.path.join("outputs", today)

    chunks = load_chunks(input_dir)
    if not chunks:
        print("No chunks found to embed.")
    else:
        build_faiss_index(chunks)
