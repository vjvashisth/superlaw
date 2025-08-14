import os
import json
import pickle
import faiss
import numpy as np
from openai import OpenAI
from glob import glob
from tqdm import tqdm

OpenAI.api_key = os.getenv("OPENAI_API_KEY")
# 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
client = OpenAI()

def load_chunks_from_outputs(folder):
    all_chunks = []
    for filepath in glob(os.path.join(folder, "*.json")):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            for chunk in data.get("chunks", []):
                if "text" in chunk:
                    chunk["source"] = os.path.basename(filepath)
                    all_chunks.append(chunk)
    return all_chunks

def embed_texts(texts):
    embeddings = []
    for i in range(0, len(texts), 100):  # batch size 100
        batch = texts[i:i+100]
        response = client.embeddings.create(model="text-embedding-3-small", input=batch)
        batch_embeddings = [e.embedding for e in response.data]
        embeddings.extend(batch_embeddings)
    return np.array(embeddings).astype("float32")


def build_and_save_index(chunks, index_dir="vector_index"):
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embed_texts(texts)

    # Build FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    os.makedirs(index_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(index_dir, "faiss.index"))

    # Save metadata
    with open(os.path.join(index_dir, "index_metadata.pkl"), "wb") as f:
        pickle.dump(chunks, f)

    print(f"✅ Built FAISS index with {len(chunks)} chunks.")

if __name__ == "__main__":
    latest_output_dir = sorted(os.listdir("outputs"))[-1]
    folder = os.path.join("outputs", latest_output_dir)
    chunks = load_chunks_from_outputs(folder)
    build_and_save_index(chunks)
