"""
Generates OpenAI embeddings for chunked documents and stores them in a FAISS index.
Requires: OPENAI_API_KEY in env, openai>=1.0.0, faiss-cpu, numpy
"""

import os
import json
import pickle
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional

import faiss
import numpy as np
from tqdm import tqdm
from openai import OpenAI

# -----------------------------
# Config
# -----------------------------
INDEX_DIR = "vector_index"
INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")
# By default we will write JSON; we can also read legacy pickle if present.
METADATA_JSON_PATH = os.path.join(INDEX_DIR, "metadata.json")
METADATA_PKL_PATH = os.path.join(INDEX_DIR, "index_metadata.pkl")

# For OpenAI: use env override if present (also works for Azure OpenAI if you
# set OPENAI_EMBEDDING_MODEL to your deployment name).
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# Initialize client (relies on OPENAI_API_KEY in env or Streamlit secrets)
client = OpenAI()


# -----------------------------
# Embedding helpers
# -----------------------------
def get_embedding(text: str) -> List[float]:
    """
    Generate a single embedding using OpenAI's v1.x client.
    Returns a Python list of floats (so it can be JSON-serialized if needed).
    """
    if not isinstance(text, str):
        text = str(text)

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[text]
    )
    return response.data[0].embedding


def get_query_embedding(question: str) -> np.ndarray:
    """
    Returns a float32 numpy vector for the query text.
    """
    emb = get_embedding(question)
    return np.array(emb, dtype="float32")


# -----------------------------
# I/O for chunks
# -----------------------------
def load_chunks(input_dir: str) -> List[Dict[str, Any]]:
    """
    Load all chunks from JSON files in a given folder.
    Expects each file to have a structure like:
      {
        "document_metadata": {...},
        "chunks": [{"chunk_id": "...", "text": "..."}, ...]
      }
    Returns a list of dicts with text + flattened metadata.
    """
    chunks: List[Dict[str, Any]] = []
    if not os.path.isdir(input_dir):
        return chunks

    for fname in os.listdir(input_dir):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(input_dir, fname), "r", encoding="utf-8") as f:
            data = json.load(f)

        doc_meta = data.get("document_metadata", {}) or {}
        for ch in data.get("chunks", []):
            # Normalize a single chunk entry
            text = ch.get("text", "")
            chunk_id = ch.get("chunk_id")
            meta = {
                **doc_meta,
                "chunk_id": chunk_id,
            }
            chunks.append({
                "text": text,
                "metadata": meta,
            })
    return chunks


# -----------------------------
# Index builder
# -----------------------------
def build_faiss_index(chunks: List[Dict[str, Any]]) -> None:
    """
    Embed each chunk and build a FAISS index.
    Also writes aligned metadata list that includes the chunk text so retrieval
    can display context without reloading source files.
    """
    if not chunks:
        print("No chunks to index.")
        return

    print(f"Embedding {len(chunks)} chunks with model '{EMBEDDING_MODEL}'...")
    embeddings = []
    metadata_list = []  # aligned with embeddings

    for ch in tqdm(chunks):
        text = ch.get("text", "")
        meta = ch.get("metadata", {}) or {}
        emb = get_embedding(text)
        embeddings.append(emb)
        # Store text alongside metadata to simplify retrieval
        metadata_list.append({
            "text": text,
            **meta,
        })

    # Build FAISS
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings, dtype="float32"))

    # Write index + metadata
    os.makedirs(INDEX_DIR, exist_ok=True)
    faiss.write_index(index, INDEX_PATH)

    # Prefer JSON for portability
    with open(METADATA_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata_list, f, ensure_ascii=False, indent=2)

    # (Optional) also write a pickle for backward-compatibility with older code
    with open(METADATA_PKL_PATH, "wb") as f:
        pickle.dump(metadata_list, f)

    print(f"Stored FAISS index with {len(embeddings)} vectors at {INDEX_PATH}")
    print(f"Stored metadata at {METADATA_JSON_PATH} (and {METADATA_PKL_PATH})")


# -----------------------------
# Index loader
# -----------------------------
def load_faiss_index(index_path: str, metadata_path: Optional[str] = None) -> Tuple[faiss.Index, List[Dict[str, Any]]]:
    """
    Load FAISS index and aligned metadata list.
    Supports both JSON (preferred) and legacy pickle metadata.
    If metadata_path is provided, attempts to infer format from extension; otherwise,
    tries JSON first then pickle in default locations.
    """
    index = faiss.read_index(index_path)

    # Try an explicit metadata_path first
    if metadata_path:
        if metadata_path.endswith(".json"):
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        else:
            with open(metadata_path, "rb") as f:
                metadata = pickle.load(f)
        return index, metadata

    # Fall back to defaults
    if os.path.exists(METADATA_JSON_PATH):
        with open(METADATA_JSON_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return index, metadata

    if os.path.exists(METADATA_PKL_PATH):
        with open(METADATA_PKL_PATH, "rb") as f:
            metadata = pickle.load(f)
        return index, metadata

    raise FileNotFoundError(
        f"Could not find metadata next to index. Looked for:\n"
        f"  {METADATA_JSON_PATH}\n  {METADATA_PKL_PATH}\n"
        f"Or provide an explicit metadata_path."
    )


# -----------------------------
# Retrieval
# -----------------------------
def get_top_k_chunks(
    index: faiss.Index,
    metadata: List[Dict[str, Any]],
    question: str,
    top_k: int = 5,
    k: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Search the FAISS index and return top_k metadata entries.

    Accepts both 'top_k' and legacy 'k' names. Returns a list of dicts like:
      [{"text": "...", "source": "...", ...}, ...]
    If your metadata doesn't have a 'source', we fall back to 'path' or 'unknown'.
    """
    if k is not None:  # compatibility
        top_k = k

    qvec = get_query_embedding(question)        # (dim,)
    qvec = np.expand_dims(qvec, axis=0)         # (1, dim)

    distances, indices = index.search(qvec, top_k)
    results: List[Dict[str, Any]] = []

    for idx in indices[0]:
        if idx < 0 or idx >= len(metadata):
            continue
        m = metadata[idx] or {}
        if isinstance(m, dict):
            # Ensure consistent keys
            text = m.get("text", "")
            source = m.get("source", m.get("path", "unknown"))
            out = dict(m)
            if "text" not in out:
                out["text"] = text
            if "source" not in out:
                out["source"] = source
            results.append(out)
        else:
            # metadata stored as plain text
            results.append({"text": str(m), "source": "unknown"})

    return results


# -----------------------------
# CLI entry (optional)
# -----------------------------
if __name__ == "__main__":
    today = datetime.now().strftime("%Y-%m-%d")
    input_dir = os.path.join("outputs", today)

    chunks = load_chunks(input_dir)
    if not chunks:
        print(f"No chunks found to embed in: {input_dir}")
    else:
        build_faiss_index(chunks)
