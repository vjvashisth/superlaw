# src/rag_chat.py
from typing import List, Optional
from openai import OpenAI
from src.embed_and_store import load_faiss_index, get_top_k_chunks

_client_cache = {}

def _client(api_key: Optional[str] = None) -> OpenAI:
    if api_key not in _client_cache:
        _client_cache[api_key] = OpenAI(api_key=api_key) if api_key else OpenAI()
    return _client_cache[api_key]

def build_prompt(question: str, context_chunks: List[str]) -> str:
    context = "\n\n".join(context_chunks)
    return f"""You are a helpful assistant. Use the following context to answer the question.

Context:
{context}

Question: {question}

Answer concisely. If the answer isn't in the context, say you don't know.
"""

def _get_chunks_any_signature(index, metadata, question: str, k: int):
    """
    Call get_top_k_chunks regardless of whether it expects k=, top_k=, or positional.
    """
    try:
        # many repos use k=
        return get_top_k_chunks(index, metadata, question, k=k)
    except TypeError:
        try:
            # some use top_k=
            return get_top_k_chunks(index, metadata, question, top_k=k)  # type: ignore
        except TypeError:
            # positional only
            return get_top_k_chunks(index, metadata, question, k)  # type: ignore

def generate_rag_response(
    question: str,
    k: int = 5,
    index=None,
    metadata=None,
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
    max_tokens: int = 600,
    temperature: float = 0.2,
) -> str:
    """
    Returns a plain-text answer from RAG.
    If index/metadata are not provided, tries to load from ./vector_index.
    """
    if index is None or metadata is None:
        index, metadata = load_faiss_index("vector_index/faiss.index", "vector_index/index_metadata.pkl")

    top_chunks = _get_chunks_any_signature(index, metadata, question, k=k)

    # tolerate either list[dict] or list[str]
    def _as_text(c):
        if isinstance(c, str):
            return c
        if isinstance(c, dict):
            return c.get("text", "")
        return str(c)

    chunk_texts = [_as_text(c) for c in top_chunks]
    prompt = build_prompt(question, chunk_texts)

    client = _client(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content
