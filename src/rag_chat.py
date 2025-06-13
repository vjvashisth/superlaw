# src/rag_chat.py

from openai import OpenAI
from src.embed_and_store import load_faiss_index, get_top_k_chunks

client = OpenAI()

# Load your FAISS index and metadata
index, metadata = load_faiss_index("vector_index/faiss.index", "vector_index/index_metadata.pkl")

def build_prompt(question: str, context_chunks: list) -> str:
    context = "\n\n".join(context_chunks)
    return f"""You are a helpful assistant. Use the following context to answer the question.

Context:
{context}

Question: {question}
Answer:"""

def generate_rag_response(question: str, top_k: int = 5) -> tuple[str, list[str], list[str]]:
    top_chunks = get_top_k_chunks(question, index, metadata, top_k)
    prompt = build_prompt(question, [chunk['text'] for chunk in top_chunks])

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=500
    )

    # answer = response.choices[0].message.content
    # sources = list({chunk.get("source", "unknown") for chunk in top_chunks})
    # context = [chunk["text"] for chunk in top_chunks]

    # return answer, sources, context
    return response.choices[0].message.content

