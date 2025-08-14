"""
chunking.py

Splits parsed document text into custom chunks for downstream embedding.
"""

import re
from typing import List


def clean_text(text):
    """
    Normalize and clean text (basic whitespace, newlines).
    """
    return re.sub(r'\s+', ' ', text.strip())


def chunk_text_blocks(blocks: List[dict], chunk_size=500, overlap=50):
    """
    Custom chunking logic:
    - Concatenates blocks
    - Splits text into rolling windows of size `chunk_size`
    - Applies overlap between chunks for context preservation

    Parameters:
        blocks (list[dict]): Parsed content with "text"
        chunk_size (int): Max characters per chunk
        overlap (int): Number of overlapping characters between chunks

    Returns:
        list[dict]: Chunked text blocks with IDs and character spans
    """
    full_text = " ".join([clean_text(b.get("text", "")) for b in blocks if b.get("text")])

    chunks = []
    start = 0
    chunk_id = 0

    while start < len(full_text):
        end = min(start + chunk_size, len(full_text))
        chunk = full_text[start:end]

        chunks.append({
            "chunk_id": f"chunk_{chunk_id}",
            "text": chunk,
            "char_start": start,
            "char_end": end
        })

        chunk_id += 1
        start += chunk_size - overlap

    return chunks
