import sys
import os

# Add src/ to sys.path dynamically
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from chunking import chunk_text_blocks

def test_chunking_produces_chunks():
    blocks = [{"text": "This is a sample block of text that is long enough to be chunked."}]
    chunks = chunk_text_blocks(blocks, chunk_size=20, overlap=5)
    assert len(chunks) > 0
    assert "chunk_id" in chunks[0]
    assert "text" in chunks[0]
