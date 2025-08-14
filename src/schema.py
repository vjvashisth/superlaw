"""
schema.py

Defines output schema and saves structured output to JSON.
"""

import os
import json
from datetime import datetime


def save_output(file_path, metadata, chunks, output_base="outputs"):
    """
    Saves combined metadata and chunks into a JSON file.

    Parameters:
        file_path (str): Original file (used to derive name)
        metadata (dict): Extracted document-level metadata
        chunks (list): Chunked content sections
        output_base (str): Base output folder (default: outputs/)

    Returns:
        str: Path to saved JSON
    """
    today = datetime.now().strftime("%Y-%m-%d")
    out_dir = os.path.join(output_base, today)
    os.makedirs(out_dir, exist_ok=True)

    file_name = os.path.splitext(os.path.basename(file_path))[0] + ".json"
    output_path = os.path.join(out_dir, file_name)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "document_metadata": metadata,
            "chunks": chunks
        }, f, indent=2, ensure_ascii=False)

    return output_path
