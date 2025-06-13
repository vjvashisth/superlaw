"""
metadata_extractor.py

Extracts document-level metadata from file path and parser output.
"""

import os
from datetime import datetime


def extract_metadata(file_path, parser_output):
    """
    Derives metadata from file path and parser-generated metadata fields.

    Parameters:
        file_path (str): Full path to original file
        parser_output (list[dict]): List of parsed content blocks

    Returns:
        dict: Normalized metadata dictionary
    """
    file_name = os.path.basename(file_path)
    name, ext = os.path.splitext(file_name)

    # Try to extract first non-empty metadata field from parsed chunks
    sample_meta = next((chunk.get("metadata", {}) for chunk in parser_output if chunk.get("metadata")), {})

    return {
        "title": sample_meta.get("title", name),
        "author": sample_meta.get("author", "Unknown"),
        "created_at": sample_meta.get("created_at", datetime.now().strftime("%Y-%m-%d")),
        "source_type": ext.lstrip('.').lower(),
        "source_path": file_path
    }
