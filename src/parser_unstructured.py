"""
parser_unstructured.py

Parses documents using Unstructured.io and normalizes the result into a
list of chunks with metadata, ready for chunking and embedding.
"""

from unstructured.partition.auto import partition
from pathlib import Path


def parse_with_unstructured(file_path):
    """
    Parses a document using Unstructured.io's `partition()` function.
    Automatically selects strategy for tabular formats (CSV, XLSX).

    Parameters:
        file_path (str): Path to the input document

    Returns:
        list[dict]: List of parsed content blocks (text + metadata)
    """
    try:
        ext = Path(file_path).suffix.lower()
        strategy = "hi_res" if ext in [".csv", ".xlsx"] else "auto"

        elements = partition(filename=file_path, strategy=strategy)

        parsed_output = []
        for idx, element in enumerate(elements):
            parsed_output.append({
                "id": idx,
                "type": element.category,  # e.g., Title, NarrativeText, ListItem, etc.
                "text": str(element),
                "metadata": element.metadata.to_dict() if element.metadata else {}
            })

        return parsed_output

    except Exception as e:
        print(f"Failed to parse {file_path} with Unstructured: {e}")
        return []
