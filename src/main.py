"""
main.py

End-to-end document parsing pipeline:
1. Scans the samples/ folder
2. Parses using Unstructured.io
3. Extracts metadata
4. Chunks the text
5. Saves output as JSON in outputs/YYYY-MM-DD/

Run: python src/main.py
"""

import os
import sys
import yaml

# Allow relative imports when running from root
sys.path.append(os.path.dirname(__file__))

from file_utils import get_supported_files
from parser_unstructured import parse_with_unstructured
from metadata_extractor import extract_metadata
from chunking import chunk_text_blocks
from schema import save_output


def load_config(path="config/config.yaml"):
    """
    Load configuration from YAML file.
    """
    if not os.path.exists(path):
        print(f"Config file not found: {path}")
        return {}

    try:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(f"Error parsing config: {e}")
        return {}


if __name__ == "__main__":
    print("Loading config...")
    config = load_config()
    chunk_size = config.get("chunk_size", 500)
    overlap = config.get("overlap", 50)

    parser_choice = config.get("parser", "unstructured")
    if parser_choice != "unstructured":
        print("Warning: Only 'unstructured' parser is supported in current version.")

    print("Scanning for documents in samples/...")
    files = get_supported_files("samples")

    if not files:
        print("No supported documents found.")
        sys.exit(0)

    for file_path in files:
        print(f"\nParsing: {file_path}")
        parsed = parse_with_unstructured(file_path)

        if not parsed:
            print("No content extracted.")
            continue

        metadata = extract_metadata(file_path, parsed)
        chunks = chunk_text_blocks(parsed, chunk_size=chunk_size, overlap=overlap)
        output_path = save_output(file_path, metadata, chunks)

        print(f"Extracted {len(chunks)} chunks")
        print(f"Output saved to: {output_path}")
