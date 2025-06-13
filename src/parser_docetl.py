"""
This file was originally intended to generate and run DocETL pipelines for document parsing.
However, due to DocETL's limitations with raw PDFs and unsupported operations,
the main pipeline has pivoted to using Unstructured.io for document parsing.

This file can be used as a reference for:
- Dynamically generating DocETL-compatible YAML configs
- Running DocETL CLI via subprocess
- Structuring pipeline steps, operations, and datasets for structured input (e.g., CSV, JSON)

Status: Deprecated in favor of Unstructured-based pipeline (see parser_unstructured.py)
"""

"""
parser_docetl.py

Runs DocETL using a dynamic DSL pipeline YAML per document,
utilizing 'scan' and 'add_uuid' operations.
"""

import os
import json
import yaml
import subprocess
import tempfile
from datetime import datetime


def parse_with_docetl(file_path):
    """
    Runs DocETL CLI on a single document using a generated pipeline config.

    Parameters:
        file_path (str): Path to the input document

    Returns:
        list[dict]: List of parsed content blocks, or empty list on failure
    """
    today = datetime.now().strftime("%Y-%m-%d")
    output_dir = os.path.join("outputs", today)
    os.makedirs(output_dir, exist_ok=True)

    file_stem = os.path.splitext(os.path.basename(file_path))[0]
    output_path = os.path.join(output_dir, f"{file_stem}.json")
    intermediate_dir = os.path.join(output_dir, "intermediate")

    # Build the pipeline config
    pipeline_config = {
        "datasets": {
            "input_file": {
                "type": "document",
                "path": file_path
            }
        },
        "operations": [
            {
                "name": "scan",
                "type": "scan",
                "dataset_name": "input_file"
            },
            {
                "name": "add_uuid",
                "type": "add_uuid"
            }
        ],
        "pipeline": {
            "steps": [
                {
                    "name": "scan_and_uuid",
                    "input": "input_file",
                    "operations": ["scan", "add_uuid"]
                }
            ],
            "output": {
                "type": "file",
                "path": output_path,
                "intermediate_dir": intermediate_dir
            }
        }
    }

    # Write YAML config to a temporary file
    with tempfile.NamedTemporaryFile(mode='w+', suffix=".yaml", delete=False) as temp_yaml:
        yaml.dump(pipeline_config, temp_yaml)
        temp_yaml_path = temp_yaml.name

    # Run the CLI
    try:
        subprocess.run(
            ["python", "-m", "docetl.cli", "run", temp_yaml_path],
            capture_output=True,
            text=True,
            check=True
        )

        if os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            return [
                {
                    "id": idx,
                    "type": chunk.get("type", "text"),
                    "text": chunk.get("text", ""),
                    "metadata": chunk.get("metadata", {})
                }
                for idx, chunk in enumerate(data.get("chunks", []))
            ]

        else:
            print(f"DocETL ran but output not found for: {file_path}")

    except subprocess.CalledProcessError as e:
        print(f"DocETL failed for {file_path}:\n{e.stderr.strip()}")

    return []
