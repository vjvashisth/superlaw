"""
file_utils.py

This module provides utility functions for:
1. Scanning a directory for supported document types.
2. Detecting MIME types based on file extensions.
3. Generating timestamped output directories.

Supported formats: PDF, DOCX, PPTX, TXT, HTML
"""

import os
import mimetypes
from datetime import datetime

# List of supported document extensions
SUPPORTED_EXTENSIONS = ['.pdf', '.docx', '.pptx', '.txt', '.html', '.htm', '.xlsx', '.csv']


def is_supported_file(file_path):
    """
    Check if a file has a supported extension.
    
    Parameters:
        file_path (str): Path to the file
    
    Returns:
        bool: True if the file is supported, False otherwise
    """
    _, ext = os.path.splitext(file_path.lower())  # Split out the extension, lowercase for safety
    return ext in SUPPORTED_EXTENSIONS


def get_supported_files(directory):
    """
    Recursively scan a folder and return all supported files.
    
    Parameters:
        directory (str): Path to the folder to scan
    
    Returns:
        list[str]: List of full paths to supported files
    """
    all_files = []
    
    for root, _, files in os.walk(directory):  # Recursively walk through directory
        for file in files:
            file_path = os.path.join(root, file)
            if is_supported_file(file_path):
                all_files.append(file_path)
    
    return all_files


def detect_mime_type(file_path):
    """
    Guess the MIME type of a file using the mimetypes module.
    
    Parameters:
        file_path (str): Path to the file
    
    Returns:
        str: Detected MIME type, or a fallback value
    """
    mime_type, _ = mimetypes.guess_type(file_path)
    
    # Fallback if MIME type is not detected
    return mime_type or "application/octet-stream"


def get_timestamped_output_dir(base_path="outputs"):
    """
    Create and return an output directory with today's date as the subfolder.
    
    Parameters:
        base_path (str): Base path where the dated folder should be created
    
    Returns:
        str: Full path to the timestamped output directory
    """
    today = datetime.now().strftime("%Y-%m-%d")
    output_dir = os.path.join(base_path, today)
    
    os.makedirs(output_dir, exist_ok=True)  # Avoids crash if folder already exists
    
    return output_dir
