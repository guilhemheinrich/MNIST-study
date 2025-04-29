"""Main module for the Jupyter Factory.

This module provides a complete CLI interface for converting Python files to notebooks,
executing them, and converting them to HTML. It can process individual files or entire directories.
"""

import os
import argparse
from pathlib import Path
from .py_to_notebook import convert_to_notebook
from .execute_notebook import execute_notebook
from .notebook_to_html import convert_notebook_to_html

def ensure_directory_structure(input_path: str, base_dir: str) -> tuple[str, str, str]:
    """Create necessary directory structure and return paths.
    
    Args:
        input_path: Path to the input file or directory
        base_dir: Base directory to compute relative paths from
        
    Returns:
        Tuple of (build_path, pdf_path, public_path)
    """
    # Get relative path from base directory
    rel_path = os.path.relpath(input_path, base_dir)
    
    # Create paths preserving directory structure relative to base_dir
    build_path = os.path.join('build', os.path.dirname(rel_path))
    pdf_path = os.path.join('pdf', os.path.dirname(rel_path))
    public_path = os.path.join('public', os.path.dirname(rel_path))

    # Create directories
    os.makedirs(build_path, exist_ok=True)
    os.makedirs(pdf_path, exist_ok=True)
    os.makedirs(public_path, exist_ok=True)

    return build_path, pdf_path, public_path

def process_file(input_path: str, base_dir: str) -> None:
    """Process a single .py file through the complete pipeline.
    
    Args:
        input_path: Path to the input Python file
        base_dir: Base directory for output paths
    """
    # Create directory structure
    build_path, pdf_path, public_path = ensure_directory_structure(input_path, base_dir)
    
    # Create notebook path
    notebook_name = Path(input_path).stem + '.ipynb'
    notebook_path = os.path.join(build_path, notebook_name)
    
    # Convert Python to notebook
    convert_to_notebook(input_path, notebook_path)
    
    # Execute notebook
    execute_notebook(notebook_path)
    
    # Convert to HTML
    html_path = os.path.join(public_path, Path(input_path).stem + '.html')
    convert_notebook_to_html(notebook_path, html_path)
    
    print(f"Successfully processed {input_path}")

def process_directory(input_dir: str) -> None:
    """Process all .py files in a directory.
    
    Args:
        input_dir: Path to the input directory
    """
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.py'):
                input_path = os.path.join(root, file)
                process_file(input_path, input_dir)

def main():
    """CLI entry point for the Jupyter Factory."""
    parser = argparse.ArgumentParser(description='Jupyter Factory - Process Python files to notebooks and HTML')
    parser.add_argument('input', help='Path to input file or directory')
    parser.add_argument('--base-dir', help='Base directory for output paths (default: input directory)')
    
    args = parser.parse_args()
    
    # Set base directory
    base_dir = args.base_dir if args.base_dir else os.path.dirname(args.input)
    
    # Create base output directories
    os.makedirs('build', exist_ok=True)
    os.makedirs('pdf', exist_ok=True)
    os.makedirs('public', exist_ok=True)
    
    if os.path.isfile(args.input):
        process_file(args.input, base_dir)
    elif os.path.isdir(args.input):
        process_directory(args.input)
    else:
        print(f"Error: {args.input} is not a valid file or directory.")

if __name__ == "__main__":
    main() 