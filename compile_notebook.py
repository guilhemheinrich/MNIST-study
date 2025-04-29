import os
import sys
import shutil
from pathlib import Path
import jupytext
import nbconvert
import nbformat
import subprocess

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

def convert_to_notebook(input_path: str, output_path: str) -> None:
    """Convert a .py file to a .ipynb notebook.
    
    Args:
        input_path: Path to the input Python file
        output_path: Path to the output notebook file
        
    Note:
        Tags can be added to any cell using the syntax:
        # %% [markdown] tags=["tag1", "tag2"]  # pour les cellules markdown
        # %% tags=["tag1", "tag2"]            # pour les cellules code
    """
    # Read the notebook with jupytext
    nb = jupytext.read(input_path)
    
    # Convert all cells and handle their metadata
    for cell in nb.cells:
        # Ensure metadata exists
        if not hasattr(cell, 'metadata'):
            cell.metadata = {}
            
        # Initialize tags if they don't exist
        if 'tags' not in cell.metadata:
            cell.metadata['tags'] = []
            
        # Handle cell markers and their tags
        source_lines = cell.source.split('\n')
        for i, line in enumerate(source_lines):
            # Check for cell markers with tags
            if line.startswith('# %%'):
                # Extract tags if present
                if 'tags=[' in line:
                    tags_start = line.find('tags=[')
                    tags_end = line.find(']', tags_start)
                    if tags_end != -1:
                        tags_str = line[tags_start+6:tags_end]
                        # Parse the tags (handling both "tag" and tag formats)
                        tags = [t.strip().strip('"\'') for t in tags_str.split(',')]
                        cell.metadata['tags'].extend(tags)
                
        # Handle markdown links
        if cell.cell_type == 'markdown':
            cell.source = cell.source.replace('.py)', '.ipynb)')
    
    # Write the notebook
    with open(output_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

def execute_notebook(notebook_path: str) -> None:
    """Execute a notebook and save the results."""
    try:
        subprocess.run([
            'jupyter', 'nbconvert',
            '--to', 'notebook',
            '--execute',
            '--inplace',
            '--TagRemovePreprocessor.remove_cell_tags', 'remove_cell',
            notebook_path
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing notebook: {e}")
        raise

def convert_notebook_to_html(notebook_path: str, output_path: str) -> None:
    """Convert a .ipynb notebook to an HTML file.
    
    Args:
        notebook_path: Path to the input notebook file
        output_path: Path to the output HTML file
    """
    # Read the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    # Convert all markdown cells to handle links
    for cell in nb.cells:
        if cell.cell_type == 'markdown':
            # Replace .ipynb links with .html
            cell.source = cell.source.replace('.ipynb)', '.html)')
    
    # Create exporter
    exporter = nbconvert.HTMLExporter()
    # Create resources dictionary
    resources = {}
    resources['metadata'] = {}
    
    # Convert notebook
    output, _ = exporter.from_notebook_node(nb, resources=resources)
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(output)

def convert_html_to_pdf(notebook_path: str, output_path: str) -> None:
    """Convert a notebook to PDF using nbconvert and wkhtmltopdf.
    
    Args:
        notebook_path: Path to the input notebook file
        output_path: Path to the output PDF file
    """
    try:
        # Read the notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        
        # Convert all markdown cells to handle links
        for cell in nb.cells:
            if cell.cell_type == 'markdown':
                # Replace .ipynb links with .pdf
                cell.source = cell.source.replace('.ipynb)', '.pdf)')
        
        # Create temporary HTML file
        temp_html = output_path.replace('.pdf', '.html')
        
        # Create exporter
        exporter = nbconvert.HTMLExporter()
        resources = {'metadata': {}}
        
        # Convert notebook to HTML
        output, _ = exporter.from_notebook_node(nb, resources=resources)
        
        # Write HTML to temporary file
        with open(temp_html, 'w', encoding='utf-8') as f:
            f.write(output)
        
        # Convert HTML to PDF using wkhtmltopdf
        subprocess.run(['wkhtmltopdf', temp_html, output_path], check=True)
        
        # Clean up temporary HTML file
        os.remove(temp_html)
        
    except subprocess.CalledProcessError as e:
        print(f"Error converting notebook to PDF: {e}")
    except FileNotFoundError:
        print("wkhtmltopdf not found. Please install it from https://wkhtmltopdf.org/downloads.html")
    except Exception as e:
        print(f"Unexpected error during PDF conversion: {e}")

def build_notebook(input_path: str, build_path: str) -> str:
    """Build a notebook from a Python file.
    
    Args:
        input_path: Path to the input Python file
        build_path: Path to the build directory
        
    Returns:
        Path to the built notebook
    """
    # Create notebook path preserving filename
    notebook_name = Path(input_path).stem + '.ipynb'
    notebook_path = os.path.join(build_path, notebook_name)
    
    # Convert and execute notebook
    convert_to_notebook(input_path, notebook_path)
    execute_notebook(notebook_path)
    
    return notebook_path

def generate_outputs(notebook_path: str, pdf_path: str, public_path: str) -> None:
    """Generate HTML and PDF outputs from a notebook.
    
    Args:
        notebook_path: Path to the notebook
        pdf_path: Path to the PDF output directory
        public_path: Path to the HTML output directory
    """
    # Create output paths
    notebook_name = Path(notebook_path).stem
    html_path = os.path.join(public_path, notebook_name + '.html')
    pdf_output_path = os.path.join(pdf_path, notebook_name + '.pdf')
    
    # Generate outputs
    convert_notebook_to_html(notebook_path, html_path)
    # convert_html_to_pdf(notebook_path, pdf_output_path)

def process_file(input_path: str, base_dir: str) -> None:
    """Process a single .py file."""
    # Create directory structure
    build_path, pdf_path, public_path = ensure_directory_structure(input_path, base_dir)
    
    # Build notebook
    notebook_path = build_notebook(input_path, build_path)
    
    # Generate outputs
    generate_outputs(notebook_path, pdf_path, public_path)

def process_directory(input_dir: str) -> None:
    """Process all .py files in a directory."""
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.py'):
                input_path = os.path.join(root, file)
                process_file(input_path, input_dir)

def main(input_path: str) -> None:
    """Main function to process input path."""
    if os.path.isfile(input_path):
        process_file(input_path, os.path.dirname(input_path))
    elif os.path.isdir(input_path):
        process_directory(input_path)
    else:
        print(f"Error: {input_path} is not a valid file or directory.")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python convert_notebooks.py <input_path>")
        sys.exit(1)

    input_path = sys.argv[1]

    # Create base output directories
    os.makedirs('build', exist_ok=True)
    os.makedirs('pdf', exist_ok=True)
    os.makedirs('public', exist_ok=True)

    main(input_path)
