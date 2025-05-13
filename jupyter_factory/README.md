# Jupyter nbconvert Documentation

This directory contains utilities for converting Jupyter notebooks to various formats using `jupyter nbconvert`.

## Overview

`jupyter nbconvert` is a powerful tool that allows you to convert Jupyter notebooks to various formats including:
- HTML
- PDF
- Python scripts
- Markdown
- LaTeX
- ReStructuredText
- Executable scripts

## Command Line Interface (CLI)

### Basic Usage

```bash
jupyter nbconvert --to <format> notebook.ipynb
```

### Common Formats and Requirements

#### HTML
- No additional requirements
- Basic format for web viewing
- Supports embedded images and interactive widgets

#### PDF
- Requires LaTeX installation
- On Windows: Install MiKTeX
- On Linux: Install TeX Live
- On macOS: Install MacTeX

#### Python Script
- No additional requirements
- Extracts code cells into a .py file
- Preserves markdown as comments

#### Markdown
- No additional requirements
- Converts to .md files
- Preserves code blocks and markdown formatting

#### LaTeX
- Requires LaTeX installation
- Generates .tex files
- Useful for academic papers

## Python Library Usage

For more advanced usage, you can use the nbconvert Python API:

```python
import nbformat
from nbconvert import HTMLExporter

# Load notebook
with open('notebook.ipynb') as f:
    nb = nbformat.read(f, as_version=4)

# Convert to HTML
html_exporter = HTMLExporter()
html_data, resources = html_exporter.from_notebook_node(nb)

# Save to file
with open('output.html', 'w') as f:
    f.write(html_data)
```

### Advanced Features
- Custom templates
- Preprocessors
- Post-processors
- Custom exporters
- Configuration management

## Limitations

### CLI Limitations
- Limited customization options
- No programmatic control over conversion process
- Basic error handling

### Library Limitations
- Memory usage with large notebooks
- Complex setup for custom exporters
- Some formats require external dependencies

## Official Documentation

For more detailed information, please refer to the official documentation:
- [Jupyter nbconvert Documentation](https://nbconvert.readthedocs.io/)
- [GitHub Repository](https://github.com/jupyter/nbconvert)
- [Config Options](https://nbconvert.readthedocs.io/en/latest/config_options.html)
## Best Practices

1. Always test conversions with a small notebook first
2. Keep dependencies up to date
3. Use version control for custom templates
4. Document any custom configurations
5. Consider using preprocessors for cleaning notebooks before conversion 