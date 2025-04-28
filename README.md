# MNIST Study Project

A pedagogical project exploring various machine learning algorithms and methods using the MNIST dataset.

## Project Overview

This project aims to provide a comprehensive study of different machine learning approaches for the MNIST dataset, including:
- Traditional machine learning algorithms
- Deep learning models
- Data preprocessing techniques
- Model evaluation and comparison
- Visualization of results

## Project Structure

```
MNIST-study/
├── data/           # MNIST dataset and other data files
├── src/            # Source code
├── notebooks/      # Jupyter notebooks
├── scripts/        # Utility scripts
├── build/          # Build artifacts
├── pdf/            # Generated PDF files
├── public/         # Public assets
├── README.md       # Project documentation
├── .cursorrules    # Project rules and guidelines
├── .gitignore      # Git ignore rules
├── requirements.txt # Python dependencies
└── compile_notebook.py # Notebook compilation script
```

## Getting Started

### Prerequisites

- Python 3.10 or higher
- pip or conda package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/MNIST-study.git
cd MNIST-study
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Development Guidelines

### Code Style and Formatting
- Follow PEP 8 style guide
- Use black for code formatting
- Use isort for import sorting
- Use flake8 for linting
- Use mypy for type checking
- Maximum line length: 88 characters (black default)

### Python Typing
- Use Python 3.10+ typing syntax:
  - Use `|` for union types instead of `Union`
  - Use `list[Type]` instead of `List[Type]`
  - Use `dict[KeyType, ValueType]` instead of `Dict[KeyType, ValueType]`
  - Use `tuple[Type1, Type2]` instead of `Tuple[Type1, Type2]`
  - Use `type` instead of `Type`
  - Use `| None` instead of `Optional`

### Documentation Standards
- Use Google-style docstrings for all functions, classes, and modules
- Include type hints in docstrings
- Document all public APIs
- Keep docstrings up to date with code changes

### Version Control
- Use semantic commit messages:
  - feat: New feature
  - fix: Bug fix
  - docs: Documentation changes
  - style: Code style changes
  - refactor: Code refactoring
  - test: Adding or modifying tests
  - chore: Maintenance tasks

## Usage

1. Download the MNIST dataset:
```bash
python src/data/download_mnist.py
```

2. Run the main script:
```bash
python src/main.py
```

3. Explore the notebooks:
```bash
jupyter notebook notebooks/
```

## Testing

Run the test suite:
```bash
python -m pytest tests/
```

## Documentation

Documentation is available in the `docs/` directory. It includes:
- API reference
- Tutorials
- Examples
- Architecture overview

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes with semantic commit messages
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- MNIST dataset creators
- Open source machine learning community
