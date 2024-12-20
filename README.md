# DNA Preprocessing Framework

Live project. Trust nothing. 
## Overview

Master thesis project aiming support the classification of fungal DNA sequences, particularly ITS barcodes, using advanced tokenization and augmentation strategies informed by state-of-the-art research.

## Features (Goals as of now)

- **Customizable Preprocessing Pipelines**: Supports dynamic loading of strategies for augmentation, tokenization, padding, and truncation.
- **Flexible Tokenization**: Includes k-mer tokenization and plans to explore alternatives such as BPE etc.
- **Extensive Logging**: Detailed logging for debugging and tracking processing pipelines.
- **Multilevel Taxonomic Classification**: Designed to integrate with BERT-style architectures for hierarchical taxonomic classification.

## Directory Structure

```
.
├── README.md                # Project documentation
├── errors.py                # Custom error definitions
├── factory.py               # Factory methods for creating vocabularies and preprocessors
├── logging_utils.py         # Logging utilities
├── main.py                  # Entry point for the application
├── preprocessing/           # Preprocessing strategies and utilities
│   ├── augmentation.py      # Sequence augmentation strategies
│   ├── padding.py           # Padding strategies
│   ├── preprocessor.py      # Preprocessor implementation
│   ├── tokenization.py      # Tokenization strategies
│   ├── truncation.py        # Truncation strategies
├── tests/                   # Unit tests for preprocessing components
├── logs/                    # Log files
├── requirements.txt         # Python dependencies
├── vocab.py                 # Vocabulary management
```

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-repo/dna-preprocessing-framework.git
   cd dna-preprocessing-framework
   ```

2. Set up a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Configure preprocessing settings: Modify the configuration dictionary in `main.py` to specify strategies for augmentation, tokenization, padding, and truncation.

2. Run the application:

   ```bash
   python main.py
   ```

3. View logs: Logs are saved in the `logs/` directory, with separate files for system and training logs.

## Configuration

The framework is driven by a configuration dictionary, as shown below:

```python
config = {
    "model": {"name": "example_model"},
    "augmentation": {"strategy": "base", "alphabet": ["A", "C", "G", "T"], "modification_probability": 0.5},
    "tokenization": {"strategy": "kmer", "k": 3},
    "padding": {"strategy": "random", "optimal_length": 8},
    "truncation": {"strategy": "slidingwindow", "optimal_length": 8}
}
```

This configuration is passed to the factory methods for creating the vocabulary and preprocessing pipeline.

## Extensibility
    TBC
- **Add New Strategies**:
    TBC

- **Custom Logging Levels**: Add or modify logging levels in `logging_utils.py` as needed.
    TBC
## Error Handling

The project includes custom exceptions for error handling, such as:

- `PreprocessingError`: Base class for preprocessing-related errors.
- `AugmentationError`, `TokenizationError`, `PaddingError`, `TruncationError`: Specific errors for preprocessing stages.
- `ConstructionError`: Errors related to vocabulary and preprocessor construction.

## Tests

Unit tests are located in the `tests/` directory. Run tests using:

```bash
pytest tests/
```

## Contributors

Filip Berntsson, Marcus Axelssion


## Acknowledgments

This project builds on concepts introduced by Rochas Cayulas (2016) and Mock et al. (2020). It is part of an ongoing master thesis project "Improved classification of DNA barcodes using transformers," a supervised by Erik Kristiansson.

