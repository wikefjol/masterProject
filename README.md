# DNA Preprocessing Framework

Live project. Trust nothing.

## Overview

Master thesis project aimed at supporting the classification of fungal DNA sequences, particularly ITS barcodes, using advanced tokenization and augmentation strategies informed by state-of-the-art research. This framework facilitates pretraining and finetuning workflows for BERT-style models, enabling hierarchical taxonomic classification.

## Features

- **Two-Stage Training**:
  - **Pretraining**: Masked language modeling (MLM) without next sentence prediction (NSP).
  - **Finetuning**: Supervised classification at various taxonomic levels.
- **Scenario-Based Configurations**:
  - Modular configuration setup for general, pretraining, and finetuning scenarios.
  - Supports pairing multiple pretraining scenarios with different finetuning scenarios for comparative studies.
- **Customizable Preprocessing Pipelines**:
  - Dynamic strategies for augmentation, tokenization, padding, and truncation.
  - Built-in support for k-mer tokenization with plans to explore other strategies like BPE.
- **Efficient Logging**:
  - Configurable logging levels to balance debugging detail and performance.
- **Hierarchical Classification**:
  - Designed to integrate with BERT-style architectures for multilevel taxonomic classification.
- **Performance Optimized**:
  - Tuning preprocessing to achieve speeds of 2000+ sequences/second for large datasets.

## Directory Structure

```
.
├── README.md                # Project documentation
├── data/                    # Raw, prepared, and processed data files
│   ├── raw/                 # Raw FASTA files
│   ├── prepared/            # Prepared training/testing data
│   ├── processed/           # Processed pretraining/finetuning data
├── scenarios/               # Configuration files for different scenarios
│   ├── scenario_1/          # Folder for a specific scenario
│   │   ├── general_config.json
│   │   ├── pretraining_config.json
│   │   └── finetuning_config.json
├── runs/                    # Output of runs
│   ├── scenario_1/          # Logs and outputs of a specific scenario run
│   │   ├── pretraining_data.csv
│   │   ├── pretraining_vocab.json
│   │   ├── system_*.log
│   │   └── training_*.log
├── src/                     # Source code
│   ├── errors.py            # Custom error definitions
│   ├── factory.py           # Factory methods for vocabularies and preprocessors
│   ├── main.py              # Entry point for the application
│   ├── preprocessing/       # Preprocessing strategies
│   │   ├── augmentation.py  # Sequence augmentation strategies
│   │   ├── padding.py       # Padding strategies
│   │   ├── preprocessor.py  # Preprocessor implementation
│   │   ├── tokenization.py  # Tokenization strategies
│   │   └── truncation.py    # Truncation strategies
│   ├── run_scenario.py      # Core scenario runner
│   ├── utils/               # Utility scripts
│   │   ├── generate_configs.py  # Script to generate configuration files
│   │   └── logging_utils.py     # Logging utilities
├── tests/                   # Unit tests for preprocessing components
├── requirements.txt         # Python dependencies
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/<repo-url>/barcode-classifier.git
   cd barcode-classifier
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

### Step 1: Generate Configurations
Use the `generate_configs.py` script to generate scenarios. Each scenario consists of:
- A general configuration.
- A pretraining configuration.
- A finetuning configuration.

```bash
python src/utils/generate_configs.py
```

### Step 2: Run a Scenario
To run a specific scenario:

```bash
python src/run_scenario.py scenarios/scenario_1
```

### Step 3: Pretraining and Finetuning
- Pretraining prepares data using the `SequenceDataPreparer`, creates a vocabulary, and preprocesses sequences.
- Finetuning uses the pretraining vocabulary and prepares data for classification.

## Logging
Logs are saved in the `runs/<scenario>` directory, with system and training logs separated for better organization. Logging levels can be adjusted in the general configuration file.

## Tests

Run unit tests with:
```bash
pytest tests/
```

## Configuration

### General Configuration
Shared settings for pretraining and finetuning, including:
- Logging levels.
- Output directories.

### Pretraining Configuration
Specifies parameters for masked language modeling, including:
- Tokenization (e.g., k-mer size).
- Data augmentation strategies.
- Padding and truncation strategies.

### Finetuning Configuration
Specifies settings for supervised classification, sharing tokenization settings with pretraining.

## Performance Notes

- Without low-level logging wrappers, preprocessing achieves speeds of **2000+ sequences/second**.
- Extensive logging is available for debugging but introduces significant overhead.

## Contributors

- Filip Berntsson
- Marcus Axelsson

## Acknowledgments

This project builds on concepts introduced by Rochas Cayulas (2016) and Mock et al. (2020). It is part of an ongoing master thesis project: *Improved classification of DNA barcodes using transformers*, supervised by Erik Kristiansson.