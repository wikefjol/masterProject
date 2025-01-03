import os
import json
import itertools

# Define reduced parameters for testing
augmentation_strategies = ["base", "random"]
truncation_strategies = ["front", "end"]
padding_strategies = ["front", "end"]
k_values = [3, 5]
optimal_length = 8

# Default values for general, pretraining, and finetuning configs
fasta_file_path = "data/raw/raw.fasta"
prepared_data_dir = "data/prepared"
pretraining_dataset = "data/pretraining_data.csv"
finetuning_dataset = "data/finetuning_data.csv"
test_size = 0.2
random_seed = 42

# Output directory for scenarios
output_dir = "scenarios"
os.makedirs(output_dir, exist_ok=True)

# General configuration template
general_config = {
    "tokenization_strategy": "kmer",
    "log_dir": "runs/logs",
    "system_log_level": 20,
    "training_log_level": 20
}

# Pretraining configuration template
base_pretraining_config = {
    "enabled": True,
    "fasta_file": fasta_file_path,
    "prepared_data_dir": prepared_data_dir,
    "dataset": pretraining_dataset,
    "force_reprocess": False,
    "test_size": test_size,
    "random_seed": random_seed,
    "vocab_options": {
        "min_frequency": 2,
        "max_tokens": 5000
    },
    "preprocessor_options": {
        "augmentation_strategy": {},
        "tokenization_strategy": {
            "strategy": "kmer",
            "k": None
        },
        "padding_strategy": {},
        "truncation_strategy": {}
    }
}

# Finetuning configuration template
base_finetuning_config = {
    "enabled": True,
    "fasta_file": fasta_file_path,
    "prepared_data_dir": prepared_data_dir,
    "dataset": finetuning_dataset,
    "force_reprocess": False,
    "test_size": test_size,
    "random_seed": random_seed,
    "preprocessor_options": {
        "augmentation_strategy": {},
        "tokenization_strategy": {
            "strategy": "kmer",
            "k": None
        },
        "padding_strategy": {},
        "truncation_strategy": {}
    }
}

# Generate 5 combinations of pretraining and finetuning configurations
combinations = list(itertools.product(
    k_values,
    augmentation_strategies,
    truncation_strategies,
    padding_strategies
))[:5]  # Limit to 5 combinations

for i, (k, augmentation, truncation, padding) in enumerate(combinations):
    # Create a folder for each full scenario
    scenario_dir = os.path.join(output_dir, f"scenario_{i+1}")
    os.makedirs(scenario_dir, exist_ok=True)

    # Create copies of the base configurations
    pretraining_config = base_pretraining_config.copy()
    finetuning_config = base_finetuning_config.copy()

    # Update tokenization strategy for both pretraining and finetuning
    pretraining_config["preprocessor_options"]["tokenization_strategy"]["k"] = k
    finetuning_config["preprocessor_options"]["tokenization_strategy"]["k"] = k

    # Update truncation strategy
    pretraining_config["preprocessor_options"]["truncation_strategy"] = {
        "strategy": truncation,
        "optimal_length": optimal_length
    }
    finetuning_config["preprocessor_options"]["truncation_strategy"] = {
        "strategy": truncation,
        "optimal_length": optimal_length
    }

    # Update padding strategy
    pretraining_config["preprocessor_options"]["padding_strategy"] = {
        "strategy": padding,
        "optimal_length": optimal_length
    }
    finetuning_config["preprocessor_options"]["padding_strategy"] = {
        "strategy": padding,
        "optimal_length": optimal_length
    }

    # Update augmentation strategy
    pretraining_config["preprocessor_options"]["augmentation_strategy"] = {
        "strategy": augmentation,
        "alphabet": ["A", "C", "G", "T"],
        "modification_probability": 0.5
    }
    finetuning_config["preprocessor_options"]["augmentation_strategy"] = {
        "strategy": augmentation,
        "alphabet": ["A", "C", "G", "T"],
        "modification_probability": 0.5
    }

    # Save the configurations in the scenario folder
    general_config_path = os.path.join(scenario_dir, "general_config.json")
    pretraining_config_path = os.path.join(scenario_dir, "pretraining_config.json")
    finetuning_config_path = os.path.join(scenario_dir, "finetuning_config.json")

    with open(general_config_path, "w") as f:
        json.dump(general_config, f, indent=4)

    with open(pretraining_config_path, "w") as f:
        json.dump(pretraining_config, f, indent=4)

    with open(finetuning_config_path, "w") as f:
        json.dump(finetuning_config, f, indent=4)

print(f"Generated {len(combinations)} full scenarios in '{output_dir}'")
