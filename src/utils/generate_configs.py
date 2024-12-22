import os
import json

# Define parameters
augmentation_strategies = ["base", "random", "identity"]  # Add your strategies here
truncation_strategies = ["front", "end", "slidingwindow"]
padding_strategies = ["front", "end", "random"]
k_values = [3, 5]
optimal_length = 8

# Output directory for scenarios
output_dir = "scenarios"
os.makedirs(output_dir, exist_ok=True)

# Template for configuration
base_config = {
    "model": {
        "name": "test_model"
    },
    "augmentation": {},
    "tokenization": {
        "strategy": "kmer",
    },
    "padding": {},
    "truncation": {}
}

# Generate all combinations
for k in k_values:
    for truncation in truncation_strategies:
        for padding in padding_strategies:
            for augmentation in augmentation_strategies:
                # Create a copy of the base configuration
                config = base_config.copy()

                # Set k-mer size
                config["tokenization"]["k"] = k

                # Set truncation strategy
                config["truncation"] = {
                    "strategy": truncation,
                    "optimal_length": optimal_length
                }

                # Set padding strategy
                config["padding"] = {
                    "strategy": padding,
                    "optimal_length": optimal_length
                }

                # Set augmentation strategy
                config["augmentation"] = {
                    "strategy": augmentation,
                    "alphabet": ["A", "C", "G", "T"],
                    "modification_probability": 0.5
                }

                # Generate filename
                filename = f"config_k{k}_{augmentation}_{truncation}_{padding}.json"

                # Save configuration to file
                with open(os.path.join(output_dir, filename), "w") as f:
                    json.dump(config, f, indent=4)

print(f"Generated {len(augmentation_strategies) * len(truncation_strategies) * len(padding_strategies) * len(k_values)} configuration files in '{output_dir}'")
