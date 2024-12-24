from run_scenario import run_scenario
import os
import json

def main():
    """
    Entry point for running a specific scenario with hierarchical configuration.

    Workflow:
    - This script expects the path to a scenario folder containing:
        - `general_config.json`: Defines shared settings across scenarios.
        - `pretraining_config.json`: Defines settings specific to the pretraining phase.
        - `finetuning_config.json`: Defines settings specific to the finetuning phase.

    Steps:
    1. Generate scenario folders and configurations using `generate_configs.py` if none exist.
    2. Choose a specific scenario folder to run.
    3. Pass the folder path to `run_scenario()`.

    Notes:
    - The script processes the configuration files in the scenario folder.
    - Outputs (logs, vocab, preprocessed data, etc.) are saved under `/runs/<scenario_folder_name>`.
    """

    # Specify the path to the scenario folder
    SCENARIO_FOLDER = "scenarios/scenario_1"  # Update with your desired scenario folder path

    # Validate scenario folder existence
    if not os.path.exists(SCENARIO_FOLDER):
        raise FileNotFoundError(f"Scenario folder not found: {SCENARIO_FOLDER}")

    # Run the scenario
    run_scenario(SCENARIO_FOLDER)


if __name__ == "__main__":
    main()
