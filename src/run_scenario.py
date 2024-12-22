import argparse
import json
import os
from factory import create_preprocessor, create_vocabulary
from errors import ConstructionError, PreprocessingError
from utils.logging_utils import setup_logging
import shutil


def load_config(config_path):
    """
    Load the configuration from the specified file path.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, 'r') as config_file:
        return json.load(config_file)


def run_scenario(config_path):
    """
    Run the scenario using the provided configuration file.
    """
    # Create scenario folder in runs
    config_name = os.path.splitext(os.path.basename(config_path))[0]  # Extract config file name without extension
    scenario_dir = os.path.join("runs", config_name)
    os.makedirs(scenario_dir, exist_ok=True)

    # Copy config file to scenario folder (preserve original in scenarios)
    copied_config_path = os.path.join(scenario_dir, os.path.basename(config_path))
    shutil.copyfile(config_path, copied_config_path)

    # Set up loggers
    system_level = 20
    training_level = 10
    system_logger, training_logger = setup_logging(system_level, training_level, scenario_dir)

    try:
        # Load the configuration
        config = load_config(copied_config_path)

        # Generate vocabulary
        vocab = create_vocabulary(config)
        vocab_file_path = os.path.join(scenario_dir, "vocab.json")
        vocab.save(vocab_file_path)

        # Create and process preprocessor
        preprocessor = create_preprocessor(config, vocab)
        sequence = 'ACGTGCTTCGATCACGCTAGCTCGATCGATAGATCGCTCGCTCGCATAGCTAGATAGCGCGCATAGCTCCCATAGAACT'

        # Phase 1:
        mapped_sentence = preprocessor.process(sequence)
        system_logger.info(f"Processed sequence: {mapped_sentence}")

        # Additional phases can be added here.

    except FileNotFoundError as e:
        system_logger.error(f"Configuration file error: {e}")
    except PreprocessingError as e:
        system_logger.error(f"Preprocessing error occurred: {e}")
    except ConstructionError as e:
        system_logger.critical(f"Critical configuration error during model construction: {e}")
    except Exception as e:
        system_logger.critical(f"Unexpected error: {e}")


if __name__ == "__main__":
    # Parse arguments from the command line
    parser = argparse.ArgumentParser(description="Run a scenario with a given config file.")
    parser.add_argument(
        "config_path",
        help="Path to the configuration file."
    )

    args = parser.parse_args()

    # Run the scenario with the provided config file
    run_scenario(args.config_path)
