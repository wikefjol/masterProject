import argparse
import json
import os
import shutil
import hashlib
import pandas as pd
from factory import create_preprocessor, create_vocabulary
from errors import ConstructionError, PreprocessingError
from utils.logging_utils import setup_logging
from preparer import SequenceDataPreparer


def hash_file(file_path):
    """Compute a hash of the file to ensure raw data integrity."""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def create_unique_scenario_dir(base_dir, config_name):
    """Create a unique directory by appending a number if the directory already exists."""
    scenario_dir = os.path.join(base_dir, config_name)
    counter = 0
    while os.path.exists(scenario_dir):
        counter += 1
        scenario_dir = f"{os.path.join(base_dir, config_name)}_{counter}"
    os.makedirs(scenario_dir)
    return scenario_dir


def prepare_data(fasta_file, output_dir, test_size, random_seed, logger, scenario_dir=None, force_reprocess=False):
    """Prepare training and testing data with options for reuse or run-specific preparation."""
    fasta_hash = hash_file(fasta_file)
    metadata = {
        "test_size": test_size,
        "random_seed": random_seed,
        "fasta_hash": fasta_hash,
    }

    # Decide where to store data
    if force_reprocess and scenario_dir:
        data_dir = scenario_dir  # Save to run-specific directory
    else:
        data_dir = os.path.join(output_dir, f"scenario_testsize_{test_size}_seed_{random_seed}")
        os.makedirs(data_dir, exist_ok=True)

    train_file = os.path.join(data_dir, "train_sequences.csv")
    test_file = os.path.join(data_dir, "test_sequences.csv")
    metadata_file = os.path.join(data_dir, "prepared_metadata.json")

    if not force_reprocess and os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            if json.load(f) == metadata:
                logger.info(f"Prepared data already exists in {data_dir}. Skipping preparation.")
                return train_file, test_file

    # Prepare data
    logger.info(f"Preparing data in {data_dir}...")
    preparer = SequenceDataPreparer(fasta_file, data_dir)
    sequences_df = preparer.parse_fasta_to_dataframe()
    train_df, test_df = preparer.split_data(sequences_df, test_size, random_seed)
    preparer.save_dataframe_to_csv(train_df, "train_sequences.csv")
    preparer.save_dataframe_to_csv(test_df, "test_sequences.csv")

    with open(metadata_file, 'w') as f:
        json.dump(metadata, f)

    logger.info(f"Data preparation complete: Train: {train_file}, Test: {test_file}")
    return train_file, test_file


def process_sequences(train_file, preprocessor, logger, limit=None):
    """Preprocess sequences from the training data."""
    train_df = pd.read_csv(train_file)
    preprocessed_data = []

    for idx, row in train_df.iterrows():
        if limit and idx >= limit:
            break
        sequence = row["Sequence"]
        preprocessed_sequence = preprocessor.process(sequence)
        row_data = row.to_dict()
        row_data["Preprocessed_Sequence"] = preprocessed_sequence
        preprocessed_data.append(row_data)
        logger.info(f"Preprocessed sequence {idx + 1}: {preprocessed_sequence}")

    return pd.DataFrame(preprocessed_data)


def setup_environment(config_path):
    """Set up the scenario directory and copy configuration."""
    config_name = os.path.splitext(os.path.basename(config_path))[0]
    scenario_dir = create_unique_scenario_dir("runs", config_name)
    shutil.copy(config_path, os.path.join(scenario_dir, "config.json"))
    return scenario_dir

def load_config(config_path):
    """Load the configuration from the specified file path."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, 'r') as config_file:
        return json.load(config_file)


def run_scenario(config_path):
    """Run the scenario using the provided configuration."""
    scenario_dir = setup_environment(config_path)
    system_logger, _ = setup_logging(11, 10, scenario_dir)

    try:
        config = load_config(config_path)
        fasta_file = config["fasta_file"]
        output_dir = config["prepared_data_dir"]
        test_size = config["test_size"]
        random_seed = config["random_seed"]

        force_reprocess = config.get("force_reprocess", False)  # New option in config
        train_file, _ = prepare_data(fasta_file, output_dir, test_size, random_seed, system_logger, scenario_dir, force_reprocess)

        vocab = create_vocabulary(config)
        vocab.save(os.path.join(scenario_dir, "vocab.json"))

        preprocessor = create_preprocessor(config, vocab)
        preprocessed_df = process_sequences(train_file, preprocessor, system_logger, limit=10)

        preprocessed_df_path = os.path.join(scenario_dir, "preprocessed_data.csv")
        preprocessed_df.to_csv(preprocessed_df_path, index=False)
        system_logger.info(f"Saved preprocessed data to {preprocessed_df_path}")

    except Exception as e:
        system_logger.error(f"An error occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a scenario.")
    parser.add_argument("config_path", help="Path to the configuration file.")
    args = parser.parse_args()
    run_scenario(args.config_path)
