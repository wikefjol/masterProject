import argparse
import json
import os
import pandas as pd
from factory import create_preprocessor, create_vocabulary
from errors import ConstructionError, PreprocessingError
from utils.logging_utils import setup_logging
from preparer import SequenceDataPreparer
from tqdm import tqdm  # Ensure tqdm is imported


def load_configs(scenario_folder):
    """Load all configuration files from the scenario folder."""
    general_config_path = os.path.join(scenario_folder, "general_config.json")
    pretraining_config_path = os.path.join(scenario_folder, "pretraining_config.json")
    finetuning_config_path = os.path.join(scenario_folder, "finetuning_config.json")

    if not all(os.path.exists(path) for path in [general_config_path, pretraining_config_path, finetuning_config_path]):
        raise FileNotFoundError(f"One or more config files are missing in the scenario folder: {scenario_folder}")

    with open(general_config_path, 'r') as f:
        general_config = json.load(f)
    with open(pretraining_config_path, 'r') as f:
        pretraining_config = json.load(f)
    with open(finetuning_config_path, 'r') as f:
        finetuning_config = json.load(f)

    return general_config, pretraining_config, finetuning_config


def prepare_data(fasta_file, output_dir, test_size, random_seed, logger, scenario_dir=None, force_reprocess=False):
    """Prepare training and testing data."""
    preparer = SequenceDataPreparer(fasta_file, output_dir)
    train_file, test_file = preparer.prepare(test_size, random_seed)
    logger.info(f"Data prepared: Train file: {train_file}, Test file: {test_file}")
    return train_file, test_file


def run_pretraining(pretraining_config, scenario_dir, logger):
    """Run the pretraining process."""
    logger.info("Running pretraining...")
    fasta_file = pretraining_config["fasta_file"]
    output_dir = os.path.abspath(pretraining_config["prepared_data_dir"])
    test_size = pretraining_config["test_size"]
    random_seed = pretraining_config["random_seed"]
    force_reprocess = pretraining_config.get("force_reprocess", False)

    # Step 1: Data Preparation
    os.makedirs(output_dir, exist_ok=True)
    train_file, _ = prepare_data(
        fasta_file, output_dir, test_size, random_seed, logger, scenario_dir, force_reprocess
    )

    # Step 2: Create Vocabulary and Preprocessor
    vocab = create_vocabulary(pretraining_config)
    vocab.save(os.path.join(scenario_dir, "pretraining_vocab.json"))
    preprocessor = create_preprocessor(pretraining_config, vocab)

    # Step 3: Load Dataset and Initialize DataLoader (Placeholder)
    logger.info("Initializing Dataset and DataLoader (Placeholder)")
    train_df = pd.read_csv(train_file)
    preprocessed_data = [
        preprocessor.process(sequence) for sequence in tqdm(train_df["Sequence"].tolist(), desc="Processing sequences")
    ]

    # Save preprocessed data for use in training
    preprocessed_file = os.path.join(scenario_dir, "pretraining_data.csv")
    pd.DataFrame({"Sequence": preprocessed_data}).to_csv(preprocessed_file, index=False)
    logger.info(f"Pretraining data saved to {preprocessed_file}")

    # Step 4: Call Trainer (Placeholder)
    logger.info("Training logic to be implemented with Trainer class (Placeholder).")

def run_finetuning(finetuning_config, scenario_dir, logger):
    """Run the finetuning process."""
    logger.info("Running finetuning...")
    fasta_file = finetuning_config["fasta_file"]
    output_dir = os.path.abspath(finetuning_config["prepared_data_dir"])
    test_size = finetuning_config["test_size"]
    random_seed = finetuning_config["random_seed"]
    force_reprocess = finetuning_config.get("force_reprocess", False)

    # Prepare data
    os.makedirs(output_dir, exist_ok=True)  # Ensure directory exists
    train_file, test_file = prepare_data(
        fasta_file, output_dir, test_size, random_seed, logger, scenario_dir, force_reprocess
    )

    # Load pretraining vocabulary
    vocab_path = os.path.join(scenario_dir, "pretraining_vocab.json")
    vocab = create_vocabulary(finetuning_config)
    vocab.load(vocab_path)

    preprocessor = create_preprocessor(finetuning_config, vocab)

    # Process sequences
    train_df = pd.read_csv(train_file)
    preprocessed_data = [
        preprocessor.process(sequence) for sequence in train_df["Sequence"].tolist()
    ]

    # Save preprocessed data
    preprocessed_file = os.path.join(scenario_dir, "finetuning_data.csv")
    pd.DataFrame({"Sequence": preprocessed_data}).to_csv(preprocessed_file, index=False)
    logger.info(f"Finetuning data saved to {preprocessed_file}")


def run_scenario(scenario_folder):
    """Run the scenario using the provided scenario folder."""
    general_config, pretraining_config, finetuning_config = load_configs(scenario_folder)

    # Setup logging
    log_dir = general_config.get("log_dir", "runs/logs")
    system_log_level = general_config.get("system_log_level", 20)
    training_log_level = general_config.get("training_log_level", 20)
    scenario_dir = os.path.join("runs", os.path.basename(scenario_folder))

    os.makedirs(scenario_dir, exist_ok=True)
    system_logger, _ = setup_logging(system_log_level, training_log_level, scenario_dir)

    try:
        # Handle pretraining
        if pretraining_config.get("enabled", False):
            #TODO: Change the structure here, so that we have one call that prepares the trainingdata, then one call that later will perform the actual training, with a trainier object. 
            # #here we will likely need both a dataset class, and a dataloader the "run_pretraining" method will eventually have to be entirely rewritten. Mirroring this will have to happen for the finetuning case later. 
            print("Preparing pretraining data")
            run_pretraining(pretraining_config, scenario_dir, system_logger)

        # Handle finetuning
        #if finetuning_config.get("enabled", False):
            #run_finetuning(finetuning_config, scenario_dir, system_logger)

    except Exception as e:
        system_logger.error(f"An error occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a scenario.")
    parser.add_argument("scenario_folder", help="Path to the scenario folder.")
    args = parser.parse_args()
    run_scenario(args.scenario_folder)
