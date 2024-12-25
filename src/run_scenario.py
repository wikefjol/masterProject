import argparse
import json
import os
import pandas as pd
from factory import create_preprocessor, create_vocabulary
from utils.logging_utils import setup_logging
from preparer import SequenceDataPreparer
from tqdm import tqdm
import torch
import random
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertForMaskedLM
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from transformers import get_scheduler
from torch.cuda.amp import GradScaler, autocast


def create_attention_mask(input_ids):
    return (input_ids != 0).long()  # Padding is assumed to be 0


def pretrain_model(sequences, model, vocab_size, mask_token_id, training_logger, device="cpu", 
                   epochs=1, batch_size=32, learning_rate=5e-5, max_grad_norm=1.0, val_sequences=None):
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scaler = GradScaler()  # Mixed precision scaler
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=epochs * len(sequences) // batch_size)
    
    for epoch in range(epochs):
        training_logger.info(f"Starting epoch {epoch + 1}/{epochs}")
        
        # Mask and create DataLoader for training set
        input_ids, labels = mask_inputs(sequences, mask_token_id=mask_token_id, vocab_size=vocab_size)
        attention_mask = create_attention_mask(input_ids)
        dataset = TensorDataset(input_ids, attention_mask, labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch + 1}/{epochs}")
        
        for idx, batch in progress_bar:
            input_ids, attention_mask, labels = [x.to(device) for x in batch]

            optimizer.zero_grad()
            with autocast():  # Mixed precision training
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()  # Adjust learning rate
            
            epoch_loss += loss.item()
            if idx % 10 == 0:
                training_logger.info(f"Epoch {epoch + 1}, Step {idx}: Loss = {loss.item()}")
            progress_bar.set_postfix({"loss": loss.item()})

        # Validation Loss Calculation
        if val_sequences:
            val_loss = validate_model(val_sequences, model, mask_token_id, vocab_size, batch_size, device)
            training_logger.info(f"Epoch {epoch + 1} Validation Loss: {val_loss}")

        average_loss = epoch_loss / len(dataloader)
        training_logger.info(f"Epoch {epoch + 1} completed. Average Loss: {average_loss}")

        # Save checkpoint
        checkpoint_path = f"checkpoint_epoch_{epoch + 1}.pt"
        torch.save(model.state_dict(), checkpoint_path)
        training_logger.info(f"Model checkpoint saved to {checkpoint_path}")

def validate_model(val_sequences, model, mask_token_id, vocab_size, batch_size, device):
    model.eval()
    val_loss = 0.0

    input_ids, labels = mask_inputs(val_sequences, mask_token_id=mask_token_id, vocab_size=vocab_size)
    attention_mask = create_attention_mask(input_ids)
    dataset = TensorDataset(input_ids, attention_mask, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, labels = [x.to(device) for x in batch]
            with autocast():  # Mixed precision
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()

    return val_loss / len(dataloader)


def mask_inputs(sequences, mask_token_id, vocab_size, mask_prob=0.15):
    input_ids = []
    labels = []

    for seq in sequences:
        seq_labels = [-100] * len(seq)
        seq_ids = list(seq)

        for i in range(len(seq_ids)):
            if random.random() < mask_prob:
                seq_labels[i] = seq_ids[i]
                if random.random() < 0.8:
                    seq_ids[i] = mask_token_id
                elif random.random() < 0.5:
                    seq_ids[i] = random.randint(0, vocab_size - 1)

        input_ids.append(seq_ids)
        labels.append(seq_labels)

    return torch.tensor(input_ids), torch.tensor(labels)


def load_configs(scenario_folder):
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


def prepare_data(fasta_file, output_dir, test_size, random_seed, system_logger):
    preparer = SequenceDataPreparer(fasta_file, output_dir)
    train_file, test_file = preparer.prepare(test_size, random_seed)
    system_logger.info(f"Data prepared: Train file: {train_file}, Test file: {test_file}")
    return train_file, test_file


def run_pretraining(pretraining_config, general_config, scenario_dir, system_logger, training_logger):
    system_logger.info("Running pretraining...")

    system_logger.info("Preparing data")
    train_file, test_file = prepare_data(
        fasta_file=pretraining_config["fasta_file"],
        output_dir=pretraining_config["prepared_data_dir"],
        test_size=pretraining_config["test_size"],
        random_seed=pretraining_config["random_seed"],
        system_logger=system_logger,
    )

    system_logger.info("Constructing vocab")
    vocab = create_vocabulary(general_config)
    vocab.save(os.path.join(scenario_dir, "vocab.json"))

    system_logger.info("Constructing preprocesser")
    preprocessor = create_preprocessor(
        general_config,
        vocab,
        augmentation_config=pretraining_config["augmentation_strategy"],
    )

    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(test_file)

    # Preprocess sequences
    preprocessed_train = []
    preprocessed_val = []

    # Training sequences
    for sequence in tqdm(train_df["Sequence"], desc="Processing training sequences"):
        preprocessed_train.append(preprocessor.process(sequence))

    # Validation sequences
    for sequence in tqdm(val_df["Sequence"], desc="Processing validation sequences"):
        preprocessed_val.append(preprocessor.process(sequence))

    sequences = preprocessed_train
    val_sequences = preprocessed_val

    # Model Setup
    model = BertForMaskedLM.from_pretrained("bert-base-uncased")
    model.resize_token_embeddings(len(vocab))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 72
    pretrain_model(
        sequences, 
        model, 
        vocab_size=len(vocab), 
        mask_token_id=4, 
        training_logger=training_logger, 
        device=device, 
        epochs=pretraining_config["num_epochs"], 
        batch_size=batch_size, 
        val_sequences=val_sequences,
    )

    model_save_path = os.path.join(scenario_dir, "pretrained_model")
    os.makedirs(model_save_path, exist_ok=True)
    model.save_pretrained(model_save_path)
    vocab.save(os.path.join(model_save_path, "vocab.json"))
    system_logger.info(f"Model and vocab saved to {model_save_path}")


def run_finetuning(finetuning_config, general_config, scenario_dir, logger):
    pass


def run_scenario(scenario_folder):
    general_config, pretraining_config, finetuning_config = load_configs(scenario_folder)
    scenario_dir = os.path.join("runs", os.path.basename(scenario_folder))

    os.makedirs(scenario_dir, exist_ok=True)
    system_logger, training_logger = setup_logging(
        system_level=general_config.get("system_log_level", 20),
        training_level=general_config.get("training_log_level", 20),
        log_dir=scenario_dir,
    )

    try:
        if pretraining_config.get("enabled", False):
            system_logger.info("Preparing pretraining data")
            run_pretraining(pretraining_config, general_config, scenario_dir, system_logger, training_logger)
    except Exception as e:
        system_logger.error(f"An error occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a scenario.")
    parser.add_argument("scenario_folder", help="Path to the scenario folder.")
    args = parser.parse_args()
    run_scenario(args.scenario_folder)
