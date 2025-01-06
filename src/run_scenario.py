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
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import BertConfig, BertForMaskedLM, DataCollator
from torch.optim import AdamW

from typing import List, Dict
import torch
from transformers import DataCollator


class CustomMLMCollator:
    def __init__(self, mask_prob=0.15, pad_token_id=0, mask_token_id=103, vocab_size=100):
        """
        Custom data collator for Masked Language Modeling without formal tokenizer.
        Args:
            mask_prob (float): Probability of masking a token for MLM.
            pad_token_id (int): Token ID used for padding sequences.
            mask_token_id (int): Token ID representing the [MASK] token.
            vocab_size (int): Size of the vocabulary (used for random token replacement).
        """
        self.mask_prob = mask_prob
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        self.vocab_size = vocab_size

    def __call__(self, batch: List[Dict[str, List[int]]]):
        """
        Processes a batch of sequences for MLM.
        Args:
            batch (List[Dict[str, List[int]]]): List of samples with 'sequence' key.

        Returns:
            dict: Dictionary with input_ids, attention_mask, and labels.
        """
        # Extract sequences
        sequences = [item['sequence'] for item in batch]
        
        # Determine max length in the batch for padding
        max_length = max(len(seq) for seq in sequences)
        
        # Pad sequences and prepare labels
        input_ids, labels = [], []
        for seq in sequences:
            padded_seq = seq + [self.pad_token_id] * (max_length - len(seq))
            input_ids.append(padded_seq)
            
            # Mask tokens for MLM
            label_seq = [-100] * len(padded_seq)  # Initialize labels with -100
            for i in range(len(seq)):
                if torch.rand(1).item() < self.mask_prob:
                    label_seq[i] = padded_seq[i]  # Preserve original token as label
                    if torch.rand(1).item() < 0.8:
                        padded_seq[i] = self.mask_token_id  # Replace with [MASK]
                    elif torch.rand(1).item() < 0.5:
                        padded_seq[i] = torch.randint(0, self.vocab_size, (1,)).item()  # Replace with random token
            
            labels.append(label_seq)

        
        # Convert to tensors
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        attention_mask = (input_ids != self.pad_token_id).long()  # Attention mask
        print(f"Inside CustomMLMCollator __call__ Batch Input IDs Shape: {input_ids.shape}")
        print(f"Inside CustomMLMCollator __call__ Batch Input IDs Shape: {input_ids.shape}")
        print(f"Inside CustomMLMCollator __call__ Batch Labels Shape: {labels.shape}")

        assert input_ids.size(0) == labels.size(0), "Batch size mismatch!"
        assert input_ids.size(1) == labels.size(1), "Sequence length mismatch!"

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

class CustomDataset(Dataset):
    def __init__(self, data):
        """
        Custom Dataset for integer-mapped sequences.
        Args:
            data (List[Dict]): List of samples with 'sequence' as key.
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def create_attention_mask(input_ids):
    return (input_ids != 0).long()  # Padding is assumed to be 0


def pretrain_model(sequences, model, vocab_size, mask_token_id, training_logger, device="cpu", 
                   epochs=1, batch_size=32, learning_rate=5e-5, max_grad_norm=1.0, val_sequences=None):
    print("Here 1")
    model.to(device)
    print("Here 2")
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    print("Here 3")
    scaler = GradScaler()  # Mixed precision scaler
    print("Here 4")
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=epochs * len(sequences) // batch_size)
    print("Here 5")
    
    for epoch in range(epochs):
        training_logger.info(f"Starting epoch {epoch + 1}/{epochs}")
        
        # Mask and create DataLoader for training set
        print("First halt")
        input_ids, labels = mask_inputs(sequences, mask_token_id=mask_token_id, vocab_size=vocab_size)
        print("Second halt")
        attention_mask = create_attention_mask(input_ids)
        print("Third halt")
        dataset = TensorDataset(input_ids, attention_mask, labels)
        print("Fourth halt")
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        print("Fifth halt")

        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch + 1}/{epochs}")
        
        for idx, batch in progress_bar:

            input_ids, attention_mask, labels = [x.to(device) for x in batch]
            assert torch.all(input_ids < vocab_size), "Vocab size mixup"
            print(f"In for idx, batch, in progress_bar Batch Input IDs Shape: {input_ids.shape}")
            print(f"In for idx, batch, in progress_bar Batch Input IDs Shape: {input_ids.shape}")
            print(f"In for idx, batch, in progress_bar Batch Labels Shape: {labels.shape}")
            optimizer.zero_grad()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            print(f"In for idx, batch, in progress_bar outputs shape: {outputs.shape}")

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

    print("Preprocessing training and validation sequences...")


    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(test_file)

    # Training sequences
    preprocessed_train = []
    for sequence in tqdm(train_df["Sequence"][:1000], desc="Processing training sequences, limited to 1000 for proof of concept"):
        processed_sequence = preprocessor.process(sequence)
        preprocessed_train.append({"sequence": processed_sequence})

    # Validation sequences
    preprocessed_val = []
    for sequence in tqdm(val_df["Sequence"][:100], desc="Processing validation sequences"):
        processed_sequence = preprocessor.process(sequence)
        preprocessed_val.append({"sequence": processed_sequence})

    # train_dataset = CustomDataset(preprocessed_train)
    # val_dataset = CustomDataset(preprocessed_val)

    # collator = CustomMLMCollator(mask_prob=0.15, pad_token_id=0, mask_token_id=103, vocab_size=100)

    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collator)
    # val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collator)


    # model_config = BertConfig(
    # vocab_size=len(vocab),     # Match your vocabulary size
    # hidden_size=256,           # Dimensionality of the embeddings and hidden states
    # num_attention_heads=4,     # Number of attention heads
    # num_hidden_layers=4,       # Number of transformer layers
    # intermediate_size=512,     # Size of the feed-forward layer
    # max_position_embeddings=512,  # Max sequence length
    # )

    # system_logger.info(f"Initializing model with config: {model_config}")
    # # Initialize a BERT model for MLM
    # model = BertForMaskedLM(model_config)

    # #TODO: Create a new bert model, which is entirely untrained
    # #TODO: Pretrain the bert model with mlm
    # # Define training parameters
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)

    # optimizer = AdamW(model.parameters(), lr=5e-5)

    # torch.optim.AdamW
    # epochs = 3  # Number of epochs
    # max_grad_norm = 1.0  # Gradient clipping

    # # Training loop
    # for epoch in range(epochs):
    #     system_logger.info("model.train")
    #     model.train()
    #     total_train_loss = 0
    #     progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} Training")

    #     system_logger.info("for batch in progress_bar")
    #     for batch in progress_bar:
    #         system_logger.info("for batch in progress_bar")
    #         optimizer.zero_grad()

    #         # Move data to the correct device
    #         system_logger.info("Move data to the correct device")
    #         input_ids = batch['input_ids'].to(device)
    #         attention_mask = batch['attention_mask'].to(device)
    #         labels = batch['labels'].to(device)

    #         # Forward pass
    #         system_logger.info("Forward pass")
    #         outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    #         loss = outputs.loss

    #         # Backward pass
    #         system_logger.info("Backward pass")
    #         loss.backward()
    #         clip_grad_norm_(model.parameters(), max_grad_norm)
    #         optimizer.step()

    #         total_train_loss += loss.item()
    #         progress_bar.set_postfix({"loss": loss.item()})

    #     avg_train_loss = total_train_loss / len(train_loader)
    #     print(f"Epoch {epoch + 1} Train Loss: {avg_train_loss:.4f}")

    #     # Validation loop
    #     model.eval()
    #     total_val_loss = 0
    #     with torch.no_grad():
    #         for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} Validation"):
    #             input_ids = batch['input_ids'].to(device)
                
    #             attention_mask = batch['attention_mask'].to(device)
    #             labels = batch['labels'].to(device)

    #             # Forward pass
    #             outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    #             loss = outputs.loss
    #             total_val_loss += loss.item()

    #     avg_val_loss = total_val_loss / len(val_loader)
    #     print(f"Epoch {epoch + 1} Validation Loss: {avg_val_loss:.4f}")

    # # Save the pretrained model
    # model.save_pretrained("pretrained_bert_mlm")
    # print("Model saved to 'pretrained_bert_mlm'.")



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
