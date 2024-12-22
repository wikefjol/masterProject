import pandas as pd
from Bio import SeqIO
from sklearn.model_selection import train_test_split

def parse_fasta_to_dataframe(fasta_path):
    """
    Parse a FASTA file into a Pandas DataFrame with 'ID' and 'Sequence' columns.

    Args:
        fasta_path (str): Path to the FASTA file.

    Returns:
        pd.DataFrame: DataFrame containing IDs and sequences.
    """
    records = []
    for record in SeqIO.parse(fasta_path, "fasta"):
        records.append({"ID": record.id, "Sequence": str(record.seq)})
    return pd.DataFrame(records)

def save_dataframe_to_csv(df, output_path):
    """
    Save a DataFrame to a CSV file.

    Args:
        df (pd.DataFrame): DataFrame to save.
        output_path (str): Path to save the CSV file.
    """
    df.to_csv(output_path, index=False)

def split_data(df, test_size=0.2, random_state=42):
    """
    Split the DataFrame into training and testing sets.

    Args:
        df (pd.DataFrame): DataFrame to split.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed.

    Returns:
        tuple: Training and testing DataFrames.
    """
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    return train_df, test_df

# Example usage:
fasta_file = "data/raw/raw.fasta"
output_csv = "output/sequences.csv"
train_csv = "output/train_sequences.csv"
test_csv = "output/test_sequences.csv"

# Parse the FASTA file to a DataFrame
sequences_df = parse_fasta_to_dataframe(fasta_file)

# Save the DataFrame to a CSV file
save_dataframe_to_csv(sequences_df, output_csv)

# Split into training and testing datasets
train_df, test_df = split_data(sequences_df, test_size=0.2)

# Save the training and testing datasets to CSV files
save_dataframe_to_csv(train_df, train_csv)
save_dataframe_to_csv(test_df, test_csv)

print(f"Processed {len(sequences_df)} sequences. Training set: {len(train_df)}, Test set: {len(test_df)}")
