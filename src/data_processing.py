import os
import pandas as pd
from Bio import SeqIO
from typing import Tuple
from sklearn.model_selection import train_test_split


class DataProcessor:
    """
    Handles the entire data processing pipeline, from reading raw data to preparing it
    for further stages such as preprocessing.
    """
    def __init__(self, fasta_path: str, output_dir: str):
        """
        Initialize the DataProcessor.

        Args:
            fasta_path (str): Path to the raw FASTA file.
            output_dir (str): Directory to save processed datasets.
        """
        self.fasta_path = fasta_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def parse_fasta(self) -> pd.DataFrame:
        """
        Parse the FASTA file into a Pandas DataFrame.

        Returns:
            pd.DataFrame: DataFrame with 'ID' and 'Sequence' columns.
        """
        records = [
            {"ID": record.id, "Sequence": str(record.seq)}
            for record in SeqIO.parse(self.fasta_path, "fasta")
        ]
        return pd.DataFrame(records)

    def save_dataframe(self, df: pd.DataFrame, filename: str):
        """
        Save a DataFrame as a CSV file.

        Args:
            df (pd.DataFrame): DataFrame to save.
            filename (str): Name of the output file.
        """
        output_path = os.path.join(self.output_dir, filename)
        df.to_csv(output_path, index=False)
        print(f"Saved: {output_path}")

    def split_data(self, df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split the DataFrame into training and testing sets.

        Args:
            df (pd.DataFrame): DataFrame to split.
            test_size (float): Proportion of the dataset to include in the test split.
            random_state (int): Random seed.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Training and testing DataFrames.
        """
        return train_test_split(df, test_size=test_size, random_state=random_state)

    def process_data(self, test_size: float = 0.2):
        """
        Execute the full data processing pipeline.

        Args:
            test_size (float): Proportion of the dataset to include in the test split.
        """
        print("Parsing FASTA file...")
        sequences_df = self.parse_fasta()

        print("Splitting data into training and testing sets...")
        train_df, test_df = self.split_data(sequences_df, test_size=test_size)

        print("Saving datasets...")
        self.save_dataframe(sequences_df, "sequences.csv")
        self.save_dataframe(train_df, "train_sequences.csv")
        self.save_dataframe(test_df, "test_sequences.csv")

        print(f"Processing complete: {len(sequences_df)} sequences processed. "
              f"Training set: {len(train_df)}, Test set: {len(test_df)}")


if __name__ == "__main__":
    # Example usage
    fasta_file = "data/raw/raw.fasta"
    output_directory = "data/processed"

    processor = DataProcessor(fasta_path=fasta_file, output_dir=output_directory)
    processor.process_data(test_size=0.2)
