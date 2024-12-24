import os
import pandas as pd
from Bio import SeqIO
from sklearn.model_selection import train_test_split


class SequenceDataPreparer:
    """
    Handles the parsing, preparation, and organization of sequence data
    from raw FASTA files for downstream tasks.
    """

    def __init__(self, fasta_path: str, output_dir: str):
        """
        Initialize the SequenceDataPreparer.

        Args:
            fasta_path (str): Path to the raw FASTA file.
            output_dir (str): Directory to store prepared outputs.
        """
        self.fasta_path = fasta_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def parse_fasta_to_dataframe(self) -> pd.DataFrame:
        """
        Parse a FASTA file into a Pandas DataFrame with split taxonomic levels.

        Returns:
            pd.DataFrame: DataFrame containing IDs, sequences, and taxonomy columns.
        """
        records = []
        for record in SeqIO.parse(self.fasta_path, "fasta"):
            # Extract ID and Sequence
            record_id = record.id
            sequence = str(record.seq)
            
            # Parse the taxonomy string from the ID
            # Assuming the taxonomy string is delimited by `;` and starts with `k__`
            parts = record_id.split('|')
            taxonomy_string = parts[-1] if parts[-1].startswith('k__') else ''
            taxonomic_levels = self._parse_taxonomy(taxonomy_string)

            # Combine all parsed data into a single record
            record_data = {
                "ID": record_id,
                "Sequence": sequence,
                **taxonomic_levels  # Unpack the taxonomy columns
            }
            records.append(record_data)

        return pd.DataFrame(records)

    def _parse_taxonomy(self, taxonomy_string: str) -> dict:
        """
        Parse a taxonomy string into individual taxonomic levels.

        Args:
            taxonomy_string (str): The full taxonomy string (e.g., "k__Fungi;p__Basidiomycota;c__Agaricomycetes;...").

        Returns:
            dict: A dictionary with taxonomic levels as keys (e.g., "Kingdom", "Phylum").
        """
        taxonomic_levels = {
            "Kingdom": None,
            "Phylum": None,
            "Class": None,
            "Order": None,
            "Family": None,
            "Genus": None,
            "Species": None,
        }

        if taxonomy_string:
            # Split the taxonomy string by `;` and parse each level
            levels = taxonomy_string.split(';')
            for level in levels:
                if level.startswith('k__'):
                    taxonomic_levels["Kingdom"] = level[3:]
                elif level.startswith('p__'):
                    taxonomic_levels["Phylum"] = level[3:]
                elif level.startswith('c__'):
                    taxonomic_levels["Class"] = level[3:]
                elif level.startswith('o__'):
                    taxonomic_levels["Order"] = level[3:]
                elif level.startswith('f__'):
                    taxonomic_levels["Family"] = level[3:]
                elif level.startswith('g__'):
                    taxonomic_levels["Genus"] = level[3:]
                elif level.startswith('s__'):
                    taxonomic_levels["Species"] = level[3:]

        return taxonomic_levels

    def save_dataframe_to_csv(self, df: pd.DataFrame, filename: str) -> None:
        """
        Save a DataFrame to a CSV file in the output directory.

        Args:
            df (pd.DataFrame): DataFrame to save.
            filename (str): Name of the CSV file.
        """
        output_path = os.path.join(self.output_dir, filename)
        df.to_csv(output_path, index=False)

    def split_data(self, df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> tuple:
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

    def prepare(self) -> None:
        """
        Execute the full sequence data preparation pipeline:
        1. Parse FASTA file into a DataFrame.
        2. Save the full DataFrame to a CSV file.
        3. Split the data into training and testing sets.
        4. Save the training and testing sets to separate CSV files.
        """
        # Step 1: Parse FASTA file
        sequences_df = self.parse_fasta_to_dataframe()

        # Step 2: Save full dataset
        self.save_dataframe_to_csv(sequences_df, "prepared_sequences.csv")

        # Step 3: Split the data
        train_df, test_df = self.split_data(sequences_df)

        # Step 4: Save training and testing sets
        self.save_dataframe_to_csv(train_df, "train_prepared_sequences.csv")
        self.save_dataframe_to_csv(test_df, "test_prepared_sequences.csv")

        print(
            f"Prepared {len(sequences_df)} sequences. "
            f"Training set: {len(train_df)}, Test set: {len(test_df)}"
        )


# Example usage:
if __name__ == "__main__":
    fasta_file = "data/raw/raw.fasta"
    output_directory = "data/prepared"

    preparer = SequenceDataPreparer(fasta_path=fasta_file, output_dir=output_directory)
    preparer.prepare()
