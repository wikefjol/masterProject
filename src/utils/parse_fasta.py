def parse_fasta_custom(fasta_path):
    """
    Custom parser for FASTA files.

    Args:
        fasta_path (str): Path to the FASTA file.

    Returns:
        list[dict]: A list of dictionaries with 'ID' and 'Sequence'.
    """
    sequences = []
    with open(fasta_path, "r") as file:
        current_id = None
        current_sequence = []
        for line in file:
            line = line.strip()
            if line.startswith(">"):
                # Save the previous sequence
                if current_id:
                    sequences.append({"ID": current_id, "Sequence": "".join(current_sequence)})
                # Start a new sequence
                current_id = line[1:].split()[0]  # Capture only the first part of the ID
                current_sequence = []
            else:
                current_sequence.append(line)
        # Add the last sequence
        if current_id:
            sequences.append({"ID": current_id, "Sequence": "".join(current_sequence)})
    return sequences

# Example usage
fasta_file = "/Users/filipberntsson/Documents/Studies/Thesis/Programming/BarcodeClassifier/data/raw/raw.fasta"
sequences = parse_fasta_custom(fasta_file)

# Print in a neat format
cnt = 0
for seq in sequences:
    cnt +=1
    print(f"ID: {seq['ID']}, Sequence: {seq['Sequence']}")
    if cnt>9:
        break 
