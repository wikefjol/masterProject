# vocab.py
import json
from typing import Dict, List
from abc import ABC, abstractmethod
from itertools import product
from utils.logging_utils import with_logging

class VocabConstructor(ABC):
    """
    Abstract base class for vocabulary constructors.
    """
    @abstractmethod
    def build_vocab(self, data: List[str], vocab: 'Vocabulary') -> None:
        """
        Build the vocabulary from the provided data.
        
        Args:
            data (List[str]): List of raw sequences.
            vocab (Vocabulary): Vocabulary instance to populate.
        """
        pass

class Vocabulary:
    """
    Vocabulary class to handle token-ID mappings.
    """
    def __init__(self):
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.pad_token = 'PAD'
        self.unk_token = 'UNK'
        self.pad_id = 0
        self.unk_id = 1
        # Initialize with PAD and UNK tokens
        self.add_token(self.pad_token)
        self.add_token(self.unk_token)
    
    def add_token(self, token: str):
        if token not in self.token_to_id:
            idx = len(self.token_to_id)
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
    
    def get_id(self, token: str) -> int:
        return self.token_to_id.get(token, self.unk_id)  # Default to UNK ID
    
    def get_token(self, idx: int) -> str:
        return self.id_to_token.get(idx, self.unk_token)  # Default to UNK token
    
    @with_logging(level=9)
    def map_sentence(self, processed_sentence: List[List[str]]) -> List[List[int]]:
        """
        Maps a processed sentence from tokens to their corresponding IDs using the vocabulary.

        Args:
            processed_sentence (List[List[str]]): The preprocessed sentence as a list of token lists.

        Returns:
            List[List[int]]: The sentence with tokens replaced by their corresponding IDs.
        """
        return [
            [self.get_id(token) for token in token_list]
            for token_list in processed_sentence
        ]
    
    
    def save(self, filepath: str):
        """
        Save the vocabulary to a JSON file.
        
        Args:
            filepath (str): Path to the JSON file.
        """
        with open(filepath, 'w') as f:
            json.dump(self.token_to_id, f, indent=4)
    
    def load(self, filepath: str):
        """
        Load the vocabulary from a JSON file.
        
        Args:
            filepath (str): Path to the JSON file.
        """
        with open(filepath, 'r') as f:
            self.token_to_id = json.load(f)
        self.id_to_token = {int(idx): token for token, idx in self.token_to_id.items()}
    
    def build_from_constructor(self, constructor: 'VocabConstructor', data: List[str]) -> None:
        """
        Build the vocabulary using a specified constructor.
        
        Args:
            constructor (VocabConstructor): An instance of a vocabulary constructor.
            data (List[str]): List of raw sequences.
        """
        constructor.build_vocab(data, self)

class KmerVocabConstructor(VocabConstructor):
    """
    Vocabulary constructor for k-mer tokenization.
    Generates all possible k-mers based on the provided k and alphabet.
    """
    def __init__(self, k: int, alphabet: List[str]):
        """
        Initialize the k-mer constructor.
        
        Args:
            k (int): Length of each k-mer.
            alphabet (List[str]): List of characters to construct k-mers.
        """
        self.k = k
        self.alphabet = alphabet
    
    def build_vocab(self, data: List[str], vocab: 'Vocabulary') -> None:
        """
        Build the vocabulary by generating all possible k-mers from the alphabet.
        Ignores the input data as the vocabulary is exhaustive.
        
        Args:
            data (List[str]): List of raw sequences. (Ignored)
            vocab (Vocabulary): Vocabulary instance to populate.
        """
        # Generate all possible k-mers using Cartesian product
        all_kmers = [''.join(p) for p in product(self.alphabet, repeat=self.k)]
        
        # Add all k-mers to the Vocabulary
        for kmer in sorted(all_kmers):
            vocab.add_token(kmer)
