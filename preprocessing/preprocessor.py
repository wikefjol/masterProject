from typing import Protocol
from logging_utils import with_logging
from vocab import Vocabulary
class Strategy(Protocol):
    '''Augments sequence by imitating sequencing errors'''
    def execute(self, sequence: list[str]) -> list[str]:
        """
        Parameters
        ----------
        sequence : str
            DNA sequence

        Returns
        ----------
        sequence : str
            Augmented DNA sequence
        """

# preprocessor.py
from typing import List, Any
from logging_utils import with_logging

class Preprocessor:
    def __init__(
        self,
        augmentation_strategy: Strategy,
        tokenization_strategy: Strategy,
        padding_strategy: Strategy,
        truncation_strategy: Strategy,
        optimal_sentence_length: int = None,
        vocab: Vocabulary = None
    ):
        self.augmentation_strategy = augmentation_strategy
        self.tokenization_strategy = tokenization_strategy
        self.padding_strategy = padding_strategy
        self.truncation_strategy = truncation_strategy
        self.optimal_sentence_length = optimal_sentence_length
        self.vocab = vocab

    @with_logging(level=8)
    def process(self, sequence: str) -> List[List[str]]:
        sequence = list(sequence)  # Convert string to list of characters
        
        augmented_sequence: List[str] = self.augmentation_strategy.execute(sequence)
        tokenized_sentence: List[List[str]] = self.tokenization_strategy.execute(augmented_sequence) 
        padded_sentence: List[List[str]] = self.padding_strategy.execute(tokenized_sentence)
        processed_sentence: List[List[str]] = self.truncation_strategy.execute(padded_sentence)
        mapped_sentence: List[List[int]] = self.vocab.map_sentence(processed_sentence)

        return mapped_sentence