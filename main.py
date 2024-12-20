from factory import create_preprocessor, create_vocabulary
from errors import ConstructionError, PreprocessingError
from logging_utils import setup_logging, with_logging
import logging

import random
import string


def main():
    # Set up loggers. Levels: DEBUGGING: 8,9,10 for levels low, medium and high. INFO: 20, , WARNING: W, ERROR: E, CRITICAL: C
    system_level = 8
    training_level = 10
    system_logger, training_logger = setup_logging(system_level, training_level)

    # mock config file for now
    config = {
        "model": {"name": "example_model"},
        "augmentation": {"strategy": "base", "alphabet": ['A', 'C', 'G', 'T'], "modification_probability": 0.5},
        "tokenization": {"strategy": "kmer", "k":3},
        "padding": {"strategy": "random", "optimal_length":8},
        "truncation": {"strategy": "slidingwindow", "optimal_length":8}
    }

    

    try:
        vocab = create_vocabulary(config)
        preprocessor = create_preprocessor(config, vocab)
        sequence = 'ACGTGCTTCGATC'
        
        # Phase 1:
        mapped_sentence = preprocessor.process(sequence) 


        # Phase 2:

        # Phase 3:

    except PreprocessingError as e:
        system_logger.error(f"Preprocessing error occurred: {e}")
    except ConstructionError as e:
        system_logger.critical(f"Critical configuration error during model construction: {e}")
    except Exception as e:
        system_logger.critical(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()
