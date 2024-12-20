import importlib
import os
import logging
from typing import Any, Protocol
from preprocessing.augmentation import SequenceModifier, BaseStrategy
from preprocessing.preprocessor import Preprocessor
from inspect import signature
from errors import ConstructionError, StrategyError
from logging_utils import with_logging
from vocab import Vocabulary, KmerVocabConstructor

class Modifier(Protocol):
    '''Modifies the list at a specific index'''
    alphabet: list[str]
    def _insert(self, seq: list[str], idx: int) -> None: pass
    def _replace(self, seq: list[str], idx: int) -> None: pass
    def _delete(self, seq: list[str], idx: int) -> None: pass
    def _swap(self, seq: list[str], idx: int) -> None: pass


@with_logging(level=8)
def load_strategy_module(strategy_type: str, strategy_name: str) -> Any:
    """Load and return the strategy module dynamically."""
    full_module_name = f"preprocessing.{strategy_type}"
    module_file_path = os.path.join("preprocessing", f"{strategy_type}.py")

    if not os.path.exists(module_file_path):
        raise ModuleNotFoundError(f"Expected file '{module_file_path}' does not exist.")
    
    try:
        return importlib.import_module(full_module_name)
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(f"Module '{full_module_name}' could not be imported.") from e


@with_logging(level=8)
def prepare_strategy(module: Any, class_name: str, **kwargs) -> Any:
    """Validate and instantiate the strategy class from the module."""
    try:
        strategy_class = getattr(module, class_name)
    except AttributeError as e:
        available_classes = [attr for attr in dir(module) if not attr.startswith("_")]
        raise StrategyError(
            f"Class '{class_name}' not found in module '{module.__name__}'. "
            f"Available classes: {available_classes}"
        ) from e
    except Exception as e:
        raise StrategyError("Something else happend")

    # Identify the exepcted arguments from the strategy signature
    init_signature = signature(strategy_class)
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in init_signature.parameters}
    
    # Identify if the list of provided arguments match the expected arguments
    missing_args = [
        param.name for param in init_signature.parameters.values()
        if param.default == param.empty and param.name not in filtered_kwargs
    ]
    
    if missing_args:
        raise ValueError(f"Missing required arguments for {strategy_class.__name__}: {missing_args}")
    
    return strategy_class(**filtered_kwargs)

@with_logging(level=9)
def get_strategy(strategy_type: str, **kwargs) -> Any:
    """Dynamically load and return an instance of a strategy class based on the type and name."""
    strategy_name = kwargs.pop("strategy", None)
    if not strategy_name:
        raise ValueError(f"Missing 'strategy' in configuration for {strategy_type}.")
        
    class_name = strategy_name.capitalize() + "Strategy"
    try:
        module = load_strategy_module(strategy_type, strategy_name)
        strategy = prepare_strategy(module, class_name, **kwargs)

    
    except (ModuleNotFoundError, StrategyError, ValueError) as e:
        #TODO: eventually replace the belowline with logic to set up a default strategy if one or some arguments are missing. Leave this for now. 
        raise 
    return strategy 

@with_logging(level=logging.INFO)
def create_preprocessor(config: dict[str, Any], vocab: Vocabulary) -> Preprocessor:

    try:
        aug_config = config["augmentation"]
        tok_config = config["tokenization"]
        pad_config = config["padding"]
        trun_config = config["truncation"]
        
        modifier: Modifier = SequenceModifier(aug_config["alphabet"])
        augmentation_strategy = get_strategy("augmentation", modifier = modifier, **aug_config)
        tokenization_strategy = get_strategy("tokenization", **tok_config)
        padding_strategy=get_strategy("padding", **pad_config)
        truncation_strategy=get_strategy("truncation", **trun_config)

    except KeyError as e:
        raise ConstructionError(f"Strategy configuration error: {e}")
    except StrategyError as e:
        raise ConstructionError(f"Error in strategy setup: {e}")
    return Preprocessor(
        augmentation_strategy=augmentation_strategy,
          tokenization_strategy=tokenization_strategy,
          padding_strategy = padding_strategy,
          truncation_strategy = truncation_strategy,
          vocab = vocab
          )

def create_vocabulary(config: dict[str, Any]) -> Vocabulary:
    """
    Create a vocabulary based on the tokenization strategy specified in the configuration.

    Args:
        config (Dict[str, Any]): Configuration dictionary containing tokenization and augmentation settings.

    Returns:
        Vocabulary: An instance of the Vocabulary class populated with tokens as per the chosen strategy.

    Raises:
        ConstructionError: If the tokenization strategy is unsupported or required parameters are missing.
    """
    # Define a mapping from tokenization strategies to their corresponding VocabConstructor classes
    strategy_map = {
        'kmer': KmerVocabConstructor,
        # 'bpe': BPVocabConstructor,      # Future tokenization strategies can be added here
        # 'word': WordVocabConstructor,    # Example placeholders
        # Add other tokenization strategies and their constructors as needed
    }

    # Extract tokenization configuration
    tokenization_config = config.get('tokenization', {})
    strategy = tokenization_config.get('strategy', '').lower()

    if not strategy:
        raise ConstructionError("Tokenization strategy not specified in the configuration.")

    # Retrieve the corresponding constructor class from the strategy map
    constructor_class = strategy_map.get(strategy)

    if not constructor_class:
        raise ConstructionError(
            f"Unsupported tokenization strategy: '{strategy}'. "
            f"Available strategies: {list(strategy_map.keys())}"
        )

    # Extract necessary parameters for the constructor
    try:
        if strategy == 'kmer':
            k = tokenization_config['k']
            # Extract alphabet from augmentation configuration
            augmentation_config = config.get('augmentation', {})
            alphabet = augmentation_config.get('alphabet', ['A', 'C', 'G', 'T'])
            constructor = constructor_class(k=k, alphabet=alphabet)
        else:
            # Handle other strategies by extracting their required parameters
            # Example:
            # if strategy == 'bpe':
            #     merges = tokenization_config['merges']
            #     constructor = constructor_class(merges=merges)
            pass  # Replace with actual parameter extraction for other strategies
    except KeyError as e:
        raise ConstructionError(
            f"Missing required parameter '{e.args[0]}' for tokenization strategy '{strategy}'."
        ) from e

    # Initialize Vocabulary
    vocab = Vocabulary()

    # Build the vocabulary using the selected constructor
    # For strategies like 'kmer' where data is not utilized for exhaustive vocab,
    # pass an empty list or handle accordingly within the constructor
    try:
        if strategy == 'kmer':
            # Since 'kmer' constructor generates exhaustive vocab, data is not needed
            vocab.build_from_constructor(constructor, data=[])
        else:
            # For other strategies that may rely on data, pass the actual data
            # Example:
            # data = config.get('data', [])
            # vocab.build_from_constructor(constructor, data=data)
            pass  # Replace with actual data handling for other strategies
    except Exception as e:
        raise ConstructionError(
            f"Failed to build vocabulary using strategy '{strategy}': {e}"
        ) from e

    # Optionally, save the vocabulary to a file
    # Uncomment and modify the path as needed
    # vocab.save('data/processed/vocab.json')

    return vocab