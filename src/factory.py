import importlib
import os
from inspect import signature
from typing import Any, Protocol


from preprocessing.augmentation import SequenceModifier
from preprocessing.preprocessor import Preprocessor
from errors import ConstructionError, StrategyError
from utils.logging_utils import with_logging
from vocab import Vocabulary, KmerVocabConstructor



class Modifier(Protocol):
    """Protocol defining sequence modification methods."""
    alphabet: list[str]

    def _insert(self, seq: list[str], idx: int) -> None: pass
    def _replace(self, seq: list[str], idx: int) -> None: pass
    def _delete(self, seq: list[str], idx: int) -> None: pass
    def _swap(self, seq: list[str], idx: int) -> None: pass


@with_logging(level=8)
def load_strategy_module(strategy_type: str) -> Any:
    """Dynamically load and return the strategy module."""
    full_module_name = f"preprocessing.{strategy_type}"
    try:
        return importlib.import_module(full_module_name)
    except ModuleNotFoundError:
        raise ModuleNotFoundError(f"Module '{full_module_name}' could not be imported.")


@with_logging(level=8)
def prepare_strategy(module: Any, class_name: str, **kwargs) -> Any:
    """Validate and instantiate the strategy class from the module."""
    try:
        strategy_class = getattr(module, class_name)
    except AttributeError:
        available_classes = [attr for attr in dir(module) if not attr.startswith("_")]
        raise StrategyError(
            f"Class '{class_name}' not found in module '{module.__name__}'. "
            f"Available classes: {available_classes}"
        )

    init_signature = signature(strategy_class)
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in init_signature.parameters}

    missing_args = [
        param.name for param in init_signature.parameters.values()
        if param.default == param.empty and param.name not in filtered_kwargs
    ]
    if missing_args:
        raise ValueError(f"Missing required arguments for {strategy_class.__name__}: {missing_args}")

    return strategy_class(**filtered_kwargs)


@with_logging(level=9)
def get_strategy(strategy_type: str, **kwargs) -> Any:
    """Dynamically load and return an instance of a strategy class."""
    strategy_name = kwargs.pop("strategy", None)
    if not strategy_name:
        raise ValueError(f"Missing 'strategy' in configuration for {strategy_type}.")
    
    class_name = strategy_name.capitalize() + "Strategy"
    module = load_strategy_module(strategy_type)
    return prepare_strategy(module, class_name, **kwargs)

@with_logging(level=10)
def create_preprocessor(config, vocab, augmentation_config=None) -> Preprocessor:
    """Create and return a Preprocessor instance based on the configuration."""
    try:
        tokenization_config = config["tokenization_strategy"]
        padding_config = config["padding_strategy"]
        truncation_config = config["truncation_strategy"]

        # Create the modifier instance
        alphabet = augmentation_config.get("alphabet", ["A", "C", "G", "T"])
        modifier = SequenceModifier(alphabet)

        # Create strategies
        augmentation_strategy = get_strategy("augmentation", modifier=modifier, **augmentation_config)
        tokenization_strategy = get_strategy("tokenization", **tokenization_config)
        padding_strategy = get_strategy("padding", **padding_config)
        truncation_strategy = get_strategy("truncation", **truncation_config)

    except KeyError as e:
        raise ConstructionError(f"Strategy configuration error: {e}")
    except StrategyError as e:
        raise ConstructionError(f"Error in strategy setup: {e}")

    return Preprocessor(
        augmentation_strategy=augmentation_strategy,
        tokenization_strategy=tokenization_strategy,
        padding_strategy=padding_strategy,
        truncation_strategy=truncation_strategy,
        vocab=vocab,
    )


@with_logging(level=10)
def create_vocabulary(config: dict) -> Vocabulary:
    """Create a vocabulary based on the tokenization strategy from the general configuration."""
    tokenization_config = config.get("tokenization_strategy", {})
    strategy = tokenization_config.get("strategy", "").lower()

    if strategy == "kmer":
        k = tokenization_config["k"]
        alphabet = config.get("augmentation_strategy", {}).get("alphabet", ["A", "C", "G", "T"])
        constructor = KmerVocabConstructor(k=k, alphabet=alphabet)
    else:
        raise ConstructionError(f"Unsupported tokenization strategy: '{strategy}'")

    vocab = Vocabulary()
    vocab.build_from_constructor(constructor, data=[])
    return vocab