import os
import logging
import inspect
from typing import Any, Callable
from logging.handlers import RotatingFileHandler
from datetime import datetime

LOG_LEVELS = {
    "DL": 8,
    "DM": 9,
    "DH": 10,
    "I": logging.INFO,
    "W": logging.WARNING,
    "E": logging.ERROR,
    "C": logging.CRITICAL,
}

def with_logging(level: int) -> Callable[..., Any]:
    """Decorator to log the start and end of a function, including its module and class."""
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            logger = logging.getLogger("system_logger")

            # Identify the module and function name
            module_name = func.__module__.upper()
            qual_name = func.__qualname__

            # Get function signature and bind arguments
            signature = inspect.signature(func)
            bound_arguments = signature.bind(*args, **kwargs)
            bound_arguments.apply_defaults()
            
            # Filter out 'self' and other non-informative arguments
            parameters = ", ".join(
                f"{k}={v!r}" for k, v in bound_arguments.arguments.items() if k != 'self'
            )

            padding_width = 60 + (10-level)*4
            # Call the function and log the result
            log_message = f"[{module_name}] '{qual_name}'".ljust(padding_width) + f"input: {parameters}"
            logger.log(level, log_message)

            # Call the function and log the result
            result = func(*args, **kwargs)
            if result is not None:
                result_message = f"[{module_name}] '{qual_name}'".ljust(padding_width) + f"output: {result!r}"
                logger.log(level, result_message)

            return result
        return wrapper
    return decorator

def setup_logging(system_level, training_level):

    DEBUG_LOW = 8
    DEBUG_MEDIUM = 9
    DEBUG_HIGH = 10
    logging.addLevelName(logging.CRITICAL, "CRITICAL !!!")
    logging.addLevelName(logging.ERROR,    "ERROR    !!")
    logging.addLevelName(logging.WARNING,  "WARNING  !--")
    logging.addLevelName(logging.INFO,     "INFO     ---")
    logging.addLevelName(DEBUG_LOW,        "DEBUG    --*")
    logging.addLevelName(DEBUG_MEDIUM,     "DEBUG    -*-")
    logging.addLevelName(DEBUG_HIGH,       "DEBUG    *--")

    # Create the logs directory if it doesn't exist
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    # Timestamp for file naming
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    
    # Set up the system logger
    system_logger = logging.getLogger("system_logger")
    system_logger.setLevel(system_level)
    system_log_filename = os.path.join(log_dir, f"{timestamp}_system.log")
    system_handler = RotatingFileHandler(system_log_filename, maxBytes=5 * 1024 * 1024, backupCount=3)
    system_formatter = logging.Formatter("%(asctime)s - %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    system_handler.setFormatter(system_formatter)
    system_logger.addHandler(system_handler)
    
    # Set up the training logger
    training_logger = logging.getLogger("training_logger")
    training_logger.setLevel(training_level)
    training_log_filename = os.path.join(log_dir, f"{timestamp}_training.log")
    training_handler = RotatingFileHandler(training_log_filename, maxBytes=5 * 1024 * 1024, backupCount=3)
    training_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    training_handler.setFormatter(training_formatter)
    training_logger.addHandler(training_handler)

    # Returning both loggers for usage in main
    return system_logger, training_logger