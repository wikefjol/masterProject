import os
import logging
import inspect
from logging.handlers import RotatingFileHandler
from datetime import datetime
from typing import Callable, Any

LOG_LEVELS = {
    "DL": 8,
    "DM": 9,
    "DH": 10,
    "I": logging.INFO,
    "W": logging.WARNING,
    "E": logging.ERROR,
    "C": logging.CRITICAL,
}

class CustomRotatingFileHandler(RotatingFileHandler):
    """
    A custom rotating file handler to name backups as `filename_<number>.log`.
    """
    def _rotate_filename(self, filename, count):
        """Helper method to construct rotated filenames."""
        base, ext = os.path.splitext(filename)
        return f"{base}_{count}{ext}"

    def doRollover(self):
        """
        Perform a rollover, renaming log files with the custom convention.
        """
        if self.stream:
            self.stream.close()
            self.stream = None

        if self.backupCount > 0:
            for i in range(self.backupCount - 1, 0, -1):
                sfn = self._rotate_filename(self.baseFilename, i)
                dfn = self._rotate_filename(self.baseFilename, i + 1)
                if os.path.exists(sfn):
                    if os.path.exists(dfn):
                        os.remove(dfn)
                    os.rename(sfn, dfn)
            dfn = self._rotate_filename(self.baseFilename, 1)
            if os.path.exists(self.baseFilename):
                if os.path.exists(dfn):
                    os.remove(dfn)
                os.rename(self.baseFilename, dfn)

        if not self.delay:
            self.stream = self._open()

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

def setup_logging(system_level, training_level, log_dir=None):
    """
    Set up system and training loggers with specified levels and directory.
    """
    DEBUG_LOW = 8
    DEBUG_MEDIUM = 9
    DEBUG_HIGH = 10
    logging.addLevelName(DEBUG_LOW, "DEBUG --*")
    logging.addLevelName(DEBUG_MEDIUM, "DEBUG -*-")
    logging.addLevelName(DEBUG_HIGH, "DEBUG *--")

    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    # Clear existing handlers for system_logger
    system_logger = logging.getLogger("system_logger")
    while system_logger.handlers:
        system_logger.handlers.pop()

    # Clear existing handlers for training_logger
    training_logger = logging.getLogger("training_logger")
    while training_logger.handlers:
        training_logger.handlers.pop()

    # Timestamp for log files and content
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_file_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    # Set up system logger
    system_logger.setLevel(system_level)
    system_log_file = os.path.join(log_dir, f"system_{log_file_timestamp}.log") if log_dir else "system.log"
    system_handler = CustomRotatingFileHandler(system_log_file, maxBytes=5 * 1024 * 1024, backupCount=3)
    system_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    system_handler.setFormatter(system_formatter)
    system_logger.addHandler(system_handler)

    # Set up training logger
    training_logger.setLevel(training_level)
    training_log_file = os.path.join(log_dir, f"training_{log_file_timestamp}.log") if log_dir else "training.log"
    training_handler = CustomRotatingFileHandler(training_log_file, maxBytes=5 * 1024 * 1024, backupCount=3)
    training_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    training_handler.setFormatter(training_formatter)
    training_logger.addHandler(training_handler)

    # Log the "Run started" message once
    system_logger.info(f"Run started: {timestamp}")
    training_logger.info(f"Run started: {timestamp}")

    return system_logger, training_logger
