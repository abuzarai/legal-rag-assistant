import logging
import sys

def get_logger(name: str):
    """Return a configured logger with colored console output."""
    logger = logging.getLogger(name)

    if not logger.handlers:  # Prevent duplicate handlers
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="[%(levelname)s] %(message)s",
            datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger
