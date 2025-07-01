import os
from pathlib import Path
import yaml
import logging


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../../config.yaml")
cfg = load_config(CONFIG_PATH)


def setup_logging(log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    """Set up basic logging configuration."""
    logger = logging.getLogger("epimodels")
    logger.setLevel(level)

    # Avoid adding multiple handlers
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(console_handler)

        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            logger.addHandler(file_handler)

    return logger
