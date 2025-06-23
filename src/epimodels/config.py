from pathlib import Path
import logging
import yaml
import os
#CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")

from config_loader import load_config

cfg = load_config("../../config.yaml")

MODEL = cfg["model"]
T = cfg["days"]
POPULATION = cfg["population"]
NOISE_STD = cfg["noise_std"]
SUBSET_RATIO = cfg["subset_ratio"]
OPTIMIZERS = cfg["optimizers"]
PARAMS = cfg["models"]
COMPARTMENTS = cfg["models"][MODEL]["compartments"]

def setup_logging(log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    """Set up basic logging configuration.

    Args:
        log_file: Optional path to log file. If None, logs only to console
        level: Logging level (default: INFO)

    Returns:
        Logger instance
    """
    logger = logging.getLogger("epimodels")
    logger.setLevel(level)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)

    # File handler (if log_file is provided)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)

    return logger
