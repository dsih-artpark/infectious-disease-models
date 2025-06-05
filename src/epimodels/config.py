from pathlib import Path
import logging
import yaml

DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "config.yaml"


def load_config(path: Path | str | None = None) -> dict:
    """Load YAML config from given path or default project root config.yaml."""
    if path is None:
        path = DEFAULT_CONFIG_PATH
    else:
        path = Path(path)
    with open(path, "r") as f:
        return yaml.safe_load(f)


CONFIG = load_config()


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
