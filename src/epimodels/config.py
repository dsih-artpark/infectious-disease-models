import os
from pathlib import Path
import yaml
import logging

# Load config from two levels up
def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

# Construct path to config.yaml
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../../config.yaml")

# Load YAML
cfg = load_config(CONFIG_PATH)

# Top-level simulation settings
MODEL = cfg["model_name"]
T = cfg["days"]
NOISE_STD = cfg["noise_std"]
SUBSET_RATIO = cfg["subset_ratio"]
OPTIMIZERS = cfg["optimizers"]

# Model-specific settings
PARAMS = cfg["parameters"]
COMPARTMENTS = cfg["compartments"]
TRANSITIONS = cfg["transitions"]
INIT_CONDITIONS = cfg["initial_conditions"]

NUM_PATCHES = cfg["num_patches"]
POP_PER_PATCH = cfg["population_per_patch"]
POPULATION = POP_PER_PATCH * NUM_PATCHES
NETWORK_MATRIX = cfg.get("network_matrix", None)



def get_initial_state_dict():
    """Flatten initial conditions for all patches into a dict."""
    state = {}
    for i in range(NUM_PATCHES):
        patch_key = f"patch_{i}"
        for comp in COMPARTMENTS:
            state[f"{comp}_{i}"] = INIT_CONDITIONS[patch_key][comp]
    return state


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
