import os
import numpy as np
import pandas as pd

def generate_param_series(model_cfg, output_dir="data/generated_params", time_key="simulation_time"):
    """
    Generate time-varying parameter series for the model.

    Args:
        model_cfg: dictionary containing model configuration
        output_dir: folder where parameter CSVs are saved
        time_key: key in model_cfg specifying simulation length
    Returns:
        param_series: dict of {param_name: np.ndarray of values}
    """
    os.makedirs(output_dir, exist_ok=True)
    T = model_cfg.get(time_key, 100)
    time = np.linspace(0, T, T + 1)

    param_series = {}
    if "parameters" not in model_cfg:
        raise ValueError("Model config must have 'parameters' defined.")

    for param_name, base_value in model_cfg["parameters"].items():
        # Example: define simple patterns or noise
        if param_name.lower() == "beta":
            series = base_value * (1 + 0.2 * np.sin(2 * np.pi * time / 30))  # periodic variation
        elif param_name.lower() == "gamma":
            series = base_value * (1 + 0.1 * np.exp(-time / 50))  # slow decay
        else:
            series = np.full_like(time, base_value, dtype=float)

        # Save each series
        df = pd.DataFrame({"time": time, param_name: series})
        df.to_csv(os.path.join(output_dir, f"{param_name}_series.csv"), index=False)
        param_series[param_name] = series

    # save combined time series file
    combined = pd.DataFrame({"time": time, **param_series})
    combined.to_csv(os.path.join(output_dir, "all_params_series.csv"), index=False)
    return param_series


def make_extras_fn_from_series(param_series):
    """
    Build a callable extras_fn(t, y) that returns current param values
    from generated series.
    """
    time_points = np.arange(len(next(iter(param_series.values()))))

    def extras_fn(t, y):
        idx = int(np.clip(round(t), 0, len(time_points) - 1))
        return {param: values[idx] for param, values in param_series.items()}

    return extras_fn
