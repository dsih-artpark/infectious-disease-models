# time_varying_params.py
import numpy as np
import pandas as pd
import os

def generate_param_series(model_cfg, output_dir="data/time_series"):
    """
    Generates time series for time-varying parameters defined in config.
    Saves them as CSVs and returns a dictionary of arrays indexed by time.
    """
    os.makedirs(output_dir, exist_ok=True)
    TIME = model_cfg["simulation_time"]
    t_values = np.arange(TIME + 1)
    param_series = {}

    if "time_varying" not in model_cfg:
        return None

    for param, spec in model_cfg["time_varying"].items():
        values = np.zeros_like(t_values, dtype=float)

        # Option 1: schedule-based (piecewise constant)
        if isinstance(spec, dict) and "schedule" in spec:
            last_value = spec.get("default", model_cfg["parameters"].get(param, 0))
            for i, t in enumerate(t_values):
                for entry in sorted(spec["schedule"], key=lambda e: e["t"]):
                    if t >= entry["t"]:
                        last_value = entry["value"]
                values[i] = last_value

        # Option 2: formula or callable (e.g., sinusoidal beta(t))
        elif callable(spec):
            values = np.array([spec(t) for t in t_values])

        # Option 3: file-based input (if 'file' key is given)
        elif isinstance(spec, dict) and "file" in spec:
            path = spec["file"]
            df = pd.read_csv(path)
            values = np.interp(t_values, df["time"], df[param])

        # Save to CSV
        csv_path = os.path.join(output_dir, f"{param}_series.csv")
        pd.DataFrame({"time": t_values, param: values}).to_csv(csv_path, index=False)
        param_series[param] = values

    return param_series


def make_extras_fn_from_series(param_series):
    """Builds an extras_fn(t, y) that returns parameter values from precomputed arrays."""
    if param_series is None:
        return None

    def extras_fn(t, y):
        t_idx = int(np.clip(round(t), 0, len(next(iter(param_series.values()))) - 1))
        return {param: values[t_idx] for param, values in param_series.items()}

    return extras_fn
