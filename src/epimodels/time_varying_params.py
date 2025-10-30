import os
import numpy as np
import pandas as pd


def generate_param_series(model_cfg, output_dir="data/generated_params", time_key="simulation_time"):
    """
    Generate time-varying parameter series for the model.

    Reads both:
      1. Base values from model_cfg["parameters"]
      2. Optional time-varying definitions from model_cfg["time_varying"]

    Supported time variation types:
      - "sinusoidal": periodic variation
      - "rectangular": square wave variation
      - "exponential_decay": exponential drop
      - "schedule": piecewise constant values defined by 't' breakpoints
      - callable: user-supplied function f(t)
      - None: keeps constant

    Example YAML:

    time_varying:
      beta:
        type: sinusoidal
        amplitude: 0.2
        period: 30
      gamma:
        type: exponential_decay
        rate: 50
      mu:
        schedule:
          - {t: 5, value: 0.1}
          - {t: 10, value: 0.05}
    """
    os.makedirs(output_dir, exist_ok=True)
    T = model_cfg.get(time_key, 100)
    time = np.linspace(0, T, T + 1)

    param_series = {}
    params = model_cfg.get("parameters", {})
    time_varying = model_cfg.get("time_varying", {})

    for param_name, base_value in params.items():
        tv_spec = time_varying.get(param_name, None)

        if tv_spec is None:
            # constant
            series = np.full_like(time, base_value, dtype=float)

        elif isinstance(tv_spec, dict):
            tv_type = tv_spec.get("type", None)

            if tv_type == "sinusoidal":
                amp = tv_spec.get("amplitude", 0.1)
                period = tv_spec.get("period", 30)
                series = base_value * (1 + amp * np.sin(2 * np.pi * time / period))

            elif tv_type == "rectangular":
                amp = tv_spec.get("amplitude", 0.2)
                period = tv_spec.get("period", 20)
                wave = np.sign(np.sin(2 * np.pi * time / period))
                series = base_value * (1 + amp * wave)

            elif tv_type == "exponential_decay":
                rate = tv_spec.get("rate", 50)
                series = base_value * (1 + 0.1 * np.exp(-time / rate))

            elif "schedule" in tv_spec:
                schedule = sorted(tv_spec["schedule"], key=lambda e: e["t"])
                series = np.zeros_like(time)
                current_val = tv_spec.get("default", base_value)
                for i, t in enumerate(time):
                    for entry in schedule:
                        if t >= entry["t"]:
                            current_val = entry["value"]
                    series[i] = current_val

            else:
                # unknown dict structure, fallback to constant
                series = np.full_like(time, base_value, dtype=float)

        elif callable(tv_spec):
            # user-defined Python function
            series = np.array([tv_spec(t) for t in time])

        else:
            # unknown type, fallback to constant
            series = np.full_like(time, base_value, dtype=float)

        # Save each series
        df = pd.DataFrame({"time": time, param_name: series})
        df.to_csv(os.path.join(output_dir, f"{param_name}_series.csv"), index=False)
        param_series[param_name] = series

    # Save combined
    combined = pd.DataFrame({"time": time, **param_series})
    combined.to_csv(os.path.join(output_dir, "all_params_series.csv"), index=False)

    return param_series


def make_extras_fn_from_series(param_series):
    """
    Build a callable extras_fn(t, y) that returns parameter values
    at time `t` from precomputed param_series arrays.
    """
    time_points = np.arange(len(next(iter(param_series.values()))))

    def extras_fn(t, y):
        idx = int(np.clip(round(t), 0, len(time_points) - 1))
        return {param: values[idx] for param, values in param_series.items()}

    return extras_fn
