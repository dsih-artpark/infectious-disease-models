# YAML Configuration Format

The framework is configured entirely via a YAML file. Each model is defined in a separate block. You can include multiple models in the same `config.yaml`.

## Model Definition

Each model block supports the following fields:

- **`compartments`** (`list[str]`): names of compartments in the model.
- **`parameters`** (`dict[str, float]`): mapping of parameter names to initial values.
- **`transitions`** (`dict[str, str]`): mapping of transitions in the form `"A -> B": "expression"`.  
  - Use parameter names, compartment names, and `N` (population size).  
  - Example: `"S -> I": "beta * S * I / N"`.
- **`population`** (`int`): total population size.
- **`initial_conditions`** (`dict[str, int]`): initial counts for each compartment.
- **`assumptions`** (`str`, optional): description of model assumptions.
- **`simulation_time`** (`int`): total number of time units to simulate.
- **`time_unit`** (`str`): unit for simulation time (e.g. `"days"`, `"years"`).

---

## Plot Settings

Each model may specify plotting options:

- **`y_scale`** (`str`): `"linear"` (default) or `"log"`.  
- **`scale_by_population`** (`bool`): if `true`, all values are normalized by population size.

---

## Calibration Settings

Each model may specify calibration options:

- **`target_data`** (`str`): CSV file with observed data (must be placed in `data/<model_name>/`).  
- **`noise_std`** (`float`): standard deviation of Gaussian noise (used when generating synthetic data).
- **`subset_ratio`** (`float`): fraction of time points to randomly sample for fitting (0.0–1.0).
- **`optimizers`** (`list[str]`): optimizers to run (e.g., `["Nelder-Mead", "BFGS", "L-BFGS-B"]`).
- **`update_config`** (`bool`): if `true`, fitted parameters overwrite config.
- **`parameter_bounds`** (`dict[str, [float, float]]`): bounds for parameters.

---

### Example: SIR model

```yaml
SIR_model:
  compartments: [S, I, R]
  parameters: {beta: 0.25, gamma: 0.15, mu: 0.015}
  transitions: {"S -> I": "beta * S * I / N", "I -> R": "gamma * I", "-> S": "mu * N", "S ->": "mu * S", "I ->": "mu * I", "R ->": "mu * R"}
  population: 1000
  initial_conditions: {S: 990, I: 10, R: 0}
  assumptions: "The population is closed (no births or deaths). The disease is transmitted through direct contact. Immunity is permanent after recovery."
  simulation_time: 160
  time_unit: days
  plot_settings:
    y_scale: linear
    scale_by_population: false
  calibration_settings:
    target_data: noisy_data.csv 
    noise_std: 5.0
    subset_ratio: 0.7
    optimizers: [Nelder-Mead, BFGS, L-BFGS-B]
    update_config: false
    parameter_bounds:
      beta: [0.0, 5.0]
      gamma: [0.0, 2.0]
      mu: [0.0, 1.0]
```

### Syntax rules

- Use Python-style arithmetic in transition expressions (e.g. `beta * S * I / N`).
- Division by `N` is explicit; include `N` in expressions when needed.
- Parameter names and compartment names are case-sensitive.
- You may add additional model blocks in the same file for batch simulations.