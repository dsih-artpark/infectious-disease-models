# YAML Configuration Format

The framework is configured entirely via a YAML file. Below are the supported fields and an example.

## Global Settings

- `days` (int): number of days to simulate.
- `noise_std` (float): standard deviation of Gaussian noise added to trajectories.
- `subset_ratio` (float): fraction of time points to randomly sample for fitting (0.0 - 1.0).
- `optimizers` (list[str]): which optimizers to run (e.g., Nelder-Mead, BFGS, L-BFGS-B).
- `fit_compartments` (list[str], optional): compartments included in the loss (default: ['I']).

## Model Definition

Each model block has:
- `compartments`: list of compartment names (strings).
- `parameters`: map of parameter names to initial values.
- `transitions`: map of `FROM->TO` to a rate expression (use parameter names and compartment names).
- `population`: integer N (used in expressions).
- `initial_conditions`: map of compartment initial values.
- `assumptions`: (string) optional description.

### Example: SIR model

```yaml
days: 160
noise_std: 2.0
subset_ratio: 0.4
optimizers:
  - Nelder-Mead
  - BFGS
  - L-BFGS-B
fit_compartments: [I]

SIR_model:
  compartments: [S, I, R]
  parameters:
    beta: 0.3
    gamma: 0.1
  transitions:
    S->I: beta * S * I / N
    I->R: gamma * I
  population: 1000
  initial_conditions:
    S: 990
    I: 10
    R: 0
  assumptions: Closed population, permanent immunity
```

### Syntax rules

- Use Python-style arithmetic in transition expressions (e.g. `beta * S * I / N`).
- Division by `N` is explicit; include `N` in expressions when needed.
- Parameter names and compartment names are case-sensitive.
- You may add additional model blocks in the same file for batch simulations.