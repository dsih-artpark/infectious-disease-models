# SIR Model

The **SIR model** is one of the simplest compartmental models in epidemiology.  
It divides the population into three compartments:  

- **S (Susceptible):** Individuals who are healthy but can contract the disease.  
- **I (Infected):** Individuals currently infected and able to spread the disease.  
- **R (Recovered):** Individuals who have recovered and gained immunity. 

---

## Transition Diagram

```mermaid
flowchart LR
    S -->|"β * S * I / N"| I
    I -->|"γ * I"| R
    %% Inflow from the population
    dummy([ ]) -->|"μ * N"| S
    %% Natural deaths removed from compartments
    S -->|"μ * S"| 
    I -->|"μ * I"| 
    R -->|"μ * R"| 
```

---

## SIR Model Configuration

Below is an example configuration for the **SIR model** in YAML format.

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