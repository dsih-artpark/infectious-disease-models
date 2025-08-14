# model.py — Core Model Classes

This module defines the primary classes used for simulation: **Population** and **CompartmentalModel**.

---

## Population Class

### Purpose
Represents the population in which the epidemic model is simulated and validates initial conditions.

### Attributes
- `N` (**int**) — Total population size.
- `assumptions` (**str**, optional) — Description of model assumptions.

### Methods
- `__init__(self, N, assumptions=None)`  
  Validates that `N > 0`. Stores population size and optional assumptions.
- `__repr__(self)`  
  Returns a human-readable summary.

---

## CompartmentalModel Class

### Purpose
Encapsulates all model logic: ODE simulation, noise injection, and timepoint sampling.

### Attributes
- `compartments` (**list[str]**) — Names of model compartments.
- `parameters` (**dict**) — Parameter name → value.
- `transitions` (**dict**) — Mapping `FROM->TO` → expression string.
- `population` (**Population**) — Linked Population object.
- `initial_conditions` (**dict**) — Initial values for each compartment.

### Key Methods
- `compute_transition_rates(self, y, params)`  
  Parses transitions and evaluates rates for given state and parameters.
- `compute_rhs(self, t, y, params)`  
  Constructs the system of ODEs (`dC/dt`).
- `simulate(self, y0, t, params)`  
  Integrates the ODE system using `odeint` or `solve_ivp`.
- `add_noise(self, trajectories, sigma)`  
  Adds Gaussian noise to simulation outputs.
- `sample_timepoints(self, trajectories, ratio)`  
  Randomly selects observation points for parameter fitting.

---

### Example Usage
    ```python
    from model import Population, CompartmentalModel

    pop = Population(N=1000)
    sir_model = CompartmentalModel(
        compartments=["S", "I", "R"],
        parameters={"beta": 0.3, "gamma": 0.1},
        transitions={"S->I": "beta * S * I / N", "I->R": "gamma * I"},
        population=pop,
        initial_conditions={"S": 990, "I": 10, "R": 0}
    )

    t, y = sir_model.simulate()
    ```
