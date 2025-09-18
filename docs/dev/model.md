# Model Module

The `model.py` module defines the core classes for epidemic simulation: **Population** and **CompartmentalModel**.  
These classes handle ODE integration, state transitions, noise injection, and sampling of timepoints.

---

## Population Class

### Purpose
Represents the population in which the epidemic model runs and validates initial conditions.

### Attributes
- `N` (**int**) — Total population size.
- `assumptions` (**str**, optional) — Description of model assumptions.

### Methods
- `__init__(self, N, assumptions=None)`  
  Validates that `N > 0`. Stores population size and optional assumptions.
- `__repr__(self)`  
  Returns a human-readable summary of the population.

---

## CompartmentalModel Class

### Purpose
Encapsulates all model logic, including:

- ODE simulation of compartments
- Noise injection for synthetic data
- Sampling of time points for calibration

### Attributes
- `compartments` (**list[str]**) — Names of model compartments.
- `parameters` (**dict**) — Parameter name → value.
- `transitions` (**dict**) — Mapping `FROM->TO` → expression string.
- `population` (**Population**) — Linked population object.
- `initial_conditions` (**dict**) — Initial compartment values.

### Key Methods
- `compute_transition_rates(self, y, params)`  
  Evaluates transition rates for a given state and parameter set.
- `compute_rhs(self, t, y, params)`  
  Constructs the system of ODEs (`dC/dt`) for integration.
- `simulate(self, y0, t, params)`  
  Integrates the ODE system using `odeint` or `solve_ivp`.
- `add_noise(self, trajectories, sigma)`  
  Adds Gaussian noise to simulation outputs.
- `sample_timepoints(self, trajectories, ratio)`  
  Randomly selects observation points for parameter fitting or calibration.

---

## Example Usage

```python
from model import Population, CompartmentalModel

# Define population
pop = Population(N=1000)

# Define SIR model
sir_model = CompartmentalModel(
    compartments=["S", "I", "R"],
    parameters={"beta": 0.3, "gamma": 0.1},
    transitions={"S->I": "beta * S * I / N", "I->R": "gamma * I"},
    population=pop,
    initial_conditions={"S": 990, "I": 10, "R": 0}
)

# Run simulation
t, y = sir_model.simulate(y0={"S": 990, "I": 10, "R": 0}, t=np.linspace(0, 160, 161))
```