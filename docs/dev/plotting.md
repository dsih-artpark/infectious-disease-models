# Plotting Module

The `plotting.py` module provides utilities for visualizing simulation results, noisy data, parameter estimation, and MCMC posterior distributions.

---

## Functions

### `plot_simulation_only(time_points, compartments, true_data, plot_dir, model_cfg=None, population=None, compartment_choice=None)`
Plots the clean simulation trajectories for all compartments, optionally highlighting a chosen compartment.  
**Parameters**
- `time_points` — Array of time points.
- `compartments` — List of compartment names.
- `true_data` — Simulated trajectories.
- `plot_dir` — Directory to save plots.
- `model_cfg` — Optional model configuration (for scaling/time units).
- `population` — Population size for per-unit scaling.
- `compartment_choice` — Compartment to highlight.

---

### `plot_calibration_results(time_points, compartments, true_data, noisy_data, subset_t, subset_infected, fitted_results, model_name, plot_dir, true_params=None, param_names=None, mcmc_sampler=None, model_cfg=None, population=None, compartment_choice="I")`
Generates comprehensive calibration plots, including:

1. Noisy data for all compartments  
2. Comparison of true vs noisy vs fitted trajectories for calibration compartments  
3. Parameter estimation bar plots for fitted parameters and MCMC mean  
4. MCMC corner plot  

**Parameters**
- `time_points` — Array of simulation time points.
- `compartments` — List of compartment names.
- `true_data` — Clean simulated trajectories.
- `noisy_data` — Noisy observed data.
- `subset_t` — Time points used for fitting.
- `subset_infected` — Observed data for calibration compartments.
- `fitted_results` — Dictionary of fitted results from optimizers.
- `model_name` — Name of the model.
- `plot_dir` — Directory to save plots.
- `true_params` — Optional dictionary of true parameter values.
- `param_names` — List of parameter names for plotting.
- `mcmc_sampler` — Optional emcee sampler object for posterior visualization.
- `model_cfg` — Optional model configuration for scaling/time settings.
- `population` — Population size for per-unit scaling.
- `compartment_choice` — Compartment(s) to highlight in plots.

---

## Example Usage

```python
from plotting import plot_simulation_only, plot_calibration_results

# Plot clean simulation
plot_simulation_only(time_points, compartments, true_data, "plots/SIR_model")

# Plot calibration results
plot_calibration_results(
    time_points,
    compartments,
    true_data,
    noisy_data,
    subset_t,
    subset_infected,
    fitted_results,
    "SIR_model",
    "plots/SIR_model",
    true_params=PARAMS,
    param_names=list(PARAMS.keys()),
    mcmc_sampler=sampler
)
```