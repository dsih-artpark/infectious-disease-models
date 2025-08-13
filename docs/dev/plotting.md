
# plotting.py — Visualization Utilities

This module generates plots for simulation results, noisy data, parameter fits, and MCMC results.

---

## Functions

- `plot_simulation(traj, t, out_path)` — Plots clean simulation trajectories.
- `plot_noisy(noisy_traj, sampled_points, out_path)` — Plots noisy observed data.
- `plot_comparison(true_traj, fitted_traj, sampled_points, out_path)` — Compares fitted vs. true trajectories.
- `parameter_estimation_plot(fit_results, out_path)` — Shows estimated parameters from optimizers.
- `mcmc_corner_plot(samples, out_path)` — Corner plot of posterior parameter distributions.

---

## Example Usage
```python
from plotting import plot_simulation

plot_simulation(traj, t, "plots/SIR_model/plot_simulation.png")
