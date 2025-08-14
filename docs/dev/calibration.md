
# calibration.py — Parameter Fitting and MCMC

This module implements optimization-based and Bayesian parameter estimation.

---

## Functions

### `loss_function(theta, model, sampled_times, observed_data, fit_compartments)`
Computes the loss between simulated and observed data for selected compartments.

**Parameters**
- `theta` (**list[float]**) — Parameter vector.
- `model` (**CompartmentalModel**) — Model to simulate.
- `sampled_times` (**list[float]**) — Time points used for fitting.
- `observed_data` (**np.ndarray**) — Observed noisy data.
- `fit_compartments` (**list[str]**) — Compartments included in the loss.

---

### `fit_with_optimizers(model, optimizers, ...)`
Runs selected optimizers (e.g., Nelder-Mead, BFGS, L-BFGS-B) to fit parameters.

---

### `run_mcmc(model, ...)`
Uses the **emcee** sampler to estimate parameter posterior distributions.

**Workflow**
1. Define prior bounds.
2. Initialize walkers inside bounds.
3. Simulate the model and compute log-likelihood.
4. Save and plot posterior samples.

---

### Example Usage
```python
from calibration import fit_with_optimizers

results = fit_with_optimizers(model, ["BFGS", "Nelder-Mead"], ...)
```