# Calibration Module

The `calibration.py` module provides tools for fitting model parameters to data using both **optimization methods** and **Bayesian inference (MCMC)**.  
This is essential when comparing simulated epidemic trajectories with noisy or real-world data.

---

## Overview

Calibration enables:

- Parameter estimation from data.
- Comparison of different optimization strategies (Nelder-Mead, BFGS, L-BFGS-B).
- Bayesian inference via **MCMC sampling** (`emcee`) for posterior distributions.
- Integration with the `main.py` workflow for end-to-end simulation and fitting.

---

## Key Components

### `loss_function(theta, model, sampled_times, observed_data, fit_compartments)`
Computes the loss between simulated and observed data for specified compartments.

**Parameters**
- `theta` (`list[float]`) — Candidate parameter vector.
- `model` (`CompartmentalModel`) — Model instance to simulate.
- `sampled_times` (`list[float]`) — Time points to evaluate the simulation.
- `observed_data` (`np.ndarray`) — Observed noisy or real data.
- `fit_compartments` (`list[str]`) — Compartments included in the loss calculation.

**Returns**
- A scalar loss value (lower is better).

---

### `fit_with_optimizers(model, optimizers, ...)`
Fits model parameters using classical optimization algorithms.

**Supported optimizers**
- Nelder-Mead
- BFGS
- L-BFGS-B

**Returns**
- Dictionary of fitted parameters and loss values for each optimizer.

---

### `run_mcmc(model, ...)`
Runs Bayesian calibration with the **emcee** sampler.

**Workflow**
1. Define prior bounds for parameters.
2. Initialize walkers inside the prior region.
3. Simulate the model for each walker position.
4. Compute log-likelihood vs. observed data.
5. Generate posterior samples for uncertainty quantification.

**Returns**
- `sampler` object containing posterior chains.

---

## Usage in `main.py`

Calibration is optional and controlled via CLI flags:

```bash
python main.py --model SIR_model --calibrate --compartment I
```