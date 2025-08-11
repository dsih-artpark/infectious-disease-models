# Issues Encountered

### 1. Numerical instability / NaNs
- ODE solver can produce NaNs for extreme parameter values.
- Fixes: check parameter bounds, improve initialization, catch exceptions and return large loss.

### 2. Stiff dynamics
- Some parameter regimes are stiff (very small or large rates).
- Fixes: use a stiff solver (e.g., `scipy.integrate.solve_ivp` with `method='BDF'`), tighten tolerances.

### 3. Overfitting to noise
- Deterministic optimizers can pick parameters that explain a particular noise realization.
- Fixes: regularize loss, fit across multiple noise realizations, or use Bayesian inference.

### 4. MCMC walker initialization problems
- Walkers outside priors cause immediate failure.
- Fixes: sample initial walker positions from a distribution strictly inside priors.