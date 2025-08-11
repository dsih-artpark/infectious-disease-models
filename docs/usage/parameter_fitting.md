# Parameter Fitting

The framework supports deterministic optimizers and MCMC.

## Loss function

Default: mean squared error across chosen compartments and sampled time points.

\[
L(\theta) = \frac{1}{n}\sum_{i=1}^n \sum_{c\in C_{\text{fit}}} \big( y^{\text{sim}}_c(t_i;\theta) - y^{\text{obs}}_c(t_i) \big)^2
\]

`C_fit` defaults to `['I']` but can be set via `fit_compartments` in YAML.

## Deterministic optimizers

Uses `scipy.optimize.minimize` with method:
- `"Nelder-Mead"` (derivative-free)
- `"BFGS"` (quasi-Newton)
- `"L-BFGS-B"` (bounded, limited-memory)

Example usage:

```python
from scipy.optimize import minimize

def loss_fn(theta_vec):
    # map theta_vec to parameters, simulate, compute MSE on sampled times
    return mse

res = minimize(loss_fn, x0=[0.2,0.1], method='L-BFGS-B', bounds=[(1e-6,5),(1e-6,5)])
```

## Basin-hopping

Use `scipy.optimize.basinhopping` to escape local minima; internally runs a local optimizer.

## MCMC (emcee)

Set uniform priors (or others) and Gaussian likelihood assuming known noise std:

```python
import emcee
def log_prior(theta):
    beta, gamma = theta
    if 0 < beta < 5 and 0 < gamma < 5:
        return 0.0
    return -np.inf

def log_likelihood(theta):
    # simulate, compute gaussian log-likelihood with known sigma
    return ll

def log_posterior(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta)

ndim, nwalkers = 2, 32
p0 = 1e-3 + np.random.rand(nwalkers, ndim) * 0.1
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior)
sampler.run_mcmc(p0, 5000, progress=True)
```

**Tips:**
- Ensure walkers are initialized inside prior bounds.
- Filter out samples where simulation failed (NaN solutions).
- Relax overly narrow priors during debugging.