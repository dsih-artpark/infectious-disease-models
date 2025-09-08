# Simulation

## From transitions to ODEs

Given compartments `[C1, C2, ..., Ck]` and transitions `A->B: expr`, we compute for each compartment the net rate:

- For `A` subtract `expr`
- For `B` add `expr`

The result is a system:

\[
\frac{d\mathbf{y}}{dt} = f(t, \mathbf{y}; \theta)
\]

where \(\mathbf{y} = [S, I, R, ...]\) and \(\theta\) are the parameters.

## Solver

The default solver is `scipy.integrate.odeint` (LSODA). Usage example:

```python
from scipy.integrate import odeint
import numpy as np

def rhs(y, t, params):
    S, I, R = y
    beta, gamma = params['beta'], params['gamma']
    N = params['N']
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

t = np.linspace(0, time, time+1)
y0 = [990, 10, 0]
sol = odeint(rhs, y0, t, args=( {'beta':0.3,'gamma':0.1,'N':1000}, ))
```

## Time-dependent parameters

If your model requires time-varying parameters, provide an `extras_fn(t, y)` hook that returns a dict of parameter values at `t`. The `rhs` should consult those values when computing rates.