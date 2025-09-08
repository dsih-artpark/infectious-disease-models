# Workflow

High-level pipeline:

```
Define YAML → Generate ODEs → Simulate → Add Noise → Subset → Fit Parameters → Visualize & Diagnose
```

## Steps

1. **Define YAML**: create a model block with compartments, parameters, transitions, population, and initial conditions.
2. **Generate ODEs**: the framework parses `transitions` into symbolic rates and constructs the right-hand side of the ODE system automatically.
3. **Simulate**: integrate the ODE system (default: `scipy.integrate.odeint`) over the time grid `t = 0..time`.
4. **Add Noise**: Gaussian noise is added to simulated trajectories and clipped to valid ranges.
5. **Subset Sampling**: sample a fraction of time points (`subset_ratio`) to mimic sparse observations.
6. **Fit Parameters**: minimize a loss function (default MSE over specified compartments) using chosen optimizers. Optionally run MCMC for posterior estimation.
7. **Visualize**: produce trajectory plots, loss landscape, and posterior corner plots.

## CLI / Script Example

Run a specific model (defined in your YAML configuration) by name:

```bash
python main.py --model SIR_model
```

After running, you’ll find all outputs in the `plots/<model_name>/` folder.

