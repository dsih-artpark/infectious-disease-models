# Extensibility

This framework was designed to be extensible. Common extension points:

## 1. New compartment models
Add a new block in the YAML with compartments and transitions — no code changes needed.

## 2. New parameter types
Add parameters to the `parameters` map. For time-varying parameters, implement `extras_fn`.

## 3. Custom loss functions
Provide a Python callable to compute the loss instead of MSE. For example, weighted MSE or Poisson log-likelihood for count data.

## 4. Metapopulation / Network models
- Extend `transitions` syntax to include indices or use multiple population blocks.
- The ODE generator will need to be extended to create vectorized compartments per patch.

## 5. New optimizers or inference engines
- Add wrappers for other optimizers (e.g., CMA-ES, differential evolution).
- Add MCMC engines (e.g., PyMC, NumPyro) by mapping log-posterior.