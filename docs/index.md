# Generalized Compartmental Modeling Framework

This documentation describes a **modular, configurable framework** for defining, simulating,
and calibrating compartmental epidemiological models (SIR, SEIR, SIS, SIRS, SEIRS, and more).

The framework is driven by a YAML configuration file and supports:
- Arbitrary compartment structures and transition expressions
- Dynamic ODE generation from transition rules
- Simulation with `scipy.integrate.odeint`
- Noise injection and sub-sampling for realistic observations
- Parameter estimation using classical optimizers (Nelder–Mead, BFGS, L-BFGS-B, Basin-Hopping)
- Bayesian calibration via MCMC (emcee)
- Diagnostics, loss landscapes, and posterior analysis

Use the left navigation to explore the YAML specification, usage examples, extensibility notes,
and debugging tips.