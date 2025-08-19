# Generalized Compartmental Modeling Framework

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Docs](https://img.shields.io/badge/Docs-MkDocs%20Material-blueviolet)](https://dsih-artpark.github.io/infectious-disease-models/)

A flexible Python framework for simulating, fitting, and visualizing epidemiological compartmental models such as **SIR**, **SEIR**, **SIS**, and custom user-defined models.

---

## Features

- Define models entirely via a **YAML configuration file**
- Supports **ODE-based** simulations for arbitrary compartment structures
- Built-in **noise injection** and **subset sampling** for fitting
- Parameter estimation with:
  - **Nelder-Mead**
  - **BFGS**
  - **L-BFGS-B**
  - **MCMC** (via `emcee`)
- Automatic plot generation for:
  - Clean simulations
  - Noisy data
  - Model fit comparisons
  - Parameter estimation results
  - MCMC posterior distributions

---

## Installation

The source code is available on GitHub at:  
https://github.com/dsih-artpark/infectious-disease-models

```bash
git clone https://github.com/dsih-artpark/infectious-disease-models.git
cd infectious-disease-models
uv sync
source .venv/bin/activate
```
---

## Documentation

Full documentation is available here:
[Generalized Compartmental Modeling Framework Docs](https://dsih-artpark.github.io/infectious-disease-models/)