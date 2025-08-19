# Developer Guide

This guide is for **developers** and **contributors** who want to understand the internal architecture of the Generalized Compartmental Modeling Framework.

The **User-facing pages** explain how to run models and interpret results, but this section focuses on the **code internals** — how modules, classes, and functions work together.

---

## Purpose

The Developer Guide will help you:
- Understand the structure and flow of the codebase
- Extend the framework with new models, fitting methods, or visualizations
- Maintain and debug existing functionality

---

## High-Level Architecture

The framework follows a modular design:

1. **Main Controller (`main.py`)**  
    - Parses CLI arguments (`--model <model_name>`)  
    - Loads the YAML configuration  
    - Orchestrates the workflow: simulation → noise injection → sampling → calibration → plotting → saving outputs  

2. **Core Model Logic (`model.py`)**  
    - `Population` — Population size & assumptions  
    - `CompartmentalModel` — Builds and simulates the ODE system  

3. **Calibration Layer (`calibration.py`)**  
    - Optimization (Nelder-Mead, BFGS, L-BFGS-B, etc.)  
    - MCMC for Bayesian inference  
    - Loss calculation functions  

4. **Plotting Layer (`plotting.py`)**  
    - Generates simulation plots, noisy data plots, comparisons, parameter estimation graphs, and MCMC corner plots  

---

## Workflow for Developers

1. **Add or modify models** in the YAML configuration  
2. **Update model logic** in `model.py` if new compartments or transitions are needed  
3. **Extend calibration methods** in `calibration.py` for new fitting techniques  
4. **Enhance visualization** in `plotting.py` for additional plots or formats  
5. Test your changes with:
    ```bash
    python main.py --model <model_name>
    ```
