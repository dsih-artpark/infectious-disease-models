# Overview

In real-world epidemic modeling, parameters such as the infection rate (β) or recovery rate (γ) rarely stay constant.  
They can change over time due to interventions, behavioral changes, vaccination drives, or environmental factors.

Parameters like the transmission rate `β(t)` or recovery rate `γ(t)` are often nonstationary in practice (interventions, seasonality, behaviour).  
Our framework separates **generation** of a parameter time series (saved to `data/`) from **consumption** of that series by simulators and calibrators.

Workflow summary:

1. Generate param series → `data/<MODEL_NAME>/beta.csv`, `data/<MODEL_NAME>/gamma.csv`  
2. Build an `extras_fn` wrapper to provide parameter values at each simulation time step.  
3. Pass `extras_fn` into `model.simulate(...)` and calibration routines like `Calibrator.fit()` and `Calibrator.run_mcmc()`.  
4. Use the saved CSVs for plotting and record keeping.

---

## Generating parameter series

The generator function produces time series and writes CSV(s) to `data/<MODEL_NAME>/`.

**Example behavior of `time_varying_params.generate_param_series`:**

- Input: `model_cfg` (the YAML for the model) and `output_dir` (e.g. `data/SIR_model`)
- Output: Writes CSV files such as:
  - `data/SIR_model/beta.csv`
  - `data/SIR_model/gamma.csv` (if configured)
- Return value: a dictionary-like `param_series` (in-memory arrays) for immediate use.

**Example CSV format (`data/SIR_model/beta.csv`):**
```csv
time,beta
0,0.30
1,0.30
2,0.30
3,0.30
4,0.30
5,0.10
6,0.10
...