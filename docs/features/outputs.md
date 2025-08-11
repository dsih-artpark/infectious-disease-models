# Outputs

The framework produces:

## Data files (CSV)
Located in `data/<model_name>/`:

- `true_data.csv` — clean simulation trajectories
- `noisy_data.csv` — noisy observed trajectories
- `time_points.csv` — sampled observation points used for fitting

## Plots (PNG)
Located in `plots/<model_name>/`:

- `plot_simulation.png` — clean trajectories of all compartments
- `plot_noisy.png` — noisy simulated data
- `plot_comparison.png` — fitted vs. true trajectories
- `parameter_estimation.png` — fitted parameter summary
- `mcmc_corner_plot.png` — posterior distributions (if MCMC is run)

## Recommended directory layout


```
plots/
  └─ SIR_model/
      ├─ plot_simulation.png
      ├─ plot_noisy.png
      ├─ plot_comparison.png
      ├─ parameter_estimation.png
      └─ mcmc_corner_plot.png

data/
  └─ SIR_model/
      ├─ true_data.csv
      ├─ noisy_data.csv
      └─ time_points.csv
```