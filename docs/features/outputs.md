# Outputs

The framework produces:
- Clean simulation trajectories (`clean_{model}.csv`)
- Noisy observed trajectories (`noisy_{model}.csv`)
- Sampled observation points used for fitting (`sampled_obs_{model}.csv`)
- Report with fitted parameter estimates for each optimizer (`fit_results.json`)
- Plot images: trajectories, loss landscape, posterior corner plots.

## Recommended directory layout for results

```
results/
  ├─ SIR_model/
  │   ├─ clean_SIR.csv
  │   ├─ noisy_SIR.csv
  │   ├─ sampled_obs_SIR.csv
  │   ├─ fit_results.json
  │   ├─ plots/
  │   │   ├─ trajectories.png
  │   │   ├─ loss_landscape.png
  │   │   └─ corner.png
```