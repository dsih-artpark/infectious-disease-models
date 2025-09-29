import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # ✅ Force non-GUI backend for pytest

from epimodels.plotting import plot_simulation_only, plot_calibration_results

def test_plot_simulation_only(tmp_path):
    t = np.linspace(0, 10, 11)
    compartments = ["S", "I", "R"]
    true_data = np.random.rand(len(t), 3)  # dummy data
    outdir = tmp_path / "plots"
    plot_simulation_only(t, compartments, true_data, str(outdir))
    assert (outdir / "plot_simulation.png").exists()

def test_plot_calibration_results(tmp_path):
    t = np.linspace(0, 10, 11)
    compartments = ["S", "I", "R"]
    true_data = np.random.rand(len(t), 3)
    noisy_data = np.random.rand(len(t), 3)
    subset_t = t[::2]
    subset_infected = noisy_data[::2, 1]
    fitted_results = {
        "Nelder-Mead": {
            "params": {"beta": 0.2, "gamma": 0.1},
            "trajectory": true_data,
            "loss": 0.01,
        }
    }
    outdir = tmp_path / "plots"
    plot_calibration_results(
        t, compartments, true_data, noisy_data,
        subset_t, subset_infected, fitted_results,
        model_name="SIR", plot_dir=str(outdir),
        true_params={"beta": 0.3, "gamma": 0.1},
        param_names=["beta", "gamma"]
    )
    assert (outdir / "plot_comparison.png").exists()
