import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os
from config import (
    MODEL, T, POPULATION, NOISE_STD, SUBSET_RATIO, OPTIMIZERS,
    PARAMS, COMPARTMENTS, TRANSITIONS, INIT_CONDITIONS
)
from model import CompartmentalModel

PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

def simulate(model, y0, t_range):
    return odeint(model.ode_rhs, y0, t_range)

def add_noise(data, std):
    return data + np.random.normal(0, std, size=data.shape)

def subsample(t, data, ratio):
    n = int(ratio * len(t))
    indices = sorted(np.random.choice(len(t), n, replace=False))
    return t[indices], data[indices], indices

def fit_model(model, t_sub, observed, initial_params, param_names, y0):
    def loss(p):
        for i, name in enumerate(param_names):
            model.parameters[name] = p[i]
        sim = simulate(model, y0, t_sub)
        return np.sum((sim[:, 1] - observed)**2)  # I compartment

    results = {}
    t_full = np.linspace(0, T, T)
    for opt in OPTIMIZERS:
        res = minimize(loss, initial_params, method=opt)
        # Update parameters
        for i, name in enumerate(param_names):
            model.parameters[name] = res.x[i]
        # Simulate using full time range for plotting
        fitted_sim = simulate(model, y0, t_full)
        results[opt] = {
            "loss": res.fun,
            "params": dict(zip(param_names, res.x)),
            "simulated": fitted_sim,
        }
    return results

def plot_simulation(t, clean_data):
    plt.figure(figsize=(10, 6))
    for i, comp in enumerate(COMPARTMENTS):
        plt.plot(t, clean_data[:, i], label=f"{comp}")
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title("Simulated (True) Trajectories")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "plot_simulation.png"))
    plt.close()

def plot_noisy(t, noisy_data):
    plt.figure(figsize=(10, 6))
    for i, comp in enumerate(COMPARTMENTS):
        plt.plot(t, noisy_data[:, i], label=f"Noisy {comp}")
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title("Noisy Observations")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "plot_noisy.png"))
    plt.close()
    
def plot_comparison(t_full, true_I, noisy_I, subsampled_t, subsampled_I, results):
    plt.figure(figsize=(10, 6))
    plt.plot(t_full, true_I, label='True I', lw=2)
    plt.scatter(subsampled_t, subsampled_I, color='red', label='Observed I (Subset)', zorder=5)
    for opt, res in results.items():
        plt.plot(t_full, res['simulated'][:, 1], '--', label=f'Fitted I ({opt})')
    plt.xlabel('Time')
    plt.ylabel('Infected')
    plt.legend()
    plt.grid(True)
    plt.title("Fitting Comparison (I Compartment)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "plot_comparison.png"))
    plt.close()


def main():
    t = np.linspace(0, T, T)
    model = CompartmentalModel(COMPARTMENTS, PARAMS, TRANSITIONS)

    # Use initial conditions from first patch
    patch_key = list(INIT_CONDITIONS.keys())[0]
    y0_dict = INIT_CONDITIONS[patch_key]
    y0 = [y0_dict[c] for c in COMPARTMENTS]

    clean_sol = simulate(model, y0, t)
    noisy_data = add_noise(clean_sol, NOISE_STD)
    noisy_I = noisy_data[:, 1]

    # Subsample for fitting
    t_sub, I_sub, idx = subsample(t, noisy_I, SUBSET_RATIO)

    # Fit
    initial_guess = [model.parameters[k] for k in model.parameters]
    param_names = list(model.parameters.keys())
    fit_results = fit_model(model, t_sub, I_sub, initial_guess, param_names, y0)

    # Print fitted info
    for opt, res in fit_results.items():
        print(f"\nOptimizer: {opt}")
        print(f"Loss: {res['loss']:.4f}")
        print("Estimated Parameters:")
        for k, v in res["params"].items():
            print(f"  {k}: {v:.4f}")

    # Plotting
    plot_simulation(t, clean_sol)
    plot_noisy(t, noisy_data)
    plot_comparison(t, clean_sol[:, 1], noisy_I, t_sub, I_sub, fit_results)

if __name__ == "__main__":
    main()