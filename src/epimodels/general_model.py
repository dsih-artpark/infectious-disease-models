import numpy as np
import matplotlib.pyplot as plt
from model import EpidemicModel
from config import MODEL, PARAMS, POPULATION, T, NOISE_STD, SUBSET_RATIO, OPTIMIZERS
from scipy.optimize import minimize
import os

os.makedirs("outputs", exist_ok=True)

def add_noise(data, std=5.0):
    return data + np.random.normal(0, std, data.shape)

def get_subset(data, t, ratio, seed=42):
    np.random.seed(seed)
    n = int(len(data) * ratio)
    indices = np.sort(np.random.choice(len(data), n, replace=False))
    return data[indices], t[indices], indices

def loss_fn(params, model_name, initial_state, t_full, true_data, N, indices, fit_all=False, compartment_index=1):
    try:
        model = EpidemicModel(model_name, params, initial_state, N, len(t_full))
        sim_data = model.simulate()

        if np.any(np.isnan(sim_data)) or np.any(np.isinf(sim_data)):
            return np.inf

        if fit_all:
            pred = sim_data[indices, :]
            return np.mean((pred - true_data) ** 2)
        else:
            pred = sim_data[indices, compartment_index]
            return np.mean((pred - true_data) ** 2)

    except Exception as e:
        print(f"Loss function error: {e}")
        return np.inf

def plot_simulation(true_data, model_name, compartment_names=None, save_path="outputs/plot_simulation.png"):
    if compartment_names is None:
        compartment_names = [f"C{i}" for i in range(true_data.shape[1])]
    plt.figure()
    for i in range(true_data.shape[1]):
        plt.plot(true_data[:, i], label=f"{compartment_names[i]}")
    plt.legend()
    plt.title(f"{model_name} Simulation")
    plt.xlabel("Time")
    plt.ylabel("Population")
    plt.savefig(save_path)
    plt.close()



def plot_noisy(noisy_data, model_name, compartment_names=None, save_path="outputs/plot_noisy.png"):
    if compartment_names is None:
        compartment_names = [f"C{i}" for i in range(noisy_data.shape[1])]
    plt.figure()
    for i in range(noisy_data.shape[1]):
        plt.plot(noisy_data[:, i], label=f"Noisy {compartment_names[i]}")
    plt.legend()
    plt.title(f"{model_name} Noisy Data")
    plt.xlabel("Time")
    plt.ylabel("Population")
    plt.savefig(save_path)
    plt.close()


def plot_comparison(true_data, noisy_subset, fitted_data, model_name, t_full, t_subset, compartment_names=None, save_path="outputs/plot_comparison.png"):
    subset_indices = np.searchsorted(t_full, t_subset)  
    plt.figure(figsize=(10, 6))
    plt.plot(t_subset, fitted_data[subset_indices, 1], '-', label="Fitted Infected")
    plt.plot(t_full, true_data[:, 1], '--', label="True Infected")
    plt.plot(t_subset, noisy_subset[:, 1], 'o', label="Observed Infected", alpha=0.6)
    #plt.plot(t_full, fitted_data[:, 1], '-', label="Fitted Infected")
    plt.xlabel("Time")
    plt.ylabel("Infected")
    plt.title(f"{model_name}: Infected Curve Fit")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    param_dict = PARAMS[MODEL]
    initial_params = param_dict["params"]
    y0 = param_dict["y0"]

    # True simulation
    model = EpidemicModel(MODEL, initial_params, y0, POPULATION, T)
    true_data = model.simulate()

    # Add noise
    noisy_data = add_noise(true_data, std=NOISE_STD)

    # Subset of noisy data
    t_full = model.t
    noisy_subset, t_subset, indices = get_subset(noisy_data, t_full, SUBSET_RATIO)

    # Fit only infected
    observed_infected = noisy_subset[:, 1]
    results = {}
    best_fit = None
    best_loss = float("inf")

    for optimizer in OPTIMIZERS:
        print(f"Running optimizer: {optimizer}")
        res = minimize(
            loss_fn,
            initial_params,
            args=(MODEL, y0, t_full, observed_infected, POPULATION, indices),  
            method=optimizer
        )
        results[optimizer] = res.fun
        print(f"{optimizer} loss: {res.fun:.4f}")
        if res.fun < best_loss:
            best_loss = res.fun
            best_fit = res.x
        print(f"Optimizer: {optimizer}")
        print(f"Fitted parameters: {res.x}")
        print(f"Loss: {res.fun:.4f}")

    # Simulate best-fit
    fitted_model = EpidemicModel(MODEL, best_fit, y0, POPULATION, T)
    fitted_data = fitted_model.simulate()

    # Plot
    plot_simulation(true_data, MODEL)
    plot_noisy(noisy_data, MODEL)
    plot_comparison(true_data, noisy_subset, fitted_data, MODEL, t_full, t_subset)  

if __name__ == "__main__":
    main()
