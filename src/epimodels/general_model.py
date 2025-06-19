import numpy as np
import matplotlib.pyplot as plt
from model import EpidemicModel
from config import MODEL, PARAMS, POPULATION, T, NOISE_STD, SUBSET_RATIO, OPTIMIZERS
from scipy.optimize import minimize
from model import EpidemicModel

import os
os.makedirs("outputs", exist_ok=True)

def add_noise(data, std=5.0):
    return data + np.random.normal(0, std, data.shape)

def get_subset(data, ratio):
    n = int(len(data) * ratio)
    return data[:n]

def loss_fn(params, model_name, initial_state, t, true_data, N):
    model = EpidemicModel(model_name, params, initial_state, N, len(t))
    sim_data = model.simulate()
    return np.mean((sim_data[:len(true_data)] - true_data) ** 2)

def plot_simulation(true_data, model_name, save_path="outputs/plot_simulation.png"):
    labels = ["S", "I", "R"] if true_data.shape[1] == 3 else ["S", "E", "I", "R"]
    plt.figure()
    for i in range(true_data.shape[1]):
        plt.plot(true_data[:, i], label=f"{labels[i]}")
    plt.legend()
    plt.title(f"{model_name} Simulation")
    plt.savefig(save_path)
    plt.close()

def plot_noisy(noisy_data, model_name, save_path="outputs/plot_noisy.png"):
    labels = ["S", "I", "R"] if noisy_data.shape[1] == 3 else ["S", "E", "I", "R"]
    plt.figure()
    for i in range(noisy_data.shape[1]):
        plt.plot(noisy_data[:, i], label=f"Noisy {labels[i]}")
    plt.legend()
    plt.title(f"{model_name} Noisy Data")
    plt.savefig(save_path)
    plt.close()

def plot_comparison(true_data, noisy_subset, fitted_data, model_name, t_full, t_subset, save_path="outputs/plot_comparison.png"):
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(t_full, true_data[:, 1], '--', label="True Infected")
    plt.plot(t_subset, noisy_subset[:, 1], 'o', label="Observed Infected", alpha=0.6)
    plt.plot(t_full, fitted_data[:, 1], '-', label="Fitted Infected")
    plt.xlabel("Time")
    plt.ylabel("Infected")
    plt.title(f"{model_name}: Infected Curve Fit")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    param_dict = PARAMS[MODEL]
    model = EpidemicModel(MODEL, param_dict["params"], param_dict["y0"], POPULATION, T)
    true_data = model.simulate()
    noisy_data = add_noise(true_data, std=NOISE_STD)
    observed_data = get_subset(noisy_data, SUBSET_RATIO)
    t_full = model.t
    t_subset = t_full[:len(observed_data)]
    results = {}
    best_fit = None
    best_loss = float("inf")

    for optimizer in OPTIMIZERS:
        res = minimize(
            loss_fn,
            param_dict["params"],
            args=(MODEL, param_dict["y0"], model.t, observed_data, POPULATION),
            method=optimizer
        )
        results[optimizer] = res.fun
        if res.fun < best_loss:
            best_loss = res.fun
            best_fit = res.x

    fitted_model = EpidemicModel(MODEL, best_fit, param_dict["y0"], POPULATION, T)
    fitted_data = fitted_model.simulate()

    plot_simulation(true_data, MODEL)
    plot_noisy(noisy_data, MODEL)
    plot_comparison(true_data, observed_data, fitted_data, MODEL, t_full, t_subset)

if __name__ == "__main__":
    main()
