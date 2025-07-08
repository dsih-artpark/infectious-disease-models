import os
import matplotlib.pyplot as plt
import numpy as np

def plot_results(time_points, compartments, true_data, noisy_data, subset_t, subset_infected, fitted_results, model_name, plot_dir):
    os.makedirs(plot_dir, exist_ok=True)

    # Plot 1: True simulation
    plt.figure(figsize=(10, 6))
    for i, comp in enumerate(compartments):
        plt.plot(time_points, true_data[:, i], label=f"{comp} (true)")
    plt.title("True Simulation of Compartments")
    plt.xlabel("Days")
    plt.ylabel("Population")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "plot_simulation.png"))
    plt.show()
    plt.close()

    # Plot 2: Noisy data
    plt.figure(figsize=(10, 6))
    for i, comp in enumerate(compartments):
        plt.plot(time_points, noisy_data[:, i], linestyle="--", label=f"{comp} (noisy)")
    plt.title("Noisy Data for All Compartments")
    plt.xlabel("Days")
    plt.ylabel("Population")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "plot_noisy.png"))
    plt.show()
    plt.close()

    # Plot 3: Infected - true vs noisy subset vs fitted
    plt.figure(figsize=(10, 6))
    i_index = compartments.index("I")
    plt.plot(time_points, true_data[:, i_index], label="True Infected", linewidth=2)
    plt.scatter(subset_t, subset_infected, label="Noisy Subset", color="black", zorder=5)

    for method, result in fitted_results.items():
        plt.plot(time_points, result['trajectory'][:, i_index], label=f"Fitted ({method})")

    plt.title("Infected Compartment: True vs Noisy Subset vs Fitted")
    plt.xlabel("Days")
    plt.ylabel("Population")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "plot_comparison.png"))
    plt.show()
    plt.close()
