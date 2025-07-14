import os
import matplotlib.pyplot as plt
import numpy as np

def plot_results(time_points, compartments, true_data, noisy_data, subset_t, subset_infected, fitted_results, model_name, plot_dir, true_params, param_names):
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

    # Plot 4: Visualize Estimated vs True Parameters
    def plot_parameter_estimates(true_params, fitted_results, mcmc_sampler=None, param_names=None):
        methods = list(fitted_results.keys())
        n_params = len(param_names)
        fig, axes = plt.subplots(nrows=1, ncols=n_params, figsize=(5 * n_params, 5))

        for i, pname in enumerate(param_names):
            ax = axes[i]
            true_val = true_params[pname]

            # Plot point estimates from optimization
            estimates = [fitted_results[m]['params'][pname] for m in methods]
            ax.bar(methods, estimates, alpha=0.6, label='Estimated')

            # MCMC posterior mean
            if mcmc_sampler is not None:
                mcmc_estimates = mcmc_sampler.get_chain(discard=1000, flat=True)
                mcmc_mean = np.mean(mcmc_estimates[:, i])
                ax.bar('MCMC', mcmc_mean, color='orange', label='MCMC Mean')

            ax.axhline(true_val, color='red', linestyle='--', label='True')
            ax.set_title(f"Parameter: {pname}")
            ax.set_ylabel('Value')
            ax.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "parameter_estimation.png"))
        plt.show()
        plt.close()
    plot_parameter_estimates(true_params, fitted_results, param_names=param_names)