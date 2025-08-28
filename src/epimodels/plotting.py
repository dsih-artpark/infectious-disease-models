import os
import matplotlib.pyplot as plt
import corner
import numpy as np


def plot_simulation_only(time_points, compartments, true_data, plot_dir):
    """Always plot the baseline true simulation."""
    os.makedirs(plot_dir, exist_ok=True)

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
    plt.close()


def plot_calibration_results(time_points, compartments, true_data, noisy_data,
                             subset_t, subset_infected, fitted_results,
                             model_name, plot_dir, true_params, param_names,
                             mcmc_sampler=None):
    """Plot calibration-related outputs, only if calibration is run."""
    # Plot noisy data
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
    plt.close()

    # Plot comparison for chosen compartment (default: I)
    if "I" in compartments:
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
        plt.close()

    # Plot parameter estimation bars
    if true_params is not None and param_names is not None:
        methods = list(fitted_results.keys())
        n_params = len(param_names)
        fig, axes = plt.subplots(nrows=1, ncols=n_params, figsize=(5 * n_params, 5), squeeze=False)

        for idx, pname in enumerate(param_names):
            ax = axes[0][idx]
            true_val = true_params[pname]

            estimates = [fitted_results[m]['params'][pname] for m in methods]
            ax.bar(methods, estimates, alpha=0.6, label='Estimated')

            if mcmc_sampler is not None:
                chain = mcmc_sampler.get_chain(discard=100, flat=True)
                valid_values = chain[:, idx][np.isfinite(chain[:, idx])]
                if valid_values.size > 0:
                    mcmc_mean = np.mean(valid_values)
                    ax.bar("MCMC Mean", mcmc_mean, color='orange', label='MCMC Mean')

            ax.axhline(true_val, color='red', linestyle='--', label='True')
            ax.set_title(f"Parameter: {pname}")
            ax.set_ylabel('Value')
            ax.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "parameter_estimation.png"))
        plt.close()

    # Corner plot
    if mcmc_sampler is not None:
        try:
            samples = mcmc_sampler.get_chain(discard=100, flat=True)
            if samples.size > 0 and samples.shape[0] > samples.shape[1]:
                fig = corner.corner(samples, labels=param_names)
                save_path = os.path.join(plot_dir, "mcmc_corner_plot.png")
                fig.savefig(save_path, dpi=300)
                plt.close(fig)
        except Exception as e:
            print(f"[Error] Failed to generate MCMC plot: {e}")
