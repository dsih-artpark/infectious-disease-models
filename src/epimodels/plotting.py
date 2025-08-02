import os
import matplotlib.pyplot as plt
import corner 
import numpy as np

def plot_results(time_points, compartments, true_data, noisy_data, subset_t, subset_infected, fitted_results, model_name, plot_dir, true_params, param_names, mcmc_sampler=None):
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
    if true_params is not None and param_names is not None:
        methods = list(fitted_results.keys())
        n_params = len(param_names)
        fig, axes = plt.subplots(nrows=1, ncols=n_params, figsize=(5 * n_params, 5), squeeze=False)

        for idx, pname in enumerate(param_names):
            ax = axes[0][idx]
            true_val = true_params[pname]

            # Plot point estimates from optimization
            estimates = [fitted_results[m]['params'][pname] for m in methods]
            ax.bar(methods, estimates, alpha=0.6, label='Estimated')

            # MCMC posterior mean
            if mcmc_sampler is not None:
                chain = mcmc_sampler.get_chain(discard=100, flat=True)
                valid_values = chain[:, idx][np.isfinite(chain[:, idx])]
                if valid_values.size == 0:
                    print(f"Warning: No valid MCMC samples for parameter {param_names[idx]}")
                    mcmc_mean = np.nan
                else:
                    mcmc_mean = np.mean(valid_values)

                ax.bar("MCMC Mean", mcmc_mean, color='orange', label='MCMC Mean')

            ax.axhline(true_val, color='red', linestyle='--', label='True')
            ax.set_title(f"Parameter: {pname}")
            ax.set_ylabel('Value')
            ax.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "parameter_estimation.png"))
        plt.show()
        plt.close()
    
    # Plot 5: Corner plot from MCMC samples
    if mcmc_sampler is not None:
        try:
            samples = mcmc_sampler.get_chain(discard=100, flat=True)
            print("MCMC samples shape:", samples.shape)
            if samples.size == 0:
                print("[Warning] MCMC samples are empty after discarding burn-in. Skipping corner plot.")
            elif np.any(np.isnan(samples)) or np.any(~np.isfinite(samples)):
                print("[Warning] MCMC samples contain NaN or inf. Skipping corner plot.")
            elif samples.shape[0] <= samples.shape[1]:
                print("[Warning] Not enough MCMC samples to make a corner plot.")
            else:
                # if samples.shape[0] > samples.shape[1]:  # Check: more samples than dimensions
                fig = corner.corner(samples, labels=param_names)
                save_path = os.path.join(plot_dir, "mcmc_corner_plot.png")
                fig.savefig(save_path, dpi=300)
                plt.show()
                plt.close(fig)
        except Exception as e:
            print(f"[Error] Failed to generate MCMC plot: {e}")