import os
import matplotlib.pyplot as plt
import corner
import numpy as np


def plot_simulation_only(time_points, compartments, true_data, plot_dir,
                         model_cfg=None, population=None, compartment_choice=None):
    os.makedirs(plot_dir, exist_ok=True)

    # Defaults
    scale = 1.0
    time_for_plot = time_points
    xlabel = "Days"
    ylabel = "Population"
    yscale = "linear"

    # Apply scaling/time axis settings if available
    if model_cfg and "plot_settings" in model_cfg:
        settings = model_cfg["plot_settings"]
        if settings.get("time_unit") == "years":
            xlabel = "Time (years)"
        if settings.get("scale_by_population", False) and population:
            per_unit = settings.get("per_unit", 100000)
            scale = per_unit / population
            ylabel = f"Cases per {per_unit}"
        if settings.get("y_scale", "linear") == "log":
            yscale = "log"

    scaled_data = true_data * scale

    plt.figure(figsize=(10, 6))
    for i, comp in enumerate(compartments):
        plt.plot(time_for_plot, scaled_data[:, i], label=f"{comp} (true)")

    #if compartment_choice and "+" not in compartment_choice:
     #   if compartment_choice in compartments and compartment_choice != "S":
      #      comp_index = compartments.index(compartment_choice)
        #    plt.plot(time_for_plot, scaled_data[:, comp_index],
         #            label=f"{compartment_choice} (highlight)", linewidth=2.5)

    plt.title("True Simulation of Compartments (excluding S)")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.yscale(yscale)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "plot_simulation.png"))
    plt.close()



def plot_calibration_results(
    time_points,
    compartments,
    true_data,
    noisy_data,
    subset_t,
    subset_infected,
    fitted_results,
    model_name,
    plot_dir,
    true_params=None,
    param_names=None,
    mcmc_sampler=None,
    model_cfg=None,
    population=None,
    compartment_choice="I",
):
    """Plot calibration-related results (simulation vs noisy data vs fitted)."""

    os.makedirs(plot_dir, exist_ok=True)

    # --- scaling + time axis ---
    scale = 1.0
    xlabel, ylabel = "Days", "Population"
    time_for_plot, subset_time_for_plot = time_points, subset_t

    if model_cfg and "plot_settings" in model_cfg:
        settings = model_cfg["plot_settings"]

        # Time axis
        if settings.get("time_unit") == "years":
            xlabel = "Time (years)"

        # Scaling
        if settings.get("scale_by_population", False) and population:
            per_unit = settings.get("per_unit", 100000)
            scale = per_unit / population
            ylabel = f"Cases per {per_unit}"

    # --- scaling applied ---
    true_scaled = true_data * scale
    noisy_scaled = noisy_data * scale
    subset_scaled = subset_infected * scale

    # --- (1) Noisy data plot for all compartments ---
    plt.figure(figsize=(10, 6))
    for i, comp in enumerate(compartments):
        plt.plot(time_for_plot, noisy_scaled[:, i], "--", label=f"{comp} (noisy)")
    plt.title("Noisy Data (All Compartments)")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if model_cfg and model_cfg.get("plot_settings", {}).get("y_scale") == "logarithmic":
        plt.yscale("log")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "plot_noisy.png"))
    plt.close()

    # --- (2) Comparison plot for calibration compartment(s) ---
    plt.figure(figsize=(10, 6))

    # Select compartments
    if "+" in compartment_choice:
        comp_parts = [c.strip() for c in compartment_choice.split("+")]
        comp_indices = [compartments.index(c) for c in comp_parts if c in compartments]
        true_series = true_scaled[:, comp_indices].sum(axis=1)
        noisy_series = noisy_scaled[:, comp_indices].sum(axis=1)
    else:
        comp_index = compartments.index(compartment_choice)
        comp_indices = [comp_index]
        true_series = true_scaled[:, comp_index]
        noisy_series = noisy_scaled[:, comp_index]

    # True vs noisy subset
    plt.plot(time_for_plot, true_series, label=f"True {compartment_choice}", linewidth=2)
    plt.scatter(subset_time_for_plot, subset_scaled, label="Noisy Subset", color="black", zorder=5)

    # Add fitted trajectories
    for method, result in fitted_results.items():
        fitted = result["trajectory"] * scale
        if len(comp_indices) > 1:
            fitted_series = fitted[:, comp_indices].sum(axis=1)
        else:
            fitted_series = fitted[:, comp_indices[0]]
        plt.plot(time_for_plot, fitted_series, label=f"Fitted ({method})")

    plt.title(f"{compartment_choice}: Calibration Results")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if model_cfg and model_cfg.get("plot_settings", {}).get("y_scale") == "logarithmic":
        plt.yscale("log")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "plot_comparison.png"))
    plt.close()

    # --- (3) Parameter estimation bar plot ---
    if true_params is not None and param_names is not None:
        methods = list(fitted_results.keys())
        n_params = len(param_names)
        fig, axes = plt.subplots(1, n_params, figsize=(5 * n_params, 5), squeeze=False)

        for idx, pname in enumerate(param_names):
            ax = axes[0][idx]
            true_val = true_params[pname]

            estimates = [fitted_results[m]["params"][pname] for m in methods]
            ax.bar(methods, estimates, alpha=0.6, label="Estimated")

            if mcmc_sampler is not None:
                chain = mcmc_sampler.get_chain(discard=100, flat=True)
                valid_values = chain[:, idx][np.isfinite(chain[:, idx])]
                if valid_values.size > 0:
                    mcmc_mean = np.mean(valid_values)
                    ax.bar("MCMC Mean", mcmc_mean, color="orange", label="MCMC Mean")

            ax.axhline(true_val, color="red", linestyle="--", label="True")
            ax.set_title(f"Parameter: {pname}")
            ax.set_ylabel("Value")
            ax.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "parameter_estimation.png"))
        plt.close()

    # --- (4) MCMC corner plot ---
    if mcmc_sampler is not None and param_names is not None:
        try:
            samples = mcmc_sampler.get_chain(discard=100, flat=True)
            if samples.size > 0 and samples.shape[0] > samples.shape[1]:
                fig = corner.corner(samples, labels=param_names)
                save_path = os.path.join(plot_dir, "mcmc_corner_plot.png")
                fig.savefig(save_path, dpi=300)
                plt.close(fig)
        except Exception as e:
            print(f"[Error] Failed to generate MCMC plot: {e}")
