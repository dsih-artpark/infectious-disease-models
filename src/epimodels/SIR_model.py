import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import random
import os
from model import CompartmentalModel, Population
from config import cfg

PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# MODEL = cfg["model_name"]
TIME = cfg["days"]
NOISE_STD = cfg["noise_std"]
SUBSET_RATIO = cfg["subset_ratio"]
OPTIMIZERS = cfg["optimizers"]

MODEL_CFG = cfg['SIR_model']
PARAMS = MODEL_CFG["parameters"]
COMPARTMENTS = MODEL_CFG["compartments"]
TRANSITIONS = []
for k, expr in MODEL_CFG['transitions'].items():
    src, dst = k.split('->') if '->' in k else (None, k)
    src = src.strip() if src else None
    dst = dst.strip()
    TRANSITIONS.append({'from': src, 'to': dst, 'rate': expr})
INIT_CONDITIONS = MODEL_CFG["initial_conditions"]
POPULATION = MODEL_CFG["population"]

time_points = np.linspace(0, TIME, TIME + 1)

# Initialize model
pop = Population(POPULATION, INIT_CONDITIONS)
model = CompartmentalModel(COMPARTMENTS, PARAMS, TRANSITIONS, population=POPULATION)

# Simulate model
true_sol = model.simulate(INIT_CONDITIONS, time_points)
true_data = np.array(true_sol)

# Plot true trajectory
plt.figure(figsize=(10, 6))
for i, comp in enumerate(COMPARTMENTS):
    plt.plot(time_points, true_data[:, i], label=f"{comp} (true)")
plt.title("True Simulation of Compartments")
plt.xlabel("Days")
plt.ylabel("Population")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "plot_simulation.png"))
plt.show()

# Add noise 
np.random.seed(42)
noisy_data = true_data + np.random.normal(0, NOISE_STD, true_data.shape)

# Plot noisy data
plt.figure(figsize=(10, 6))
for i, comp in enumerate(COMPARTMENTS):
    plt.plot(time_points, noisy_data[:, i], label=f"{comp} (noisy)", linestyle="--")
plt.title("Noisy Data for All Compartments")
plt.xlabel("Days")
plt.ylabel("Population")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "plot_noisy.png"))
plt.show()


subset_indices = np.sort(np.random.choice(range(TIME + 1), size=int((TIME + 1) * SUBSET_RATIO), replace=False))
subset_t = time_points[subset_indices]
subset_infected = noisy_data[subset_indices, COMPARTMENTS.index('I')]

def loss_function(param_array, model, initial_conditions, target_t, target_data):
    beta, gamma = param_array
    model.parameters = {'beta': beta, 'gamma': gamma}
    sim = model.simulate(initial_conditions, target_t)
    sim_infected = np.array(sim)[:, COMPARTMENTS.index('I')]
    return np.mean((sim_infected - target_data) ** 2)

# Fit using each optimizer 
fitted_results = {}
for method in OPTIMIZERS:
    res = minimize(
        loss_function,
        x0=np.array([0.2, 0.1]),
        args=(model, INIT_CONDITIONS, subset_t, subset_infected),
        method=method
    )
    fitted_params = res.x
    model.parameters = {'beta': fitted_params[0], 'gamma': fitted_params[1]}
    fitted_sol = model.simulate(INIT_CONDITIONS, time_points)
    fitted_results[method] = {
        'params': fitted_params,
        'trajectory': np.array(fitted_sol)
    }

# Plot Infected: True vs Noisy Subset vs Fitted 
plt.figure(figsize=(10, 6))
plt.plot(time_points, true_data[:, COMPARTMENTS.index('I')], label="True Infected", linewidth=2)
plt.scatter(subset_t, subset_infected, label="Noisy Subset", color="black", zorder=5)

for method, result in fitted_results.items():
    plt.plot(time_points, result['trajectory'][:, COMPARTMENTS.index('I')],
             label=f"Fitted ({method})")

plt.title("Infected Compartment: True vs Noisy Subset vs Fitted")
plt.xlabel("Days")
plt.ylabel("Population")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "plot_comparison.png"))
plt.show()

# Print final fitted parameters 
print("\nFinal Fitted Parameters:")
for method, result in fitted_results.items():
    beta, gamma = result['params']
    print(f"{method}: beta = {beta:.4f}, gamma = {gamma:.4f}")









# def main():
    # t = np.linspace(0, TIME, TIME)
    # model = CompartmentalModel(COMPARTMENTS, PARAMS, TRANSITIONS)
    # infected_idx = COMPARTMENTS.index("I")
    
    # y0 = [INIT_CONDITIONS[c] for c in COMPARTMENTS]

    # clean_sol = simulate(model, y0, t)
    # noisy_data = add_noise(clean_sol, NOISE_STD)
    # noisy_I = noisy_data[:, infected_idx]
    # # Subsample for fitting
    # t_sub, I_sub, idx = subsample(t, noisy_I, SUBSET_RATIO)

    # # Fit
    # initial_guess = [model.parameters[k] for k in model.parameters]
    # param_names = list(model.parameters.keys())
    # fit_results = fit_model(model, t_sub, I_sub, initial_guess, param_names, y0)

    # # Print fitted info
    # for opt, res in fit_results.items():
    #     print(f"\nOptimizer: {opt}")
    #     print(f"Loss: {res['loss']:.4f}")
    #     print("Estimated Parameters:")
    #     for k, v in res["params"].items():
    #         print(f"  {k}: {v:.4f}")

    # # Plotting
    # plot_simulation(t, clean_sol)
    # plot_noisy(t, noisy_data)
    # plot_comparison(t, clean_sol[:, 1], noisy_I, t_sub, I_sub, fit_results)

# if __name__ == "__main__":
#     main()





