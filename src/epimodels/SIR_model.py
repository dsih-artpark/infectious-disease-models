import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
import sys
import matplotlib.pyplot as plt
import random
import os
from model import CompartmentalModel, Population
from config import cfg

model_keys = [k for k in cfg.keys() if k.endswith("_model")]
if len(model_keys) != 1:
    raise ValueError(f"Expected exactly one model in config, found: {model_keys}")

MODEL_NAME = model_keys[0]
MODEL_CFG = cfg[MODEL_NAME]

PLOT_DIR = os.path.join("plots", MODEL_NAME)
os.makedirs(PLOT_DIR, exist_ok=True)

TIME = cfg["days"]
NOISE_STD = cfg["noise_std"]
SUBSET_RATIO = cfg["subset_ratio"]
OPTIMIZERS = cfg["optimizers"]

PARAMS = MODEL_CFG["parameters"]
param_names = list(PARAMS.keys())
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
noisy_data = model.add_noise(true_data, NOISE_STD)

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

def loss_function(param_array, model, initial_conditions, target_t, target_data, param_names):
    model.parameters = dict(zip(param_names, param_array))
    sim = model.simulate(initial_conditions, target_t)
    sim_infected = np.array(sim)[:, COMPARTMENTS.index('I')]
    return np.mean((sim_infected - target_data) ** 2)

# Fit using each optimizer 
fitted_results = {}
initial_guess = np.array([PARAMS[p] for p in param_names])

for method in OPTIMIZERS:
    res = minimize(
        loss_function,
        x0=initial_guess,
        args=(model, INIT_CONDITIONS, subset_t, subset_infected, param_names),
        method=method
    )
    fitted_params = dict(zip(param_names, res.x))
    model.parameters = fitted_params
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
plt.close()

# Print final fitted parameters 
print("\nFinal Fitted Parameters:")
for method, result in fitted_results.items():
    param_str = ", ".join(f"{k} = {v:.4f}" for k, v in result['params'].items())
    print(f"{method}: {param_str}")

os.makedirs("data", exist_ok=True)
np.savetxt("data/true_data.csv", true_data, delimiter=",")
np.savetxt("data/noisy_data.csv", noisy_data, delimiter=",")
np.savetxt("data/time_points.csv", time_points, delimiter=",")





