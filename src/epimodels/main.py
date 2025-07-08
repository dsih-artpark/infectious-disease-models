import argparse
import numpy as np
import os
from scipy.optimize import minimize
from model import CompartmentalModel, Population
from config import cfg
from calibration import Calibrator
from plotting import plot_results

parser = argparse.ArgumentParser(description="Run simulation and calibration for a compartmental model.")
parser.add_argument("--model", type=str, required=True, help="Model name as defined in config (e.g., SIR_model)")
args = parser.parse_args()
MODEL_NAME = args.model

MODEL_CFG = cfg[MODEL_NAME]
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

PLOT_DIR = os.path.join("plots", MODEL_NAME)
DATA_DIR = os.path.join("data", MODEL_NAME)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

time_points = np.linspace(0, TIME, TIME + 1)

pop = Population(POPULATION, INIT_CONDITIONS)
model = CompartmentalModel(COMPARTMENTS, PARAMS, TRANSITIONS, population=POPULATION)

true_data = np.array(model.simulate(INIT_CONDITIONS, time_points))

np.random.seed(42)
noisy_data = model.add_noise(true_data, NOISE_STD)

subset_indices = np.sort(np.random.choice(range(TIME + 1), size=int((TIME + 1) * SUBSET_RATIO), replace=False))
subset_t = time_points[subset_indices]
subset_infected = noisy_data[subset_indices, COMPARTMENTS.index('I')]

calibrator = Calibrator(model, param_names, compartment='I')
fitted_results = calibrator.fit(
    initial_conditions=INIT_CONDITIONS,
    full_time_points=time_points,
    subset_t=subset_t,
    subset_data=subset_infected,
    optimizers=OPTIMIZERS,
    compartments=COMPARTMENTS
)

np.savetxt("data/true_data.csv", true_data, delimiter=",")
np.savetxt("data/noisy_data.csv", noisy_data, delimiter=",")
np.savetxt("data/time_points.csv", time_points, delimiter=",")

plot_results(
    time_points=time_points,
    compartments=COMPARTMENTS,
    true_data=true_data,
    noisy_data=noisy_data,
    subset_t=subset_t,
    subset_infected=subset_infected,
    fitted_results=fitted_results,
    model_name=MODEL_NAME,
    plot_dir=PLOT_DIR
)

print("\nFinal Fitted Parameters:")
for method, result in fitted_results.items():
    param_str = ", ".join(f"{k} = {v:.4f}" for k, v in result['params'].items())
    print(f"{method}: {param_str}")


