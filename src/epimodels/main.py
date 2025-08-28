import argparse
import os
import numpy as np
from model import CompartmentalModel, Population
from config import cfg
from calibration import Calibrator
from plotting import plot_simulation_only, plot_calibration_results

parser = argparse.ArgumentParser(description="Run simulation and calibration for a compartmental model.")
parser.add_argument("--model", type=str, required=True, help="Model name as defined in config (e.g., SIR_model)")
parser.add_argument("--calibrate", action="store_true", help="Run parameter calibration")
parser.add_argument("--update_config", action="store_true", help="Update config file with fitted parameters")
parser.add_argument("--compartment", type=str, default="I", help="Compartment to calibrate on (default: 'I')")
# parser.add_argument("--config", type=str, default="config.yml", help="Path to YAML config file")
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

np.savetxt("data/true_data.csv", true_data, delimiter=",")
np.savetxt("data/noisy_data.csv", noisy_data, delimiter=",")
np.savetxt("data/time_points.csv", time_points, delimiter=",")

# --- Step 2: calibration (optional) ---
fitted_results = None
sampler = None
subset_t = None
subset_infected = None
if args.calibrate:
    if args.compartment not in COMPARTMENTS:
        raise ValueError(
            f"Compartment '{args.compartment}' not found in model. Available: {COMPARTMENTS}"
        )
    comp_idx = COMPARTMENTS.index(args.compartment)

    subset_indices = np.sort(
        np.random.choice(range(TIME + 1), size=int((TIME + 1) * SUBSET_RATIO), replace=False)
    )
    subset_t = time_points[subset_indices]
    subset_infected = noisy_data[subset_indices, comp_idx]

    calibrator = Calibrator(model, param_names, compartment=args.compartment)
    fitted_results = calibrator.fit(
        initial_conditions=INIT_CONDITIONS,
        full_time_points=time_points,
        subset_t=subset_t,
        subset_data=subset_infected,
        optimizers=OPTIMIZERS,
        compartments=COMPARTMENTS,
    )

    extras_fn = {
        "initial_conditions": INIT_CONDITIONS,
        "compartment_index": comp_idx,
        "sigma": 5.0,
    }
    sampler = Calibrator.run_mcmc(
        model=model,
        param_names=param_names,
        I_obs=subset_infected,
        t_obs=subset_t,
        extras_fn=extras_fn,
    )

    print("\nFinal Fitted Parameters:")
    for method, result in fitted_results.items():
        param_str = ", ".join(f"{k} = {v:.4f}" for k, v in result["params"].items())
        print(f"{method}: {param_str}")

    if args.update_config:
        print("Updating config dictionary with estimated parameters...")
        best_params = fitted_results.get("best", {}).get("params", {})
        MODEL_CFG["parameters"].update(best_params)
        print("Config updated in memory (not written to file).")

# --- Plot results ---
plot_simulation_only(
    time_points=time_points,
    compartments=COMPARTMENTS,
    true_data=true_data,
    plot_dir=PLOT_DIR
)

# Only: plot calibration/fitting results if calibration is run
if args.calibrate and fitted_results is not None:
    plot_calibration_results(
        time_points=time_points,
        compartments=COMPARTMENTS,
        true_data=true_data,
        noisy_data=noisy_data,
        subset_t=subset_t,
        subset_infected=subset_infected,
        fitted_results=fitted_results,
        model_name=MODEL_NAME,
        plot_dir=PLOT_DIR,
        true_params=PARAMS,
        param_names=param_names,
        mcmc_sampler=sampler
    )