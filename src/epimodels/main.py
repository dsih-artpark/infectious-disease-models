import argparse
import os
import numpy as np
from model import CompartmentalModel, Population
from config import cfg
from calibration import Calibrator
from plotting import plot_simulation_only, plot_calibration_results
import yaml

parser = argparse.ArgumentParser(description="Run simulation and calibration for a compartmental model.")
parser.add_argument("--model", type=str, required=True, help="Model name as defined in config (e.g., SIR_model)")
parser.add_argument("--calibrate", action="store_true", help="Run parameter calibration")
parser.add_argument("--update_config", action="store_true", help="Update config file with fitted parameters")
parser.add_argument("--compartment", type=str, default="I", help="Compartment to calibrate on (default: 'I')")
# parser.add_argument("--config", type=str, default="config.yml", help="Path to YAML config file")
args = parser.parse_args()

MODEL_NAME = args.model
MODEL_CFG = cfg[MODEL_NAME]
TIME = cfg["timescale"]
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

    # Normalize destination (remove suffix like _extra, _1, _2, etc.)
    if dst and "_" in dst:
        dst = dst.split("_")[0]

    TRANSITIONS.append({'from': src, 'to': dst, 'rate': expr})

INIT_CONDITIONS = MODEL_CFG["initial_conditions"]
POPULATION = MODEL_CFG["population"]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
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

np.savetxt(os.path.join(DATA_DIR, "true_data.csv"), true_data, delimiter=",")
np.savetxt(os.path.join(DATA_DIR, "noisy_data.csv"), noisy_data, delimiter=",")
np.savetxt(os.path.join(DATA_DIR, "time_points.csv"), time_points, delimiter=",")

# --- Step 2: calibration (optional) ---
fitted_results = None
sampler = None
subset_t = None
subset_infected = None
if args.calibrate:
    # Support compound compartments like "Is+Ir"
    if "+" in args.compartment:
        comp_parts = [c.strip() for c in args.compartment.split("+")]
        for c in comp_parts:
            if c not in COMPARTMENTS:
                raise ValueError(f"Compartment '{c}' not found. Available: {COMPARTMENTS}")
        comp_indices = [COMPARTMENTS.index(c) for c in comp_parts]
    else:
        if args.compartment not in COMPARTMENTS:
            raise ValueError(
                f"Compartment '{args.compartment}' not found in model. Available: {COMPARTMENTS}"
            )
        comp_indices = [COMPARTMENTS.index(args.compartment)]

    # Subset time points
    subset_indices = np.sort(
        np.random.choice(range(TIME + 1), size=int((TIME + 1) * SUBSET_RATIO), replace=False)
    )
    subset_t = time_points[subset_indices]

    # If compound: sum across multiple compartments
    if len(comp_indices) > 1:
        subset_infected = noisy_data[subset_indices][:, comp_indices].sum(axis=1)
    else:
        subset_infected = noisy_data[subset_indices, comp_indices[0]]

    # Pass indices instead of a single name
    calibrator = Calibrator(model, param_names, compartment=comp_indices)
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
        "compartment_indices": comp_indices,
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
        config_path = os.path.join(BASE_DIR, "config.yml")
        with open(config_path, "w") as f:
            yaml.safe_dump(cfg, f)
        print(f"Config file updated at {config_path}")

# --- Plot results ---
plot_simulation_only(
    time_points=time_points,
    compartments=COMPARTMENTS,
    true_data=true_data,
    plot_dir=PLOT_DIR,
    model_cfg=MODEL_CFG,
    population=POPULATION,
    compartment_choice=args.compartment,
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
        mcmc_sampler=sampler,
        model_cfg=MODEL_CFG,
        population=POPULATION,
        compartment_choice=args.compartment,
    )