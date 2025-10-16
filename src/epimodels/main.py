import argparse
import os
import numpy as np
import yaml

from model import CompartmentalModel, Population
from config import cfg
from calibration import Calibrator
from plotting import plot_simulation_only, plot_calibration_results
from time_varying_params import generate_param_series, make_extras_fn_from_series

parser = argparse.ArgumentParser(description="Run simulation and calibration for a compartmental model.")
parser.add_argument("--model", type=str, required=True, help="Model name as defined in config (e.g., SIR_model)")
parser.add_argument("--calibrate", action="store_true", help="Run parameter calibration")
parser.add_argument("--update_config", action="store_true", help="Update config file with fitted parameters")
parser.add_argument("--compartment", type=str, default="I", help="Compartment to calibrate on (default: 'I')")
args = parser.parse_args()

MODEL_NAME = args.model
MODEL_CFG = cfg[MODEL_NAME]
TIME = MODEL_CFG["simulation_time"]
# Optional time-varying parameter helper 
def build_extras_fn(model_cfg):
    """Return a function that provides time-dependent parameters if defined."""
    if "time_varying" not in model_cfg:
        return None

    def extras_fn(t, y):
        extras = {}
        for param, spec in model_cfg["time_varying"].items():
            if isinstance(spec, dict) and "schedule" in spec:
                # Example: schedule = [{"t": 5, "value": 0.05}, {"t": 10, "value": 0.01}]
                last_value = spec.get("default", model_cfg["parameters"].get(param, 0))
                for entry in sorted(spec["schedule"], key=lambda e: e["t"]):
                    if t >= entry["t"]:
                        last_value = entry["value"]
                extras[param] = last_value
            elif callable(spec):
                extras[param] = spec(t)
        return extras

    return extras_fn

NOISE_STD = MODEL_CFG["calibration_settings"]["noise_std"]
SUBSET_RATIO = MODEL_CFG["calibration_settings"]["subset_ratio"]
OPTIMIZERS = MODEL_CFG["calibration_settings"]["optimizers"]
TARGET_DATA = MODEL_CFG["calibration_settings"]["target_data"]
y_scale = MODEL_CFG["plot_settings"]["y_scale"]
scale_by_pop = MODEL_CFG["plot_settings"]["scale_by_population"]
PARAMS = MODEL_CFG["parameters"]
param_names = list(PARAMS.keys())
COMPARTMENTS = MODEL_CFG["compartments"]

# Build transitions
TRANSITIONS = []
for k, expr in MODEL_CFG['transitions'].items():
    src, dst = k.split('->') if '->' in k else (None, k)
    src = src.strip() if src else None
    dst = dst.strip()
    if dst and "_" in dst:  # normalize suffixes
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

# --- Step 1: simulate or load data ---
true_path = os.path.join(DATA_DIR, "true_data.csv")
noisy_path = os.path.join(DATA_DIR, "noisy_data.csv")
time_path = os.path.join(DATA_DIR, "time_points.csv")
extras_fn = build_extras_fn(MODEL_CFG)
if os.path.exists(true_path) and os.path.exists(noisy_path) and os.path.exists(time_path):
    true_data = np.loadtxt(true_path, delimiter=",")
    noisy_data = np.loadtxt(noisy_path, delimiter=",")
    time_points = np.loadtxt(time_path, delimiter=",")
else:
    true_data = np.array(model.simulate(INIT_CONDITIONS, time_points, extras_fn=extras_fn))
    np.random.seed(42)
    noisy_data = model.add_noise(true_data, NOISE_STD)

    np.savetxt(true_path, true_data, delimiter=",")
    np.savetxt(noisy_path, noisy_data, delimiter=",")
    np.savetxt(time_path, time_points, delimiter=",")

# Load calibration target
target_path = os.path.join(DATA_DIR, os.path.basename(TARGET_DATA))
if not os.path.exists(target_path):
    raise FileNotFoundError(f"Target data file '{TARGET_DATA}' not found in {DATA_DIR}")
target_data = np.loadtxt(target_path, delimiter=",")

# --- Step 2: calibration (optional) ---
fitted_results = None
sampler = None
subset_t = None
subset_infected = None
if args.calibrate:
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

    subset_indices = np.sort(
        np.random.choice(range(TIME + 1), size=int((TIME + 1) * SUBSET_RATIO), replace=False)
    )
    subset_t = time_points[subset_indices]

    if len(comp_indices) > 1:
        subset_infected = target_data[subset_indices][:, comp_indices].sum(axis=1)
    else:
        subset_infected = target_data[subset_indices, comp_indices[0]]

    calibrator = Calibrator(model, param_names, compartment=comp_indices)
    fitted_results = calibrator.fit(
        initial_conditions=INIT_CONDITIONS,
        full_time_points=time_points,
        subset_t=subset_t,
        subset_data=subset_infected,
        optimizers=OPTIMIZERS,
        compartments=COMPARTMENTS,
        extras_fn=extras_fn,
    )

    sampler = calibrator.run_mcmc(
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
