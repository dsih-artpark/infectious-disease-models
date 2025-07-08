# calibration.py
import numpy as np
from scipy.optimize import minimize

class Calibrator:
    def __init__(self, model, param_names, compartment='I'):
        self.model = model
        self.param_names = param_names
        self.compartment = compartment

    def loss_function(self, param_array, initial_conditions, time_points, target_data, compartment_index):
        self.model.parameters = dict(zip(self.param_names, param_array))
        sim = self.model.simulate(initial_conditions, time_points)
        sim_values = np.array(sim)[:, compartment_index]
        return np.mean((sim_values - target_data) ** 2)

    def fit(
        self, initial_conditions, full_time_points, subset_t, subset_data,
        optimizers, compartments
    ):
        results = {}
        initial_guess = np.array([self.model.parameters[p] for p in self.param_names])
        bounds = [(0.0001, 1.0)] * len(self.param_names)
        comp_index = compartments.index(self.compartment)

        for method in optimizers:
            res = minimize(
                self.loss_function,
                x0=initial_guess,
                args=(initial_conditions, subset_t, subset_data, comp_index),
                method=method,
                bounds=bounds if method in ['L-BFGS-B', 'TNC'] else None
            )
            fitted_params = dict(zip(self.param_names, res.x))
            self.model.parameters = fitted_params
            full_trajectory = np.array(self.model.simulate(initial_conditions, full_time_points))
            results[method] = {
                'params': fitted_params,
                'trajectory': full_trajectory
            }

        return results
