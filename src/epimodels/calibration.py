import numpy as np
import pandas as pd
from scipy.optimize import minimize

class Calibrator:
    def __init__(self, model, compartments, time_points, true_data_path, noisy_data_path, subset_ratio):
        self.model = model
        self.compartments = compartments
        self.t = time_points
        self.true_data = pd.read_csv(true_data_path).values
        self.noisy_data = pd.read_csv(noisy_data_path).values
        self.subset_ratio = subset_ratio
        self.results = {}

        self.subset_indices = np.sort(
            np.random.choice(range(len(self.t)), size=int(len(self.t) * self.subset_ratio), replace=False)
        )
        self.subset_t = self.t[self.subset_indices]
        self.subset_I = self.noisy_data[self.subset_indices, self.compartments.index("I")]

    def loss_function(self, param_array):
        param_names = list(self.model.parameters.keys())
        self.model.parameters = dict(zip(param_names, param_array))
        sim = self.model.simulate(self.model.initial_conditions, self.subset_t)
        sim_I = np.array(sim)[:, self.compartments.index("I")]
        return np.mean((sim_I - self.subset_I) ** 2)

    def fit(self, methods, x0):
        for method in methods:
            res = minimize(self.loss_function, x0=x0, method=method)
            fitted_params = res.x
            self.model.parameters = dict(zip(self.model.parameters.keys(), fitted_params))
            fitted_traj = self.model.simulate(self.model.initial_conditions, self.t)
            self.results[method] = {
                'params': fitted_params,
                'trajectory': np.array(fitted_traj)
            }
        return self.results
