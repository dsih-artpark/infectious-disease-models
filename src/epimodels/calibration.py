import emcee
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
    
    @staticmethod
    def log_prior(theta, param_names):
        param_dict = dict(zip(param_names, theta))
        for key, val in param_dict.items():
            if val <= 0 or val > 5:  # uniform prior bounds
                return -np.inf
        return 0.0  
    
    @staticmethod
    def log_likelihood(theta, param_names, model, I_obs, t_obs, extras_fn):
        param_dict = dict(zip(param_names, theta))
        model.set_parameters(param_dict)
        try:
            sim_data = model.simulate(extras_fn['initial_conditions'], t_obs)
            I_sim = sim_data[:, extras_fn['compartment_index']]
            if np.any(np.isnan(I_sim)) or np.any(I_sim < 0) or np.any(I_sim > 1e6):
                return -np.inf
            sigma = extras_fn.get('sigma', 5.0)
            return -0.5 * np.sum((I_obs - I_sim)**2 / sigma**2)
        except Exception as e:
            print(f"[log_likelihood error] Params: {param_dict} -> {e}")
            return -np.inf
    
    @classmethod
    def log_posterior(cls, theta, param_names, model, I_obs, t_obs, extras_fn):
        lp = cls.log_prior(theta, param_names)
        if not np.isfinite(lp):
            return -np.inf
        return lp + cls.log_likelihood(theta, param_names, model, I_obs, t_obs, extras_fn)

    @classmethod
    def run_mcmc(cls, model, param_names, I_obs, t_obs, extras_fn, n_walkers=32, n_steps=500):
        ndim = len(param_names)
        initial = np.array([model.parameters[k] for k in param_names])
        pos = initial + 1e-2 * np.random.randn(n_walkers, ndim)

        sampler = emcee.EnsembleSampler(n_walkers, ndim, cls.log_posterior, 
                                    args=(param_names, model, I_obs, t_obs, extras_fn))
        sampler.run_mcmc(pos, n_steps, progress=True)
        return sampler
    
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


