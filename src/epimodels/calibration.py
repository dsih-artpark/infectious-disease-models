import emcee
import numpy as np
from scipy.optimize import minimize

class Calibrator:
    def __init__(self, model, param_names, compartment, parameter_bounds=None, scale_by_population=False):
        """
        model: CompartmentalModel
        param_names: list of parameter names
        compartment: list of indices for compartments of interest
        parameter_bounds: dict with {param: (low, high)}, default fallback [1e-5, 5]
        scale_by_population: bool, whether to normalize by population
        """
        self.model = model
        self.param_names = param_names
        self.compartment = compartment  # always a list
        self.parameter_bounds = parameter_bounds or {p: (1e-5, 5) for p in param_names}
        self.scale_by_population = scale_by_population

    def loss_function(self, param_array, initial_conditions, time_points, target_data, comp_indices, extras_fn=None):
        self.model.parameters = dict(zip(self.param_names, param_array))
        try:
            sim = self.model.simulate(initial_conditions, time_points, extras_fn=extras_fn.get("extras_fn"))
            if len(comp_indices) > 1:
                sim_values = np.array(sim)[:, comp_indices].sum(axis=1)
            else:
                sim_values = np.array(sim)[:, comp_indices[0]]

            # normalize if per-unit scaling requested
            if self.scale_by_population and self.model.population:
                sim_values = sim_values / self.model.population

            if np.any(np.isnan(sim_values)) or np.any(sim_values > 1e6):
                return 1e10
            return np.mean((sim_values - target_data) ** 2)
        except Exception as e:
            print(f"[Loss Function Error] {e}")
            return 1e10

    def log_prior(self, theta):
        param_dict = dict(zip(self.param_names, theta))
        for key, val in param_dict.items():
            low, high = self.parameter_bounds.get(key, (1e-5, 5))
            if val < low or val > high:
                return -np.inf
        return 0.0  

    def log_likelihood(self, theta, I_obs, t_obs, extras_fn):
        param_dict = dict(zip(self.param_names, theta))
        self.model.set_parameters(param_dict)
        try:
            sim_data = self.model.simulate(extras_fn['initial_conditions'], t_obs, extras_fn=extras_fn.get("extras_fn"))
            comp_indices = extras_fn['compartment_indices']
            if len(comp_indices) > 1:
                I_sim = sim_data[:, comp_indices].sum(axis=1)
            else:
                I_sim = sim_data[:, comp_indices[0]]

            if self.scale_by_population and self.model.population:
                I_sim = I_sim / self.model.population

            if np.any(np.isnan(I_sim)) or np.any(I_sim < 0) or np.any(I_sim > 1e6):
                print(f"[Sim failure] I_sim range: {I_sim.min()} - {I_sim.max()}")
                return -np.inf
            sigma = extras_fn.get('sigma', 5.0)
            return -0.5 * np.sum((I_obs - I_sim) ** 2 / sigma**2)
        except Exception as e:
            print(f"[log_likelihood error] Params: {param_dict} -> {e}")
            return -np.inf

    def log_posterior(self, theta, I_obs, t_obs, extras_fn):
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(theta, I_obs, t_obs, extras_fn)

    def run_mcmc(self, I_obs, t_obs, extras_fn, n_walkers=32, n_steps=500, seed=42):
        np.random.seed(seed)
        ndim = len(self.param_names)
        initial = np.array([self.model.parameters[k] for k in self.param_names])
        pos = np.abs(initial + 1e-2 * np.random.randn(n_walkers, ndim))

        sampler = emcee.EnsembleSampler(
            n_walkers, ndim, self.log_posterior,
            args=(I_obs, t_obs, extras_fn)
        )
        sampler.run_mcmc(pos, n_steps, progress=True)
        return sampler

    def fit(self, initial_conditions, full_time_points, subset_t, subset_data,
        optimizers, compartments, extras_fn=None):
        results = {}
        initial_guess = np.array([self.model.parameters[p] for p in self.param_names])
        bounds = [self.parameter_bounds.get(p, (1e-8, 10.0)) for p in self.param_names]
        comp_indices = self.compartment

        for method in optimizers:
            res = minimize(
                self.loss_function,
                x0=initial_guess,
                args=(initial_conditions, subset_t, subset_data, comp_indices,extras_fn),
                method=method,
                bounds=bounds if method in ['L-BFGS-B', 'TNC'] else None
            )
            fitted_params = dict(zip(self.param_names, res.x))
            self.model.parameters = fitted_params
            full_trajectory = np.array(self.model.simulate(initial_conditions, full_time_points, extras_fn=extras_fn))
            if self.scale_by_population and self.model.population:
                full_trajectory = full_trajectory / self.model.population
            results[method] = {
                'params': fitted_params,
                'trajectory': full_trajectory,
                'loss': res.fun
            }

        return results
