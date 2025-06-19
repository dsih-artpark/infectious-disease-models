import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize


class EpidemicModel:
    def __init__(self, model_name, initial_params, initial_state, population, t):
        self.model_name = model_name.upper()
        self.params = initial_params
        self.initial_state = initial_state
        self.N = population
        self.T = t
        self.t = np.linspace(0, t, t)
        self.subset_indices = None
        self.fitted_params = None

    def _sir(self, y, t, beta, gamma):
        S, I, R = y
        dSdt = -beta * S * I / self.N
        dIdt = beta * S * I / self.N - gamma * I
        dRdt = gamma * I
        return [dSdt, dIdt, dRdt]

    def _seir(self, y, t, beta, sigma, gamma):
        S, E, I, R = y
        dSdt = -beta * S * I / self.N
        dEdt = beta * S * I / self.N - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return [dSdt, dEdt, dIdt, dRdt]

    def simulate(self, params=None, y0=None, t=None):
        if params is None:
            params = self.params
        if y0 is None:
            y0 = self.initial_state
        if t is None:
            t = self.t

        if self.model_name == "SIR":
            return odeint(self._sir, y0, t, args=tuple(params))
        elif self.model_name == "SEIR":
            return odeint(self._seir, y0, t, args=tuple(params))
        else:
            raise NotImplementedError(f"Model {self.model_name} not implemented.")

    def add_noise(self, data, std=5.0):
        noise = np.random.normal(0, std, size=data.shape)
        return np.clip(data + noise, 0, None)

    def subset_data(self, data, ratio):
        n = int(len(data) * ratio)
        indices = np.sort(np.random.choice(len(data), n, replace=False))
        self.subset_indices = indices
        return data[indices]

    def loss(self, params, observed_data):
        sim = self.simulate(params)
        if self.subset_indices is not None:
            sim = sim[self.subset_indices]
        return np.mean((sim - observed_data) ** 2)

    def fit(self, data_subset, optimizer="BFGS"):
        def objective(p):
            return self.loss(p, data_subset)

        res = minimize(
            objective,
            x0=self.params,
            method=optimizer,
            bounds=[(0, 5)] * len(self.params)
        )
        self.fitted_params = res.x
        return res.x
