import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize

np.random.seed(42)

def sir_model(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

def seirs_model(y, t, N, beta, gamma, sigma, xi):
    S, E, I, R = y
    dSdt = -beta * S * I / N + xi * R
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I - xi * R
    return dSdt, dEdt, dIdt, dRdt

def init_sir(N):
    I0 = 1
    R0 = 0
    S0 = N - I0 - R0
    return (S0, I0, R0)

def init_seirs(N):
    E0 = 0
    I0 = 1
    R0 = 0
    S0 = N - E0 - I0 - R0
    return (S0, E0, I0, R0)

models = {
    "SIR": {
        "init": init_sir,
        "params": {
            "beta": 0.2,
            "gamma": 1 / 10
        },
        "model_func": sir_model
    },
    "SEIRS": {
        "init": init_seirs,
        "params": {
            "beta": 0.2,
            "gamma": 1 / 10,
            "sigma": 1 / 5.2,
            "xi": 1 / 90
        },
        "model_func": seirs_model
    }
}

class EpidemicModel:
    def __init__(self, name, N, y0, params, model_func):
        self.name = name
        self.N = N
        self.y0 = y0
        self.params = params
        self.model_func = model_func
        self.t = np.linspace(0, 200, 200)
        self.result = None

    def simulate(self):
        self.result = odeint(self.model_func, self.y0, self.t, args=(self.N, *self.params))
        self.S, *rest = self.result.T
        if self.name == "SIR":
            self.I, self.R = rest
        elif self.name == "SEIRS":
            self.E, self.I, self.R = rest

    def add_noise(self, noise_level=2):
        self.S_noisy = np.clip(self.S + np.random.normal(0, noise_level, size=self.S.shape), 0, self.N)
        self.I_noisy = np.clip(self.I + np.random.normal(0, noise_level, size=self.I.shape), 0, self.N)
        self.R_noisy = np.clip(self.R + np.random.normal(0, noise_level, size=self.R.shape), 0, self.N)
        if self.name == "SEIRS":
            self.E_noisy = np.clip(self.E + np.random.normal(0, noise_level, size=self.E.shape), 0, self.N)

    def sample_subset(self, num_points=80):
        idx = np.sort(np.random.choice(len(self.t), size=num_points, replace=False))
        self.t_subset = self.t[idx]
        self.I_subset = self.I_noisy[idx]
        self.S0_est = self.S_noisy[idx[0]]
        self.I0_est = self.I_noisy[idx[0]]
        self.R0_est = self.R_noisy[idx[0]]

    def loss(self, params):
        if self.name == "SIR":
            y0 = (self.S0_est, self.I0_est, self.R0_est)
        else:
            y0 = (self.S0_est, 0, self.I0_est, self.R0_est)
        sol = odeint(self.model_func, y0, self.t_subset, args=(self.N, *params))
        return np.mean((sol[:, -2] - self.I_subset) ** 2)

    def fit(self):
        guess = np.random.uniform(0, 1, size=len(self.params))
        bounds = [(0.0001, 1)] * len(self.params)
        res = minimize(self.loss, guess, method='L-BFGS-B', bounds=bounds)
        self.fitted_params = res.x

    def simulate_with_fit(self):
        if self.name == "SIR":
            y0 = (self.S0_est, self.I0_est, self.R0_est)
        else:
            y0 = (self.S0_est, 0, self.I0_est, self.R0_est)
        sol = odeint(self.model_func, y0, self.t_subset, args=(self.N, *self.fitted_params))
        return sol.T

    def plot_trajectories(self):
        plt.figure()
        plt.plot(self.t, self.S, 'orange', label='Susceptible')
        if self.name == "SEIRS":
            plt.plot(self.t, self.E, 'blue', label='Exposed')
        plt.plot(self.t, self.I, 'red', label='Infected')
        plt.plot(self.t, self.R, 'green', label='Recovered')
        plt.title(f'{self.name} Model')
        plt.xlabel('Time (days)')
        plt.ylabel('Number of individuals')
        plt.ylim([0, self.N])
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

    def plot_noisy_data(self):
        plt.figure()
        plt.plot(self.t, self.S_noisy, 'orange', label='Susceptible (noisy)')
        if self.name == "SEIRS":
            plt.plot(self.t, self.E_noisy, 'blue', label='Exposed (noisy)')
        plt.plot(self.t, self.I_noisy, 'red', label='Infected (noisy)')
        plt.plot(self.t, self.R_noisy, 'green', label='Recovered (noisy)')
        plt.title(f'{self.name} Model with Noise')
        plt.xlabel('Time (days)')
        plt.ylabel('Number of individuals')
        plt.ylim([0, self.N])
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

    def plot_loss_landscape(self):
        if len(self.params) != 2:
            print("Loss landscape only available for 2D parameter models like SIR.")
            return
        beta_vals = np.linspace(0.0001, 0.3, 100)
        gamma_vals = np.linspace(0.0001, 0.3, 100)
        B, G = np.meshgrid(beta_vals, gamma_vals)
        Loss = np.zeros_like(B)
        for i in range(B.shape[0]):
            for j in range(B.shape[1]):
                Loss[i, j] = self.loss([B[i, j], G[i, j]])
        plt.figure(figsize=(8, 6))
        cp = plt.contourf(G, B, np.log10(Loss + 1e-10), levels=100, cmap='viridis')
        plt.colorbar(cp, label=r'$\log_{10}$(Loss)')
        plt.xlabel(r'$\gamma$')
        plt.ylabel(r'$\beta$')
        plt.title('Loss Landscape')
        plt.scatter([self.params[1]], [self.params[0]], c='black', marker='x', s=100, label='True')
        plt.scatter([self.fitted_params[1]], [self.fitted_params[0]], c='red', marker='o', label='Fit')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_fitted_vs_noisy(self):
        result = self.simulate_with_fit()
        I_fit = result[-2]
        plt.figure()
        plt.plot(self.t, (self.I / self.N) * 100, 'k-', label='True Infected (%)', linewidth=2)
        plt.plot(self.t_subset, (I_fit / self.N) * 100, 'g-', label='Fitted Infected (%)', linewidth=2)
        plt.scatter(self.t_subset, (self.I_subset / self.N) * 100, color='red', marker='x', label='Noisy Infected (%)')
        plt.xlabel('Time (days)')
        plt.ylabel('Infected (% of population)')
        plt.title('Fitted Infected Phase vs Noisy Data')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

def run_epidemic_model(model_name, N=1000, noise_level=2, num_points=80):
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}")
    entry = models[model_name]
    y0 = entry['init'](N)
    params = list(entry['params'].values())
    model_func = entry['model_func']
    model = EpidemicModel(model_name, N, y0, params, model_func)
    model.simulate()
    model.add_noise(noise_level=noise_level)
    model.sample_subset(num_points=num_points)
    model.fit()
    model.plot_trajectories()
    model.plot_noisy_data()
    model.plot_loss_landscape()
    model.plot_fitted_vs_noisy()

if __name__ == '__main__':
    run_epidemic_model('SEIRS')    
