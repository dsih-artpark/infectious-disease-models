import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize
import os

os.makedirs("plots", exist_ok=True)

class EpidemicModel:
    def __init__(self, name, N, initial_conditions, params, model_func, observed_vars=['I']):
        self.name = name
        self.N = N
        self.initial_conditions = initial_conditions
        self.beta, self.gamma = params
        self.model_func = model_func
        self.observed_vars = observed_vars
        self.t = np.linspace(0, 200, 200)
        self.solution = None

    def simulate(self):
        self.solution = odeint(self.model_func, self.initial_conditions, self.t,
                               args=(self.N, self.beta, self.gamma))
        return self.solution.T

    def add_noise(self, noise_level=2):
        S, I, R = self.simulate()
        self.S_noisy = np.clip(S + np.random.normal(0, noise_level, size=S.shape), 0, self.N)
        self.I_noisy = np.clip(I + np.random.normal(0, noise_level, size=I.shape), 0, self.N)
        self.R_noisy = np.clip(R + np.random.normal(0, noise_level, size=R.shape), 0, self.N)

    def sample_subset(self, num_points=80):
        idx = np.sort(np.random.choice(len(self.t), size=num_points, replace=False))
        self.t_subset = self.t[idx]
        self.S_subset = self.S_noisy[idx]
        self.I_subset = self.I_noisy[idx]
        self.R_subset = self.R_noisy[idx]
        self.y0_est = self.S_subset[0], self.I_subset[0], self.R_subset[0]

    def loss(self, params):
        beta, gamma = params
        sol = odeint(self.model_func, self.y0_est, self.t_subset, args=(self.N, beta, gamma))
        S_sim, I_sim, R_sim = sol.T
        return np.mean((I_sim - self.I_subset) ** 2)

    def fit(self):
        bounds = [(0.0001, 1), (0.0001, 1)]
        methods = ['Nelder-Mead', 'BFGS', 'L-BFGS-B']
        self.fits = {}

        for method in methods:
            x0 = np.random.uniform(0, 1, size=2)
            result = minimize(self.loss, x0, method=method, bounds=bounds if method != 'BFGS' else None)
            self.fits[method] = {
                'beta': result.x[0],
                'gamma': result.x[1],
                'loss': result.fun,
                'start': x0,
                'nit': result.nit
            }

    def plot_trajectories(self):
        S, I, R = self.simulate()
        plt.figure()
        plt.plot(self.t, S, 'orange', label='Susceptible')
        plt.plot(self.t, I, 'red', label='Infected')
        plt.plot(self.t, R, 'green', label='Recovered')
        plt.title(f"{self.name} Model")
        plt.xlabel('Time (days)')
        plt.ylabel('Number of individuals')
        plt.legend()
        plt.grid()
        plt.savefig(f"plots/{self.name}_simulated.png")
        plt.show()

    def plot_noisy_data(self):
        plt.figure()
        plt.plot(self.t, self.S_noisy, 'orange', label='Susceptible (noisy)')
        plt.plot(self.t, self.I_noisy, 'red', label='Infected (noisy)')
        plt.plot(self.t, self.R_noisy, 'green', label='Recovered (noisy)')
        plt.title(f"{self.name} Model with Noise")
        plt.xlabel('Time (days)')
        plt.ylabel('Number of individuals')
        plt.legend()
        plt.grid()
        plt.savefig(f"plots/{self.name}_noisy.png")
        plt.show()

    def plot_loss_landscape(self):
        beta_vals = np.linspace(0.0001, 0.3, 100)
        gamma_vals = np.linspace(0.0001, 0.18, 100)
        loss_grid = np.zeros((len(beta_vals), len(gamma_vals)))

        for i, b in enumerate(beta_vals):
            for j, g in enumerate(gamma_vals):
                loss_grid[i, j] = self.loss([b, g])
        log_loss = np.log10(loss_grid + 1e-10)
        G, B = np.meshgrid(gamma_vals, beta_vals)

        plt.figure(figsize=(10, 8))
        cp = plt.contourf(G, B, log_loss, levels=100, cmap='viridis')
        plt.colorbar(cp).set_label('log10(Loss)')
        for method, data in self.fits.items():
            plt.scatter([data['gamma']], [data['beta']], label=method)
        plt.scatter([self.gamma], [self.beta], color='black', marker='x', s=100, label='True')
        plt.xlabel(r'$\gamma$')
        plt.ylabel(r'$\beta$')
        plt.title('Loss Landscape')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"plots/{self.name}_loss_landscape.png")
        plt.show()

    def plot_fitted_vs_noisy(self):
        best_fit = self.fits['L-BFGS-B']
        beta_fit, gamma_fit = best_fit['beta'], best_fit['gamma']
        sol = odeint(self.model_func, self.y0_est, self.t_subset, args=(self.N, beta_fit, gamma_fit))
        _, I_fit, _ = sol.T

        plt.figure()
        plt.plot(self.t, (self.solution[:, 1] / self.N) * 100, 'k-', label='True Infected (%)')
        plt.plot(self.t_subset, (I_fit / self.N) * 100, 'g-', label='L-BFGS-B Fit (%)')
        plt.scatter(self.t_subset, (self.I_subset / self.N) * 100, color='red', label='Noisy Infected (%)')
        plt.title(f"{self.name} Fitted Infected vs Noisy")
        plt.xlabel('Time (days)')
        plt.ylabel('Infected (% of population)')
        plt.grid()
        plt.legend()
        plt.savefig(f"plots/{self.name}_fitted_percentage.png")
        plt.show()

def sir_model(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

def seirsb_model(y, t, N, beta, sigma, gamma, omega):
    S, E, I, R = y
    dSdt = -beta * S * I / N + omega * R
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I - omega * R
    return dSdt, dEdt, dIdt, dRdt

np.random.seed(42)
N = 1000
S0, I0, R0 = N - 1, 1, 0
beta_true = 0.2
gamma_true = 1 / 10

sir = EpidemicModel("SIR", N, (S0, I0, R0), (beta_true, gamma_true), sir_model)
sir.simulate()
sir.add_noise(noise_level=2)
sir.sample_subset(num_points=80)
sir.fit()
sir.plot_trajectories()
sir.plot_noisy_data()
sir.plot_loss_landscape()
sir.plot_fitted_vs_noisy()
