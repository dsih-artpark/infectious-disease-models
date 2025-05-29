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

def si_model(y, t, N, beta):
    S, I = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N
    return dSdt, dIdt

def sis_model(y, t, N, beta, gamma):
    S, I = y
    dSdt = -beta * S * I / N + gamma * I
    dIdt = beta * S * I / N - gamma * I
    return dSdt, dIdt

def seir_model(y, t, N, beta, gamma, sigma):
    S, E, I, R = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
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

def init_si(N):
    I0 = 1
    return (N - I0, I0)

def init_sis(N):
    I0 = 1
    return (N - I0, I0)

def init_seir(N):
    E0, I0, R0 = 0, 1, 0
    return (N - E0 - I0 - R0, E0, I0, R0)

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
    },
    "SI": {
        "init": init_si,
        "params": {
            "beta": 0.2
        },
        "model_func": si_model
    },
    "SIS": {
        "init": init_sis,
        "params": {
            "beta": 0.2, 
            "gamma": 1 / 10
        },
        "model_func": sis_model
    },
    "SEIR": {
        "init": init_seir,
        "params": {
            "beta": 0.2, 
            "gamma": 1 / 10, 
            "sigma": 1 / 5.2
        },
        "model_func": seir_model
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
        self.compartment_names = ['S', 'E', 'I', 'R', 'B']
        compartment_values = self.result.T
        self.compartment_dict = {}
        for i, val in enumerate(compartment_values):
            if i < len(self.compartment_names):
                self.compartment_dict[self.compartment_names[i]] = val
        self.S = self.compartment_dict['S']
        self.I = self.compartment_dict.get('I', None)
        self.R = self.compartment_dict.get('R', None)
        self.E = self.compartment_dict.get('E', None)

    def add_noise(self, noise_level=2):
        self.S_noisy = np.clip(self.S + np.random.normal(0, noise_level, size=self.S.shape), 0, self.N)
        self.I_noisy = np.clip(self.I + np.random.normal(0, noise_level, size=self.I.shape), 0, self.N)
        self.R_noisy = np.clip(self.R + np.random.normal(0, noise_level, size=self.R.shape), 0, self.N)
        if self.E is not None:
            self.E_noisy = np.clip(self.E + np.random.normal(0, noise_level, size=self.E.shape), 0, self.N)

    def sample_subset(self, num_points=80):
        idx = np.sort(np.random.choice(len(self.t), size=num_points, replace=False))
        self.t_subset = self.t[idx]
        self.I_subset = self.I_noisy[idx]
        self.S0_est = self.S_noisy[idx[0]]
        self.I0_est = self.I_noisy[idx[0]]
        self.R0_est = self.R_noisy[idx[0]]
        if self.E is not None:
            self.E0_est = self.E_noisy[idx[0]]

    def loss(self, params):
        if self.name in ["SEIR", "SEIRS"]:
            y0 = (self.S0_est, self.E0_est, self.I0_est, self.R0_est)
        else:
            y0 = (self.S0_est, self.I0_est, self.R0_est)
        sol = odeint(self.model_func, y0, self.t_subset, args=(self.N, *params))
        I_index = ['S', 'E', 'I', 'R', 'B'].index('I')
        return np.mean((sol[:, I_index] - self.I_subset) ** 2)

    def fit(self):
        guess = np.random.uniform(0, 1, size=len(self.params))
        bounds = [(0.0001, 1)] * len(self.params)
        res = minimize(self.loss, guess, method='L-BFGS-B', bounds=bounds)
        self.fitted_params = res.x

    def simulate_with_fit(self):
        if self.name in ["SEIR", "SEIRS"]:
            y0 = (self.S0_est, self.E0_est, self.I0_est, self.R0_est)
        else:
            y0 = (self.S0_est, self.I0_est, self.R0_est)
        sol = odeint(self.model_func, y0, self.t_subset, args=(self.N, *self.fitted_params))
        return sol.T
        
    def plot_trajectories(self):
        plt.figure()
        if self.S is not None:
            plt.plot(self.t, self.S, label='Susceptible', color='orange')
        if self.E is not None:
            plt.plot(self.t, self.E, label='Exposed', color='blue')
        if self.I is not None:
            plt.plot(self.t, self.I, label='Infected', color='red')
        if self.R is not None:
            plt.plot(self.t, self.R, label='Recovered', color='green')
        plt.title(f'{self.name} Model Dynamics')
        plt.xlabel('Time (days)')
        plt.ylabel('Population')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

    def plot_noisy_data(self):
        plt.figure()
        if hasattr(self, 'S_noisy'):
            plt.plot(self.t, self.S_noisy, label='Noisy Susceptible', alpha=1)
        if hasattr(self, 'E_noisy'):
            plt.plot(self.t, self.E_noisy, label='Noisy Exposed', alpha=1)
        if hasattr(self, 'I_noisy'):
            plt.plot(self.t, self.I_noisy, label='Noisy Infected', alpha=1)
        if hasattr(self, 'R_noisy'):
            plt.plot(self.t, self.R_noisy, label='Noisy Recovered', alpha=1)
        plt.title(f'{self.name} Noisy Data')
        plt.xlabel('Time (days)')
        plt.ylabel('Population')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

    def plot_loss_landscape(self):
        if len(self.params) != 2:
            print("Loss landscape only supported for models with 2 parameters.")
            return
        beta_vals = np.linspace(0.01, 1.0, 50)
        gamma_vals = np.linspace(0.01, 1.0, 50)
        loss_vals = np.zeros((len(beta_vals), len(gamma_vals)))

        for i, beta in enumerate(beta_vals):
            for j, gamma in enumerate(gamma_vals):
                loss_vals[i, j] = self.loss([beta, gamma])

        B, G = np.meshgrid(gamma_vals, beta_vals)  # Note: reversed order to match axes
        plt.figure()
        cp = plt.contourf(G, B, loss_vals, 50, cmap='viridis')
        plt.colorbar(cp)
        plt.xlabel('Gamma')
        plt.ylabel('Beta')
        plt.title('Loss Landscape')
        plt.tight_layout()
        plt.show()

    def plot_fitted_vs_noisy(self):
        result = self.simulate_with_fit()
        I_fit = result[['S', 'E', 'I', 'R', 'B'].index('I')]
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
