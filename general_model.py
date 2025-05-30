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
        self.S = self.E = self.I = self.R = None

    def simulate(self):
        self.result = odeint(self.model_func, self.y0, self.t, args=(self.N, *self.params))
        self.compartment_names = [chr(65 + i) for i in range(self.result.shape[1])]  # A, B, C... fallback
        known_compartments = {'S', 'E', 'I', 'R'}
        inferred_names = [name for name in self.model_func.__code__.co_varnames if name in known_compartments]
        if len(inferred_names) == self.result.shape[1]:
            self.compartment_names = inferred_names
        self.compartment_dict = {name: self.result[:, i] for i, name in enumerate(self.compartment_names)}
        for name in self.compartment_names:
            setattr(self, name, self.compartment_dict[name])


    def add_noise(self, noise_level=2):
        if self.S is not None:
            self.S_noisy = np.clip(self.S + np.random.normal(0, noise_level, size=self.S.shape), 0, self.N)
        if self.I is not None:
            self.I_noisy = np.clip(self.I + np.random.normal(0, noise_level, size=self.I.shape), 0, self.N)
        if self.R is not None:
            self.R_noisy = np.clip(self.R + np.random.normal(0, noise_level, size=self.R.shape), 0, self.N)
        if self.E is not None:
            self.E_noisy = np.clip(self.E + np.random.normal(0, noise_level, size=self.E.shape), 0, self.N)
    
    def sample_subset(self, num_points=80):
        idx = np.sort(np.random.choice(len(self.t), size=num_points, replace=False))
        self.t_subset = self.t[idx]
        for name in self.compartment_names:
            noisy_attr = f"{name}_noisy"
            if hasattr(self, noisy_attr):
                setattr(self, f"{name}0_est", getattr(self, noisy_attr)[idx[0]])
                if name == 'I':
                    self.I_subset = getattr(self, noisy_attr)[idx]

    def loss(self, params):
        y0 = tuple(getattr(self, f"{name}0_est") for name in self.compartment_names)
        sol = odeint(self.model_func, y0, self.t_subset, args=(self.N, *params))
        I_index = self.compartment_names.index('I')
        return np.mean((sol[:, I_index] - self.I_subset) ** 2)

    def fit(self):
        guess = np.random.uniform(0, 1, size=len(self.params))
        bounds = [(0.0001, 1)] * len(self.params)
        res = minimize(self.loss, guess, method='L-BFGS-B', bounds=bounds)
        self.fitted_params = res.x

    def simulate_with_fit(self):
        y0 = tuple(getattr(self, f"{name}0_est") for name in self.compartment_names)
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
        log_loss_grid = np.log10(loss_vals + 1e-10)
        for i, beta in enumerate(beta_vals):
            for j, gamma in enumerate(gamma_vals):
                loss_vals[i, j] = self.loss([beta, gamma])
        G, B = np.meshgrid(gamma_vals, beta_vals)  # Note: reversed order to match axes
        initial_guess_nm = np.random.uniform(0, 1, size=2)
        initial_guess_bfgs = np.random.uniform(0, 1, size=2)
        initial_guess_lbfgs = np.random.uniform(0, 1, size=2)
        bounds = [(0, 1), (0, 1)]
        result_nm = minimize(lambda x: self.loss(x), initial_guess_nm, bounds=bounds)
        beta_nm, gamma_nm = result_nm.x
        result_bfgs = minimize(lambda x: self.loss(x), initial_guess_bfgs, method='BFGS')
        beta_bfgs, gamma_bfgs = result_bfgs.x
        result_lbfgs = minimize(lambda x: self.loss(x), initial_guess_lbfgs, method='L-BFGS-B', bounds=bounds)
        beta_lbfgs, gamma_lbfgs = result_lbfgs.x
        plt.figure(figsize=(10, 8))
        cp = plt.contourf(G, B, log_loss_grid, levels=100, cmap='viridis')
        cbar = plt.colorbar(cp)
        cbar.set_label(r'$\log_{10}$(Loss)')
        plt.xlabel(r'$\gamma$')
        plt.ylabel(r'$\beta$')
        plt.title('Log-Scaled Loss Landscape for SIR Parameter Estimation')
        plt.scatter([gamma_nm], [beta_nm], color='red', marker='s', label='Nelder-Mead')
        plt.scatter([gamma_bfgs], [beta_bfgs], color='blue', marker='o', label='BFGS')
        plt.scatter([gamma_lbfgs], [beta_lbfgs], color='white', marker='^', label='L-BFGS-B')
        true_beta, true_gamma = self.params
        plt.scatter([true_gamma], [true_beta], color='black', marker='x', s=100, label='True Value')

        plt.legend()
        plt.grid(True)
        plt.axis('scaled')
        plt.tight_layout()
        plt.savefig("plots/loss_landscape.png")
        plt.show()

    def plot_fitted_vs_noisy(self):
        result = self.simulate_with_fit()
        I_index = self.compartment_names.index('I')
        I_fit = result[I_index]
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
    run_epidemic_model('SIR')
