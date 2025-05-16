import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize
np.random.seed(42)
N = 1000
I0 = 1
R0 = 0
S0 = N - I0 - R0
beta = 0.2
gamma = 1 / 10
t = np.linspace(0, 500, 500)
def func(y, t, N, beta, gamma):
  S, I, R = y
  dSdt = -beta * S * I / N
  dIdt = beta * S * I / N - gamma * I
  dRdt = gamma * I
  return dSdt, dIdt, dRdt
y0 = S0, I0, R0
res = odeint(func, y0, t, args=(N, beta, gamma))
S, I, R = res.T
plt.figure()
plt.grid()
plt.plot(t, S, 'orange', label='Susceptible')
plt.plot(t, I, 'red', label='Infected')
plt.plot(t, R, 'green', label='Recovered')
plt.xlabel('Time (days)')
plt.ylabel('Number of individuals')
plt.ylim([0,N])
plt.title('SIR Model')
plt.legend()
plt.savefig("plots/SIRmodel.png")
plt.show()
#noise
noise_level = 2
S_noisy = S + np.random.normal(0, noise_level, size=S.shape)
I_noisy = I + np.random.normal(0, noise_level, size=I.shape)
R_noisy = R + np.random.normal(0, noise_level, size=R.shape)
S_noisy = np.clip(S_noisy, 0, N)
I_noisy = np.clip(I_noisy, 0, N)
R_noisy = np.clip(R_noisy, 0, N)
plt.plot(t, S_noisy, 'orange', label='Susceptible (noisy)')
plt.plot(t, I_noisy, 'red', label='Infected (noisy)')
plt.plot(t, R_noisy, 'green', label='Recovered (noisy)')
plt.xlabel('Time (days)')
plt.ylabel('Number of individuals')
plt.ylim([0,N])
plt.title('SIR Model with noise')
plt.legend()
plt.savefig("plots/SIR_noise_model.png")
plt.show()

subset_days = 80
idx = np.random.choice(len(t), size=subset_days, replace=False)
index = np.sort(idx) 
t_subset = t[index]
S_subset = S_noisy[index]
I_subset = I_noisy[index]
R_subset = R_noisy[index]
S0_est = S_subset[0]
I0_est = I_subset[0]
R0_est = R_subset[0]
def loss(params):
  beta, gamma = params
  y0_est = S0_est, I0_est, R0_est
  sol = odeint(func, y0_est, t_subset, args=(N, beta, gamma))
  S_sim, I_sim, R_sim = sol.T
  return np.mean((I_sim - I_subset)**2)
initial_guess = [0.1, 0.1]
bounds = [(0.0001, 1), (0.0001, 1)]
result_nm = minimize(loss, initial_guess, method='Nelder-Mead', bounds=bounds)
beta_nm, gamma_nm = result_nm.x
step_size_nm = np.array(result_nm.x) - np.array(initial_guess) 
print(f"[Nelder-Mead] Start point: {initial_guess}, Estimated β = {beta_nm:.8f}, γ = {gamma_nm:.8f}, Step size: {step_size_nm}, Iterations = {result_nm.nit}")

result_bfgs = minimize(loss, initial_guess, method='BFGS')  
beta_bfgs, gamma_bfgs = result_bfgs.x
step_size_bfgs = np.array(result_bfgs.x) - np.array(initial_guess)  
print(f"[BFGS] Start point: {initial_guess}, Estimated β = {beta_bfgs:.8f}, γ = {gamma_bfgs:.8f}, Step size: {step_size_bfgs}, Iterations = {result_bfgs.nit}")

result_lbfgs = minimize(loss, initial_guess, method='L-BFGS-B', bounds=bounds)
beta_lbfgs, gamma_lbfgs = result_lbfgs.x
step_size_lbfgs = np.array(result_lbfgs.x) - np.array(initial_guess)  
print(f"[L-BFGS-B] Start point: {initial_guess}, Estimated β = {beta_lbfgs:.8f}, γ = {gamma_lbfgs:.8f}, Step size: {step_size_lbfgs}, Iterations = {result_lbfgs.nit}")

# loss landscape
beta_vals = np.linspace(0.0001, 0.3, 200)
gamma_vals = np.linspace(0.0001, 0.18, 200)
loss_grid = np.zeros((len(beta_vals), len(gamma_vals)))
for i, b in enumerate(beta_vals):
    for j, g in enumerate(gamma_vals):
        loss_grid[i, j] = loss([b, g])
log_loss_grid = np.log10(loss_grid + 1e-10)
G, B = np.meshgrid(gamma_vals, beta_vals)
initial_guess = [0.1, 0.1]
bounds = [(0, 1), (0, 1)]
result_nm = minimize(loss, initial_guess, bounds=bounds)
beta_nm, gamma_nm = result_nm.x
result_bfgs = minimize(loss, initial_guess, method='BFGS')
beta_bfgs, gamma_bfgs = result_bfgs.x
result_lbfgs = minimize(loss, initial_guess, method='L-BFGS-B', bounds=bounds)
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
plt.scatter([gamma], [beta], color='black', marker='x', s=100, label='True Value')
plt.legend()
plt.grid(True)
plt.axis('scaled')
plt.tight_layout()
plt.savefig("plots/loss_landscape.png")
plt.show()
print(f"Loss at true parameters: {loss([beta, gamma]):.2f}")
print(f"Loss at L-BFGS-B fit:    {loss([beta_lbfgs, gamma_lbfgs]):.2f}")


def simulate_sir_components(beta, gamma):
    y0_est = S0_est, I0_est, R0_est
    sol = odeint(func, y0_est, t_subset, args=(N, beta, gamma))
    return sol.T  
def plot_fits():
    plt.figure(figsize=(10, 6))
    plt.plot(t, I, 'k-', label='True Infected', linewidth=2)
    S_sim, I_sim, R_sim = simulate_sir_components(beta_lbfgs, gamma_lbfgs)
    plt.plot(t_subset, I_sim, label='L-BFGS-B Fit', color='green', linewidth=2)
    plt.scatter(t_subset, I_subset, label='Noisy Infected', color='red', marker='x')
    plt.xlabel('Time (days)')
    plt.ylabel('Infected Individuals')
    plt.title('Fitted Infected Phase with L-BFGS-B and Noisy Data')
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/fitted_data.png")
    plt.show()
plot_fits()


S_true_subset, I_true_subset, R_true_subset = odeint(func, (S0, I0, R0), t_subset, args=(N, beta, gamma)).T
loss_true_vs_noisy = np.mean((I_true_subset - I_subset) ** 2)
S_fit_lbfgs, I_fit_lbfgs, R_fit_lbfgs = simulate_sir_components(beta_lbfgs, gamma_lbfgs)
loss_fit_vs_noisy = np.mean((I_fit_lbfgs - I_subset) ** 2)
print(f"Loss (True vs Noisy): {loss_true_vs_noisy:.2f}")
print(f"Loss (L-BFGS-B Fit vs Noisy): {loss_fit_vs_noisy:.2f}")
print(len(I_fit_lbfgs))
print(len(I_subset))

