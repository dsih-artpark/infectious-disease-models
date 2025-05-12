import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize

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
plt.show()
plt.figure()
plt.grid()
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
plt.show()

subset_days = 80
t_subset = t[:subset_days]
S_subset = S_noisy[:subset_days]
I_subset = I_noisy[:subset_days]
R_subset = R_noisy[:subset_days]
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


#loss landscape

beta_vals = np.linspace(0.15, 0.25, 100)
gamma_vals = np.linspace(0.05, 0.15, 100)
loss_grid = np.zeros((len(beta_vals), len(gamma_vals)))
for i, b in enumerate(beta_vals):
    for j, g in enumerate(gamma_vals):
        loss_grid[i, j] = loss([b, g])
B, G = np.meshgrid(gamma_vals, beta_vals)
result_nm = minimize(loss, initial_guess, bounds=bounds)  
beta_nm, gamma_nm = result_nm.x
result_bfgs = minimize(loss, initial_guess, method='BFGS')
beta_bfgs, gamma_bfgs = result_bfgs.x
result_lbfgs = minimize(loss, initial_guess, method='L-BFGS-B', bounds=bounds)
beta_lbfgs, gamma_lbfgs = result_lbfgs.x
plt.figure(figsize=(10, 7))
cp = plt.contourf(G, B, loss_grid, levels=50, cmap='viridis')
plt.colorbar(cp, label='Loss (MSE)')
plt.xlabel(r'$\gamma$')
plt.ylabel(r'$\beta$')
plt.title('Loss Landscape for SIR Parameter Estimation')
plt.scatter([gamma_nm], [beta_nm], color='red', label='Nelder-Mead')
plt.scatter([gamma_bfgs], [beta_bfgs], color='blue', label='BFGS')
plt.scatter([gamma_lbfgs], [beta_lbfgs], color='white', label='L-BFGS-B')
plt.scatter([gamma], [beta], color='black', marker='x', s=100, label='True Value')
plt.legend()
plt.grid(True)
plt.show()


def simulate_sir_components(beta, gamma):
    y0_est = S0_est, I0_est, R0_est
    sol = odeint(func, y0_est, t_subset, args=(N, beta, gamma))
    return sol.T  
def plot_fits(var_index, var_label, var_color):
    plt.figure(figsize=(10, 6))
    if var_index == 0:
        plt.plot(t_subset, S_subset, 'k--', label='Noisy ' + var_label, linewidth=2)
    elif var_index == 1:
        plt.plot(t_subset, I_subset, 'k--', label='Noisy ' + var_label, linewidth=2)
    else:
        plt.plot(t_subset, R_subset, 'k--', label='Noisy ' + var_label, linewidth=2)
    for method, (beta_est, gamma_est), color in zip(
        ['Nelder-Mead', 'BFGS', 'L-BFGS-B'],
        [(beta_nm, gamma_nm), (beta_bfgs, gamma_bfgs), (beta_lbfgs, gamma_lbfgs)],
        ['red', 'blue', 'green']
    ):
        S_sim, I_sim, R_sim = simulate_sir_components(beta_est, gamma_est)
        sim_data = [S_sim, I_sim, R_sim][var_index]
        plt.plot(t_subset, sim_data, label=f'{method} Fit', color=color, linewidth=2)
    plt.xlabel('Time (days)')
    plt.ylabel(f'{var_label} individuals')
    plt.title(f'Fitted {var_label}(t) from Different Optimization Methods')
    plt.ylim([0, N])
    plt.legend()
    plt.grid(True)
    plt.show()
plot_fits(0, 'Susceptible', 'orange')
plot_fits(1, 'Infected', 'red')
plot_fits(2, 'Recovered', 'green')
