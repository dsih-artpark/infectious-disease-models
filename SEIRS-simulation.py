from scipy.integrate import odeint
import numpy as np

def seirsb_model(y, t, N, beta, sigma, gamma, omega):
    S, E, I, R = y
    dSdt = -beta * S * I / N + omega * R
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I - omega * R
    return dSdt, dEdt, dIdt, dRdt

N = 1000000
E0, I0, R0 = 10, 5, 0
S0 = N - E0 - I0 - R0
y0 = S0, E0, I0, R0

t = np.linspace(0, 160, 160)
beta, sigma, gamma, omega = 0.3, 0.2, 0.1, 0.01

solution = odeint(seirsb_model, y0, t, args=(N, beta, sigma, gamma, omega))
S, E, I, R = solution.T

