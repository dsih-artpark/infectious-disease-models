from scipy.integrate import odeint
import numpy as np

class Population:
    def __init__(self, total_population, compartments):
        self.total_population = total_population
        self.compartments = compartments
        self._validate_population()

    def _validate_population(self):
        total = sum(self.compartments.values())
        if total != self.total_population:
            raise ValueError(f"Mismatch in population: {total} != {self.total_population}")

    def update_compartment(self, name, value):
        if name not in self.compartments:
            raise ValueError(f"{name} not in compartments")
        self.compartments[name] = value

    def get_compartment(self, name):
        return self.compartments.get(name, 0)


class CompartmentalModel:
    def __init__(self, compartments, parameters, transitions, population=None):
        self.compartments = compartments
        self.parameters = parameters
        self.transitions = transitions
        self.population = population
        
    def compute_transition_rates(self, state, extras=None):
        local_env = {**state, **self.parameters, "N": self.population if self.population else max(sum(state.values()), 1e-8)}
        if extras:
            local_env.update(extras)
        local_env['N'] = self.population if self.population else max(sum(state.values()), 1e-8)
        deltas = {c: 0.0 for c in self.compartments}
        for tr in self.transitions:
            src = tr['from']
            dst = tr['to']
            rate = eval(tr['rate'], {}, local_env)
            if src:
                deltas[src] -= rate
            if dst:
                deltas[dst] += rate
        return deltas

    def ode_rhs(self, y, t, extras_fn=None):
        state = {c: y[i] for i, c in enumerate(self.compartments)}
        extras = extras_fn(t, y) if extras_fn else None
        deltas = self.compute_transition_rates(state, extras)
        return [deltas[c] for c in self.compartments]

    def simulate(self, initial_conditions, time_points, extras_fn=None):
        y0 = [initial_conditions[c] for c in self.compartments]
        sol = odeint(self.ode_rhs, y0, time_points, args=(extras_fn,))
        return sol

    def add_noise(self, data, noise_std, seed=42):
        np.random.seed(seed)
        return data + np.random.normal(0, noise_std, data.shape)