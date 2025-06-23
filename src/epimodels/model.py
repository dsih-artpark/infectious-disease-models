import numpy as np

class Population:
    def __init__(self, patch_population, compartments, patch_id=None):
        self.patch_population = patch_population
        self.compartments = compartments
        self.patch_id = patch_id
        self._validate_population()

    def _validate_population(self):
        total = sum(self.compartments.values())
        if total != self.patch_population:
            raise ValueError(f"Mismatch in population for patch {self.patch_id}: {total} != {self.patch_population}")

    def update_compartment(self, name, value):
        if name not in self.compartments:
            raise ValueError(f"{name} not in compartments")
        self.compartments[name] = value

    def get_compartment(self, name):
        return self.compartments.get(name, 0)


class CompartmentalModel:
    def __init__(self, compartments, parameters, transitions):
        self.compartments = compartments
        self.parameters = parameters
        self.transitions = transitions

    def compute_transition_rates(self, state, extras=None):
        local_env = {**state, **self.parameters}
        if extras:
            local_env.update(extras)
        local_env['N'] = sum(state.values())
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


class NetworkModel:
    def __init__(self, base_model, num_patches, network_matrix):
        self.base_model = base_model
        self.num_patches = num_patches
        self.network = np.array(network_matrix)
        self.all_compartments = [
            f"{c}_{i}" for i in range(num_patches) for c in base_model.compartments
        ]

    def compute_force_of_infection(self, full_state):
        lambdas = []
        for i in range(self.num_patches):
            force = 0
            for j in range(self.num_patches):
                I_j = full_state.get(f"I_{j}", 0)
                N_j = sum(full_state.get(f"{c}_{j}", 0) for c in self.base_model.compartments)
                if N_j > 0:
                    force += self.network[i][j] * I_j / N_j
            lambdas.append(force)
        return lambdas

    def simulate_discrete(self, y0_dict, t_range):
        state = y0_dict.copy()
        history = {k: [v] for k, v in state.items()}
        for t in t_range[1:]:
            new_state = state.copy()
            lambdas = self.compute_force_of_infection(state)
            for i in range(self.num_patches):
                patch_state = {c: state[f"{c}_{i}"] for c in self.base_model.compartments}
                extras = {"lambda_i": lambdas[i]}
                deltas = self.base_model.compute_transition_rates(patch_state, extras)
                for c, delta in deltas.items():
                    key = f"{c}_{i}"
                    new_state[key] += delta
            state = {k: max(v, 0) for k, v in new_state.items()}
            for k in state:
                history[k].append(state[k])
        return t_range, history
