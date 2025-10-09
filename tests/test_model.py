import numpy as np
import pytest
from epimodels.model import Population, CompartmentalModel

def test_population_init_valid():
    init_conditions = {"S": 999, "I": 1, "R": 0}
    pop = Population(1000, init_conditions)
    assert pop.N == 1000

def test_population_init_invalid():
    init_conditions = {"S": 0, "I": 0, "R": 0}
    with pytest.raises(ValueError):
        Population(-10, init_conditions)

def test_compartmental_model_simulate():
    init_conditions = {"S": 999, "I": 1, "R": 0}
    pop = Population(1000, init_conditions)
    model = CompartmentalModel(
        compartments=["S", "I", "R"],
        parameters={"beta": 0.3, "gamma": 0.1},
        transitions=[
            {"from": "S", "to": "I", "rate": "beta * S * I / N"},  # N will be sum of state values
            {"from": "I", "to": "R", "rate": "gamma * I"}
        ],
        population=pop
    )
    t = np.linspace(0, 10, 11)
    y = model.simulate(init_conditions, t)  # pass dict

    # Shape matches timepoints × compartments
    assert y.shape == (11, 3)
    # No negative populations
    assert np.all(y >= 0)
    # Population conserved at t=0
    assert np.isclose(y.sum(axis=1)[0], 1000)
