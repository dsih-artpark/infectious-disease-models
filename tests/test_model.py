import numpy as np
import pytest
from epimodels.model import Population, CompartmentalModel

def test_population_init_valid():
    compartments = {"S": 999, "I": 1, "R": 0}
    pop = Population(total_population=1000, compartments=compartments)
    assert pop.total_population == 1000
    assert pop.get_compartment("S") == 999
    assert pop.get_compartment("I") == 1
    assert pop.get_compartment("R") == 0

def test_population_init_invalid():
    # total does not match sum of compartments
    compartments = {"S": 500, "I": 400, "R": 50}
    with pytest.raises(ValueError):
        Population(total_population=1000, compartments=compartments)

def test_compartmental_model_simulate():
    init_conditions = {"S": 999, "I": 1, "R": 0}
    pop = Population(total_population=1000, compartments=init_conditions.copy())
    model = CompartmentalModel(
        compartments=["S", "I", "R"],
        parameters={"beta": 0.3, "gamma": 0.1},
        transitions=[
            {"from": "S", "to": "I", "rate": "beta * S * I / N"},
            {"from": "I", "to": "R", "rate": "gamma * I"}
        ],
        population=pop
    )
    t = np.linspace(0, 10, 11)
    y = model.simulate(init_conditions, t)

    # Shape matches timepoints × compartments
    assert y.shape == (11, 3)
    # No negative populations
    assert np.all(y >= 0)
    # Population conserved at t=0
    assert np.isclose(y.sum(axis=1)[0], 1000)
