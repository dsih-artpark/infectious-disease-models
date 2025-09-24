import numpy as np
import pytest
from src.epimodels.model import Population, CompartmentalModel

def test_population_init_valid():
    pop = Population(N=1000)
    assert pop.N == 1000

def test_population_init_invalid():
    with pytest.raises(ValueError):
        Population(N=-10)

def test_compartmental_model_simulate():
    pop = Population(N=1000)
    model = CompartmentalModel(
        compartments=["S", "I", "R"],
        parameters={"beta": 0.3, "gamma": 0.1},
        transitions={"S->I": "beta * S * I / N", "I->R": "gamma * I"},
        population=pop,
        initial_conditions={"S": 999, "I": 1, "R": 0}
    )
    t = np.linspace(0, 10, 11)
    y = model.simulate([999, 1, 0], t)
    assert y.shape == (11, 3)
    assert np.all(y >= 0)
    assert np.isclose(y.sum(axis=1)[0], 1000)  # population conserved
