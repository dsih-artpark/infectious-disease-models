import numpy as np
from epimodels.model import Population, CompartmentalModel
from epimodels.calibration import Calibrator

def setup_model():
    pop = Population(size=1000)
    model = CompartmentalModel(
        compartments=["S", "I", "R"],
        parameters={"beta": 0.3, "gamma": 0.1},
        transitions={
            "S->I": "beta * S * I / pop.size",  # use population size
            "I->R": "gamma * I"
        },
        population=pop,
        initial_conditions={"S": 999, "I": 1, "R": 0}
    )
    return model

def test_loss_function_runs():
    model = setup_model()
    calib = Calibrator(model, ["beta", "gamma"], [1])  # focus on "I"
    t = np.linspace(0, 10, 11)
    y = model.simulate([999, 1, 0], t)
    loss = calib.loss_function([0.3, 0.1], [999, 1, 0], t, y[:, 1], [1])
    assert isinstance(loss, float)

def test_fit_with_optimizer():
    model = setup_model()
    calib = Calibrator(model, ["beta", "gamma"], [1])
    t = np.linspace(0, 5, 6)
    y = model.simulate([999, 1, 0], t)
    results = calib.fit(
        [999, 1, 0], t, t, y[:, 1],
        optimizers=["Nelder-Mead"],
        compartments=[1]
    )
    assert "Nelder-Mead" in results
    assert "params" in results["Nelder-Mead"]
