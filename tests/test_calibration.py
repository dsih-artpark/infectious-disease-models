import numpy as np
from epimodels.model import Population, CompartmentalModel
from epimodels.calibration import Calibrator

def setup_model():
    init_conditions = {"S": 999, "I": 1, "R": 0}
    pop = Population(1000, init_conditions)
    model = CompartmentalModel(
        compartments=["S", "I", "R"],
        parameters={"beta": 0.3, "gamma": 0.1},
        transitions={"S->I": "beta * S * I / N", "I->R": "gamma * I"},
        population=pop
    )
    return model, init_conditions



def test_loss_function_runs():
    model, init_conditions = setup_model()
    calib = Calibrator(model, ["beta", "gamma"], [1])  # focus on "I"
    t = np.linspace(0, 10, 11)
    y = model.simulate([init_conditions["S"], init_conditions["I"], init_conditions["R"]], t)
    loss = calib.loss_function([0.3, 0.1],
                               [init_conditions["S"], init_conditions["I"], init_conditions["R"]],
                               t, y[:, 1], [1])
    assert isinstance(loss, float)

def test_fit_with_optimizer():
    model, init_conditions = setup_model()
    calib = Calibrator(model, ["beta", "gamma"], [1])
    t = np.linspace(0, 5, 6)
    y = model.simulate([init_conditions["S"], init_conditions["I"], init_conditions["R"]], t)
    results = calib.fit(
        [init_conditions["S"], init_conditions["I"], init_conditions["R"]],
        t, t, y[:, 1],
        optimizers=["Nelder-Mead"],
        compartments=[1]
    )
    assert "Nelder-Mead" in results
    assert "params" in results["Nelder-Mead"]
