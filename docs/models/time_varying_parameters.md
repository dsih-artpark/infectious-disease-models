# Time-Varying Parameters

In real-world epidemic modeling, parameters such as the infection rate (β) or recovery rate (γ) rarely stay constant.  
They can change over time due to interventions, behavioral changes, vaccination drives, or environmental factors.

Our framework supports **time-varying parameters** through a flexible design that lets you:

- Define parameter schedules in YAML configuration  
- Generate parameter time series automatically  
- Integrate them directly into simulations and calibration workflows  

---

## Defining Time-Varying Parameters

You can define how parameters evolve over time directly in your model configuration file (`config.yml`) under the `time_varying` section.

### Example — `SIR_model` Configuration

```yaml
SIR_model:
  compartments: [S, I, R]
  parameters:
    beta: 0.3
    gamma: 0.1

  time_varying:
    beta:
      default: 0.3
      schedule:
        - {t: 5, value: 0.1}
        - {t: 10, value: 0.05}

  transitions:
    S -> I: beta * S * I / N
    I -> R: gamma * I

  initial_conditions:
    S: 990
    I: 10
    R: 0

  simulation_time: 30
