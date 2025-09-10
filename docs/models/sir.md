# SIR Model

The **SIR model** is one of the simplest compartmental models in epidemiology.  
It divides the population into three compartments:  

- **S (Susceptible):** Individuals who are healthy but can contract the disease.  
- **I (Infected):** Individuals currently infected and able to spread the disease.  
- **R (Recovered):** Individuals who have recovered and gained immunity.  

---

## Transition Diagram

```mermaid
flowchart LR
    S -- "β * S * I / N" --> I
    I -- "γ * I" --> R
    N -- "μ * N" --> S
    S -- "μ * S" --> ∅
    I -- "μ * I" --> ∅
    R -- "μ * R" --> ∅

```