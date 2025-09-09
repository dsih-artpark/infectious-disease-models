# SIR Model

The **SIR model** is a standard epidemiological model with three compartments:
- **S**: Susceptible
- **I**: Infected
- **R**: Recovered

---

## Transition Diagram

```mermaid
flowchart LR
    S -->|β S I / N| I
    I -->|γ I| R
