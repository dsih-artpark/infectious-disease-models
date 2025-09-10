# TB Model

## Transition Diagram

```mermaid
flowchart LR
    S -- "βs * Is * S / N" --> Ls
    S -- "βr * Ir * S / N" --> Lr
    Ls -- "αs * Ls" --> Is
    Lr -- "αr * Lr" --> Ir
    Is -- "ωs * Is" --> R
    Is -- "(1 - ρ) * τs * Is" --> R
    Is -- "ρ * τs * Is" --> Ir
    Ir -- "ωr * Ir" --> R
    Ir -- "τr * Ir" --> R
    Ir -- "φr * Ir" --> S
    Is -- "φs * Is" --> S
    R -- "γ * R" --> S
    %% Natural mortality and births
    N -- "μ * N" --> S
    S -- "μ * S" --> ∅
    Ls -- "μ * Ls" --> ∅
    Lr -- "μ * Lr" --> ∅
    Is -- "μ * Is" --> ∅
    Ir -- "μ * Ir" --> ∅
    R -- "μ * R" --> ∅


TB_model:
  plot_settings:
    scale_by_population: true   
    per_unit: 100000           
    time_unit: years   

  compartments: [S, Ls, Is, Lr, Ir, R]

  parameters: 
    mu: 0.0142857
    beta_s: 1.57e-8
    beta_r: 6.25e-9
    alpha_s: 0.129
    alpha_r: 0.129
    omega_s: 0.287
    omega_r: 0.12
    rho: 0.07
    phi_s: 0.37
    phi_r: 0.37
    tau_s: 0.94
    tau_r: 0.78
    gamma: 0.1

  transitions: 
    "R -> S": "gamma * R"
    "Is -> S": "phi_s * Is"
    "Ir -> S": "phi_r * Ir"
    "S -> Ls": "beta_s * Is * S / N"
    "S -> Lr": "beta_r * Ir * S / N"
    "S ->": "mu * S"
    "-> S": "mu * N"
    "Ls -> Is": "alpha_s * Ls"
    "Ls ->": "mu * Ls"
    "Is -> R": "omega_s * Is"
    "Is -> R_extra": "(1 - rho) * tau_s * Is"
    "Is ->": "mu * Is"
    "Is -> Ir": "rho * tau_s * Is"
    "Lr -> Ir": "alpha_r * Lr"
    "Lr ->": "mu * Lr"
    "Ir -> R": "omega_r * Ir"
    "Ir -> R_extra": "tau_r * Ir"
    "Ir ->": "mu * Ir"
    "Ir -> S": "phi_r * Ir"
    "R ->": "mu * R"

  population: 159000000

  initial_conditions: 
    Ls: 636000
    Is: 477000
    Lr: 15900
    Ir: 636000
    R: 7950
    S: 157242150

  assumptions: |
    TB transmission with drug-sensitive (DS) and drug-resistant (DR) strains,
    based on Kuddus (2022).
