# Advanced Features

- **Time-dependent parameters** via `extras_fn(t, y)`.
- **Selective observation model**: fit only some compartments or use different noise models per compartment.
- **Checkpointing**: save intermediate parameter states during long runs.
- **Loss landscape visualization**: grid-evaluate loss over parameter pairs and contour plot in log-scale.
- **Posterior diagnostics**: corner plots (e.g., using `corner`), trace plots, autocorrelation.