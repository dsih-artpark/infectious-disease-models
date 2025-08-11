# Noise Injection

To simulate measurement noise, Gaussian noise is added to the clean trajectories:

\[
y_{\text{noisy}}(t) = y_{\text{clean}}(t) + \epsilon(t),\quad \epsilon(t)\sim \mathcal{N}(0, \sigma^2)
\]

- `sigma` is configured via `noise_std` in the YAML.
- After adding noise, values are clipped to valid ranges (e.g., non-negative, ≤ population).
- Optionally round to integers if modeling counts.

### Example

```python
import numpy as np

def add_noise(traj, sigma, clip_min=0, clip_max=None, round_int=False):
    noisy = traj + np.random.normal(0, sigma, traj.shape)
    if clip_max is not None:
        noisy = np.clip(noisy, clip_min, clip_max)
    else:
        noisy = np.clip(noisy, clip_min, None)
    if round_int:
        noisy = np.rint(noisy).astype(int)
    return noisy
```