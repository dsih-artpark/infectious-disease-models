# Improvements

- Relaxed prior bounds to encourage better exploration.
- Smarter walker initialization (spread inside prior region).
- Added checks to skip parameter sets that produce unstable ODE solutions.
- Improved plotting routines: explicit `savefig()` and `plt.close()` to prevent file locking.
- Exposed hooks to allow custom loss functions and time-varying parameters.