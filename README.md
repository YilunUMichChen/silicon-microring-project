# Microring Compact Model Extension

This folder is a clean extension of the original microring course project. It
does not modify the original report, Lumerical files, figures, or `MMR.py`.

## Step 1 Goal

Build a lightweight Python compact model for a silicon microring resonator that
is more unit-consistent and easier to extend than the original plotting script.

Current scope:

- all-pass microring through-port transfer function
- unit-consistent round-trip length
- dB/cm propagation loss converted to field amplitude loss
- separate `n_eff` and `n_g` using a first-order dispersion model around
  `lambda_ref_nm`
- analytic FSR estimate
- automatic resonance dip detection
- extracted FWHM, loaded Q, and extinction ratio
- first coupling-coefficient sweep

## Files

- `src/microring_model.py`: reusable compact-model functions
- `scripts/run_step1_demo.py`: demo script that prints metrics and writes plots
- `scripts/run_toolchain_example.py`: example using SciPy, pandas, and xarray
- `scripts/run_jax_example.py`: optional differentiable compact-model example
- `scripts/run_meep_minimal_waveguide.py`: optional Meep FDTD smoke test
- `scripts/run_meep_microring_2d.py`: optional 2D effective-index microring FDTD example
- `scripts/run_meep_add_drop_microring_2d.py`: optional add-drop microring FDTD example
- `scripts/run_meep_racetrack_add_drop_2d.py`: optional racetrack add-drop FDTD example
- `scripts/optimize_add_drop_compact.py`: SciPy compact-model optimizer for add-drop rings
- `scripts/optimize_racetrack_high_q_compact.py`: high-Q racetrack add-drop compact optimizer
- `scripts/run_meep_add_drop_param_sweep.py`: small Meep FDTD validation sweep
- `OPTIONAL_TOOLING.md`: setup notes for JAX and Meep
- `outputs/`: generated figures

## Run

From this folder:

```bash
/opt/anaconda3/bin/python3 scripts/run_step1_demo.py
```

On this machine, the parent project currently resolves `python3` to Anaconda,
while this new subfolder may resolve it to Homebrew Python. The explicit
Anaconda path avoids missing-package errors for `numpy` and `matplotlib`.

Expected generated figures:

- `outputs/step1_all_pass_response.png`
- `outputs/step1_coupling_sweep.png`

## Scientific / Industry-Style Toolchain Example

The core model stays lightweight, but the workflow can use standard scientific
Python tools:

- SciPy: resonance detection and Lorentzian fitting
- pandas: design-sweep table exported to CSV
- xarray: labeled multi-dimensional sweep data exported to NetCDF

Run:

```bash
/opt/anaconda3/bin/python3 scripts/run_toolchain_example.py
```

Expected generated artifacts:

- `outputs/toolchain_design_sweep.csv`
- `outputs/toolchain_design_sweep.nc`
- `outputs/toolchain_design_maps.png`

This is the bridge toward more photonics-specific tools. Later, `gdsfactory` can
generate ring/racetrack layout parameters, `SAX` can run S-parameter circuit
simulations, and FDTD/MODE tools such as Lumerical or Meep can calibrate
`n_eff`, `n_g`, loss, and coupling coefficients.

## Notes for Later Steps

This first model is still a compact analytical model, not FDTD. A practical next
step is to compare its qualitative trends with the existing Lumerical figures
and then add a semi-empirical coupling model such as:

```text
kappa(gap) = kappa0 * exp(-gap / g0)
```

After that, the active microring modulator extension can reuse the same model by
adding a resonance shift:

```text
delta_lambda_res = lambda_res * delta_n_eff / n_g
```
