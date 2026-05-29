# Microring Compact Model Extension

This repository extends an undergraduate Lumerical/Python microring resonator
project into a small silicon photonics modeling workflow. It combines analytical
compact models, scientific Python data analysis, differentiable optimization,
and open-source FDTD simulations to study microring and racetrack resonators for
optical interconnect-oriented applications.

The project is intentionally staged: fast compact models are used for design
space exploration, while Meep FDTD examples provide field maps and transmission
spectra that resemble a lightweight open-source version of a Lumerical workflow.
The current examples cover all-pass rings, add-drop rings, racetrack add-drop
resonators, coupling sweeps, high-Q compact optimization, and optional JAX-based
differentiable design.

## Highlights

- Unit-consistent silicon microring compact model with separated `n_eff` and
  `n_g`
- Automatic extraction of resonance wavelength, FSR, loaded Q, extinction ratio,
  and phase response
- SciPy/pandas/xarray workflow for design sweeps and fitted resonance metrics
- JAX example demonstrating gradient-based optimization through a microring
  transfer function
- Meep 2D effective-index FDTD examples for straight waveguides, single-bus
  rings, add-drop rings, and racetrack add-drop resonators
- Compact-model optimizers for add-drop and high-Q racetrack designs
- Research notes on curvature engineering, thermal effects, and future 3D FDTD
  extensions

## Example Results

The racetrack add-drop FDTD example demonstrates strong port transfer after
adding straight coupling sections:

```text
Through minimum = 0.0086 at 1.6313 um
Drop maximum    = 1.0099 at 1.6313 um
```

The high-Q racetrack compact optimizer finds a candidate design with:

```text
Loaded Q        ~= 8.0e4
Drop peak       ~= 0.785
Through at peak ~= 0.013
FSR             ~= 4.87 nm
```

These results are exploratory rather than foundry-calibrated. The goal is to
build a transparent modeling and optimization workflow that can later be
calibrated against Lumerical MODE/FDTD, measured data, or a process design kit.

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
- `scripts/compact/`: compact-model demos and scientific Python workflow examples
- `scripts/fdtd/`: Meep FDTD examples and validation sweeps
- `scripts/optimization/`: SciPy compact-model optimizers
- `scripts/jax/`: optional differentiable compact-model example
- `OPTIONAL_TOOLING.md`: setup notes for JAX and Meep
- `outputs/`: generated figures

## Run

From this folder:

```bash
/opt/anaconda3/bin/python3 scripts/compact/run_step1_demo.py
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
/opt/anaconda3/bin/python3 scripts/compact/run_toolchain_example.py
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
