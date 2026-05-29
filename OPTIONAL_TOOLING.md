# Optional Advanced Tooling: JAX and Meep

This project keeps the main compact model lightweight, but two optional examples
show how the same microring direction can connect to more advanced research and
industry tools.

## JAX: Differentiable Compact Modeling

JAX is useful when the compact model should become differentiable. In this
project, that means gradients can flow through the microring transfer function,
so parameters such as coupling coefficient, radius, or loss can be optimized
with gradient-based methods.

Example file:

```bash
scripts/jax/run_jax_example.py
```

Typical separate-environment install:

```bash
python -m pip install "jax[cpu]"
python scripts/jax/run_jax_example.py
```

In this project folder, a local test environment was created with:

```bash
/opt/anaconda3/bin/python3 -m venv .venv-jax
.venv-jax/bin/python -m pip install "jax[cpu]"
.venv-jax/bin/python scripts/jax/run_jax_example.py
```

This example optimizes a simple coupling-coefficient objective using
`jax.grad`/`jax.value_and_grad`.

Jupyter kernel registered on this machine:

```text
Microring Project (JAX)
```

## Meep: Open-Source FDTD Direction

Meep is an open-source FDTD solver. It is closer in spirit to Lumerical FDTD
than to the compact model, but it requires more setup and longer simulations.

Example file:

```bash
scripts/fdtd/run_meep_minimal_waveguide.py
```

Typical separate conda environment:

```bash
conda create -n meep-demo -c conda-forge pymeep python=3.11
conda activate meep-demo
python scripts/fdtd/run_meep_minimal_waveguide.py
```

In this project folder, a local conda-prefix environment was created with:

```bash
/opt/anaconda3/bin/conda create --prefix ./envs/meep-demo -c conda-forge pymeep python=3.11 -y
envs/meep-demo/bin/python scripts/fdtd/run_meep_minimal_waveguide.py
```

The example is intentionally a straight-waveguide smoke test, not a full
microring simulation. A good progression is:

1. straight waveguide transmission
2. bent waveguide loss
3. directional coupler
4. ring or racetrack resonator
5. extract `n_eff`, `n_g`, loss, and coupling for the compact model

There is also a first 2D effective-index microring example:

```bash
envs/meep-demo/bin/python scripts/fdtd/run_meep_microring_2d.py
```

It saves a dielectric/field map, a normalized through-port transmission plot,
and CSV spectrum data under:

```text
outputs/meep_microring_2d/
```

## Why These Are Optional

JAX is useful for differentiable design and optimization. Meep is useful for
field-level electromagnetic simulation. They answer different questions:

- Compact model: fast design-space sweeps
- JAX compact model: gradient-based optimization and differentiable photonics
- Meep/Lumerical FDTD: field-level calibration and validation

## Jupyter Kernels

If a notebook is using Homebrew Python 3.13, imports will fail because that
environment does not have the project packages installed. Use one of these
registered kernels instead:

```text
Microring Project (Anaconda Python 3.12)  # NumPy/SciPy/pandas/xarray/matplotlib
Microring Project (JAX)                   # JAX differentiable compact model
Microring Project (Meep)                  # Meep FDTD
```

In VS Code or Jupyter, click the kernel name in the top-right corner and select
the environment matching the notebook you want to run.
