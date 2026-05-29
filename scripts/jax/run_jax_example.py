"""Optional JAX example: differentiable microring design objective.

Install JAX first, for example in a separate environment:

    python -m pip install "jax[cpu]"

The example uses JAX automatic differentiation to tune the coupling coefficient
for a target extinction ratio near one resonance.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

if importlib.util.find_spec("jax") is None:
    raise SystemExit(
        "JAX is not installed in this Python environment.\n"
        "Try a separate environment first, then run:\n"
        "  python -m pip install \"jax[cpu]\"\n"
        "  python scripts/jax/run_jax_example.py"
    )

import jax
import jax.numpy as jnp


PROJECT_DIR = pathlib.Path(__file__).resolve().parents[2]


def all_pass_transmission(lambda_nm: jnp.ndarray, radius_um: float, n_eff: float, n_g: float, loss_db_cm: float, kappa: float) -> jnp.ndarray:
    """JAX version of the all-pass compact model."""

    lambda_ref_nm = 1550.0
    length_um = 2.0 * jnp.pi * radius_um
    length_cm = length_um * 1e-4

    alpha_power_cm = loss_db_cm * jnp.log(10.0) / 10.0
    a = jnp.exp(-alpha_power_cm * length_cm / 2.0)
    t = jnp.sqrt(1.0 - kappa**2)

    dneff_dlambda = (n_eff - n_g) / lambda_ref_nm
    n_eff_lambda = n_eff + dneff_dlambda * (lambda_nm - lambda_ref_nm)
    beta = 2.0 * jnp.pi * n_eff_lambda / lambda_nm
    phi = beta * length_um * 1e3
    g = a * jnp.exp(1j * phi)
    field = (t - g) / (1.0 - t * g)
    return jnp.abs(field) ** 2


def objective(raw_kappa: float) -> float:
    """Maximize a smooth extinction-ratio proxy while avoiding extremes."""

    kappa = 0.02 + 0.76 * jax.nn.sigmoid(raw_kappa)
    wavelengths = jnp.linspace(1548.0, 1558.0, 4096)
    transmission = all_pass_transmission(
        wavelengths,
        radius_um=10.0,
        n_eff=2.4,
        n_g=4.0,
        loss_db_cm=2.0,
        kappa=kappa,
    )
    smoothness = 250.0
    smooth_dip = -jax.nn.logsumexp(-smoothness * transmission) / smoothness
    baseline = jnp.mean(transmission)
    er_db = 10.0 * jnp.log10(baseline / jnp.maximum(smooth_dip, 1e-9))
    penalty = 0.1 * (kappa - 0.12) ** 2
    return -(er_db - penalty)


def main() -> None:
    value_and_grad = jax.jit(jax.value_and_grad(objective))
    raw_kappa = jnp.array(0.0)
    learning_rate = 0.08

    for step in range(40):
        loss, grad = value_and_grad(raw_kappa)
        raw_kappa = raw_kappa - learning_rate * grad
        if step % 10 == 0 or step == 39:
            kappa = 0.02 + 0.76 * jax.nn.sigmoid(raw_kappa)
            print(f"step={step:02d} objective={float(loss):.4f} kappa={float(kappa):.4f}")

    print("JAX example complete: differentiated through the compact microring model.")


if __name__ == "__main__":
    main()
