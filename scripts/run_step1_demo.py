"""Generate first-step compact-model figures and printed metrics."""

from __future__ import annotations

import pathlib
import sys
import os

PROJECT_DIR = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
OUTPUT_DIR = PROJECT_DIR / "outputs"
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_DIR / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_DIR / ".cache"))

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(SRC_DIR))

from microring_model import (  # noqa: E402
    RingParams,
    all_pass_response,
    extract_fsr_from_dips,
    extract_resonance_metrics,
    find_resonance_dips,
    fsr_nm,
)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    params = RingParams(
        radius_um=10.0,
        lambda_ref_nm=1550.0,
        n_eff=2.4,
        n_g=4.0,
        loss_db_per_cm=2.0,
        kappa=0.25,
    )
    wavelengths = np.linspace(1520.0, 1580.0, 50000)
    transmission, phase = all_pass_response(wavelengths, params)

    dips = find_resonance_dips(wavelengths, transmission)
    metrics = extract_resonance_metrics(wavelengths, transmission)
    measured_fsr = extract_fsr_from_dips(wavelengths, dips)
    analytic_fsr = fsr_nm(1550.0, params.radius_um, params.n_g)

    print("Step 1 compact model demo")
    print(f"Parameters: {params}")
    print(f"Analytic FSR near 1550 nm: {analytic_fsr:.3f} nm")
    if measured_fsr is not None:
        print(f"Extracted FSR from dips: {measured_fsr:.3f} nm")
    print(f"Strongest resonance: {metrics.resonance_nm:.3f} nm")
    print(f"FWHM: {metrics.fwhm_nm:.4f} nm")
    print(f"Loaded Q: {metrics.q_loaded:.0f}")
    print(f"Extinction ratio: {metrics.extinction_ratio_db:.2f} dB")

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(wavelengths, transmission, color="#1f77b4", linewidth=1.4)
    axes[0].scatter(wavelengths[dips], transmission[dips], s=12, color="#d62728", zorder=3)
    axes[0].set_ylabel("Transmission")
    axes[0].set_title("All-pass microring compact model")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(wavelengths, phase, color="#2ca02c", linewidth=1.2)
    axes[1].set_xlabel("Wavelength (nm)")
    axes[1].set_ylabel("Unwrapped phase (rad)")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "step1_all_pass_response.png", dpi=200)
    plt.close(fig)

    kappa_values = [0.05, 0.15, 0.25, 0.4, 0.6]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for kappa in kappa_values:
        sweep_params = RingParams(
            radius_um=params.radius_um,
            lambda_ref_nm=params.lambda_ref_nm,
            n_eff=params.n_eff,
            n_g=params.n_g,
            loss_db_per_cm=params.loss_db_per_cm,
            kappa=kappa,
        )
        t_sweep, _ = all_pass_response(wavelengths, sweep_params)
        ax.plot(wavelengths, t_sweep, linewidth=1.2, label=f"kappa={kappa:.2f}")

    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Transmission")
    ax.set_title("Coupling sweep")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "step1_coupling_sweep.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
