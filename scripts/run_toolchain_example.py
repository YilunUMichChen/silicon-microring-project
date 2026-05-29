"""Example using common scientific Python tools for microring analysis.

This script keeps the compact physics model in NumPy, then uses:

- SciPy for resonance finding and Lorentzian notch fitting
- pandas for tabular design-sweep results
- xarray for labeled multi-dimensional sweep data
- Matplotlib for plots
"""

from __future__ import annotations

import os
import pathlib
import sys

PROJECT_DIR = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
OUTPUT_DIR = PROJECT_DIR / "outputs"
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_DIR / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_DIR / ".cache"))
sys.path.insert(0, str(SRC_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, peak_widths

from microring_model import RingParams, all_pass_response, fsr_nm


def lorentzian_notch(lambda_nm: np.ndarray, baseline: float, depth: float, lambda0: float, gamma: float) -> np.ndarray:
    """Lorentzian transmission dip model.

    gamma is the half-width at half-maximum, so FWHM = 2 * gamma.
    """

    return baseline - depth / (1.0 + ((lambda_nm - lambda0) / gamma) ** 2)


def scipy_resonance_fit(lambda_nm: np.ndarray, transmission: np.ndarray) -> dict[str, float]:
    """Use SciPy peak detection and curve fitting to extract resonance metrics."""

    inverted = 1.0 - transmission
    dynamic_range = float(np.max(inverted) - np.min(inverted))
    prominence = max(0.01 * dynamic_range, 1e-6)
    peaks, properties = find_peaks(inverted, prominence=prominence)
    if len(peaks) == 0:
        peaks = np.array([int(np.argmax(inverted))])
        properties = {"prominences": np.array([dynamic_range])}

    strongest_peak = int(peaks[np.argmax(properties["prominences"])])
    width_samples = peak_widths(inverted, [strongest_peak], rel_height=0.5)[0][0]
    sample_spacing = float(np.mean(np.diff(lambda_nm)))
    estimated_fwhm = max(width_samples * sample_spacing, sample_spacing)

    lambda0_guess = float(lambda_nm[strongest_peak])
    baseline_guess = float(np.percentile(transmission, 90))
    depth_guess = max(baseline_guess - float(transmission[strongest_peak]), 1e-6)
    gamma_guess = max(estimated_fwhm / 2.0, sample_spacing)

    window_nm = max(8.0 * estimated_fwhm, 0.5)
    fit_mask = np.abs(lambda_nm - lambda0_guess) <= window_nm
    x_fit = lambda_nm[fit_mask]
    y_fit = transmission[fit_mask]

    lower = [0.0, 0.0, lambda0_guess - window_nm, sample_spacing / 2.0]
    upper = [1.5, 1.5, lambda0_guess + window_nm, window_nm]
    popt, _ = curve_fit(
        lorentzian_notch,
        x_fit,
        y_fit,
        p0=[baseline_guess, depth_guess, lambda0_guess, gamma_guess],
        bounds=(lower, upper),
        maxfev=10000,
    )
    baseline, depth, lambda0, gamma = [float(v) for v in popt]
    fwhm = 2.0 * abs(gamma)
    min_transmission = baseline - depth
    extinction_ratio_db = 10.0 * np.log10(max(baseline, 1e-15) / max(min_transmission, 1e-15))

    return {
        "resonance_nm": lambda0,
        "fwhm_nm": fwhm,
        "q_loaded": lambda0 / fwhm,
        "extinction_ratio_db": extinction_ratio_db,
        "baseline_transmission": baseline,
        "min_transmission": min_transmission,
        "fit_window_nm": window_nm,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    wavelengths = np.linspace(1535.0, 1565.0, 40000)
    radius_values = np.array([5.0, 7.5, 10.0, 12.5, 15.0])
    kappa_values = np.array([0.05, 0.10, 0.15, 0.25, 0.40, 0.60])
    loss_values = np.array([0.5, 2.0, 5.0])

    records: list[dict[str, float]] = []
    q_grid = np.full((len(radius_values), len(kappa_values), len(loss_values)), np.nan)
    er_grid = np.full_like(q_grid, np.nan)

    for i, radius_um in enumerate(radius_values):
        for j, kappa in enumerate(kappa_values):
            for k, loss_db_per_cm in enumerate(loss_values):
                params = RingParams(
                    radius_um=float(radius_um),
                    lambda_ref_nm=1550.0,
                    n_eff=2.4,
                    n_g=4.0,
                    loss_db_per_cm=float(loss_db_per_cm),
                    kappa=float(kappa),
                )
                transmission, _ = all_pass_response(wavelengths, params)
                metrics = scipy_resonance_fit(wavelengths, transmission)
                metrics.update(
                    {
                        "radius_um": float(radius_um),
                        "kappa": float(kappa),
                        "loss_db_per_cm": float(loss_db_per_cm),
                        "analytic_fsr_nm": fsr_nm(1550.0, float(radius_um), 4.0),
                    }
                )
                records.append(metrics)
                q_grid[i, j, k] = metrics["q_loaded"]
                er_grid[i, j, k] = metrics["extinction_ratio_db"]

    df = pd.DataFrame.from_records(records)
    df = df[
        [
            "radius_um",
            "kappa",
            "loss_db_per_cm",
            "resonance_nm",
            "analytic_fsr_nm",
            "fwhm_nm",
            "q_loaded",
            "extinction_ratio_db",
            "baseline_transmission",
            "min_transmission",
        ]
    ]
    df.to_csv(OUTPUT_DIR / "toolchain_design_sweep.csv", index=False)

    dataset = xr.Dataset(
        data_vars={
            "q_loaded": (("radius_um", "kappa", "loss_db_per_cm"), q_grid),
            "extinction_ratio_db": (("radius_um", "kappa", "loss_db_per_cm"), er_grid),
        },
        coords={
            "radius_um": radius_values,
            "kappa": kappa_values,
            "loss_db_per_cm": loss_values,
        },
        attrs={
            "description": "Microring compact-model sweep stored as labeled scientific data.",
            "n_eff": 2.4,
            "n_g": 4.0,
            "lambda_ref_nm": 1550.0,
        },
    )
    dataset.to_netcdf(OUTPUT_DIR / "toolchain_design_sweep.nc", engine="scipy")

    plot_design_maps(dataset.sel(loss_db_per_cm=2.0))

    best_q = df.sort_values("q_loaded", ascending=False).iloc[0]
    best_er = df.sort_values("extinction_ratio_db", ascending=False).iloc[0]
    print("Toolchain example complete")
    print(f"Rows saved: {len(df)}")
    print(
        "Best Q: "
        f"R={best_q.radius_um:.1f} um, kappa={best_q.kappa:.2f}, "
        f"loss={best_q.loss_db_per_cm:.1f} dB/cm, Q={best_q.q_loaded:.0f}"
    )
    print(
        "Best ER: "
        f"R={best_er.radius_um:.1f} um, kappa={best_er.kappa:.2f}, "
        f"loss={best_er.loss_db_per_cm:.1f} dB/cm, ER={best_er.extinction_ratio_db:.2f} dB"
    )


def plot_design_maps(dataset_2d: xr.Dataset) -> None:
    """Plot xarray-labeled design maps at one selected loss value."""

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)

    q_plot = dataset_2d["q_loaded"].transpose("radius_um", "kappa")
    er_plot = dataset_2d["extinction_ratio_db"].transpose("radius_um", "kappa")

    mesh0 = axes[0].pcolormesh(q_plot["kappa"], q_plot["radius_um"], q_plot.values, shading="auto", cmap="viridis")
    axes[0].set_title("Loaded Q")
    axes[0].set_xlabel("Coupling coefficient kappa")
    axes[0].set_ylabel("Radius (um)")
    fig.colorbar(mesh0, ax=axes[0])

    mesh1 = axes[1].pcolormesh(er_plot["kappa"], er_plot["radius_um"], er_plot.values, shading="auto", cmap="magma")
    axes[1].set_title("Extinction ratio (dB)")
    axes[1].set_xlabel("Coupling coefficient kappa")
    axes[1].set_ylabel("Radius (um)")
    fig.colorbar(mesh1, ax=axes[1])

    loss_value = float(dataset_2d["loss_db_per_cm"])
    fig.suptitle(f"SciPy/pandas/xarray design sweep at loss = {loss_value:.1f} dB/cm")
    fig.savefig(OUTPUT_DIR / "toolchain_design_maps.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
