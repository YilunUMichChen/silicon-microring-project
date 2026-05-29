"""Optional Meep example: minimal 2D waveguide transmission simulation.

This is not yet a full microring FDTD simulation. It is a deliberately small
Meep smoke test showing the workflow needed before building a ring:

1. define cell/material/geometry
2. excite a mode-like source
3. collect transmitted flux

Install Meep in a separate conda environment first. On many systems the package
is distributed through conda-forge, for example:

    conda create -n meep-demo -c conda-forge pymeep python=3.11
    conda activate meep-demo
    python scripts/run_meep_minimal_waveguide.py
"""

from __future__ import annotations

import importlib.util
import os
import pathlib

PROJECT_DIR = pathlib.Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_DIR / "outputs"
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_DIR / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_DIR / ".cache"))

if importlib.util.find_spec("meep") is None:
    raise SystemExit(
        "Meep is not installed in this Python environment.\n"
        "A typical conda-forge setup is:\n"
        "  conda create -n meep-demo -c conda-forge pymeep python=3.11\n"
        "  conda activate meep-demo\n"
        "  python scripts/run_meep_minimal_waveguide.py"
    )

import meep as mp
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    resolution = 20
    cell = mp.Vector3(12, 4, 0)
    pml_layers = [mp.PML(1.0)]

    silicon = mp.Medium(index=3.48)
    geometry = [
        mp.Block(
            material=silicon,
            center=mp.Vector3(),
            size=mp.Vector3(mp.inf, 0.45, mp.inf),
        )
    ]

    wavelength_um = 1.55
    fcen = 1 / wavelength_um
    df = 0.08 * fcen
    sources = [
        mp.Source(
            mp.GaussianSource(fcen, fwidth=df),
            component=mp.Ez,
            center=mp.Vector3(-4.5, 0),
            size=mp.Vector3(0, 1.2),
        )
    ]

    sim = mp.Simulation(
        cell_size=cell,
        boundary_layers=pml_layers,
        geometry=geometry,
        sources=sources,
        resolution=resolution,
    )

    trans_region = mp.FluxRegion(center=mp.Vector3(4.0, 0), size=mp.Vector3(0, 2.0))
    trans = sim.add_flux(fcen, df, 51, trans_region)
    sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, mp.Vector3(4.0, 0), 1e-6))

    freqs = mp.get_flux_freqs(trans)
    fluxes = mp.get_fluxes(trans)
    center_flux = fluxes[len(fluxes) // 2]
    wavelengths = 1.0 / np.asarray(freqs)

    eps_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Dielectric)
    ez_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Ez)
    save_field_plots(eps_data, ez_data, wavelengths, fluxes, cell)

    print(f"Meep waveguide smoke test complete. Center transmitted flux = {center_flux:.6g}")
    print(f"Frequency samples: {len(freqs)}, center wavelength ~= {1 / freqs[len(freqs) // 2]:.3f} um")
    print(f"Saved field image: {OUTPUT_DIR / 'meep_waveguide_fields.png'}")
    print(f"Saved flux spectrum: {OUTPUT_DIR / 'meep_waveguide_flux_spectrum.png'}")


def save_field_plots(
    eps_data: np.ndarray,
    ez_data: np.ndarray,
    wavelengths_um: np.ndarray,
    fluxes: list[float],
    cell: mp.Vector3,
) -> None:
    """Save dielectric/Ez field plots and a flux spectrum."""

    extent = [
        -0.5 * cell.x,
        0.5 * cell.x,
        -0.5 * cell.y,
        0.5 * cell.y,
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    eps_image = axes[0].imshow(
        eps_data.T,
        interpolation="spline36",
        cmap="gray_r",
        origin="lower",
        extent=extent,
        aspect="auto",
    )
    axes[0].set_title("Dielectric profile")
    axes[0].set_xlabel("x (um)")
    axes[0].set_ylabel("y (um)")
    fig.colorbar(eps_image, ax=axes[0], label="epsilon")

    ez_limit = float(np.max(np.abs(ez_data)))
    ez_image = axes[1].imshow(
        ez_data.T,
        interpolation="spline36",
        cmap="RdBu",
        origin="lower",
        extent=extent,
        aspect="auto",
        vmin=-ez_limit,
        vmax=ez_limit,
    )
    axes[1].set_title("Ez field after source decay")
    axes[1].set_xlabel("x (um)")
    axes[1].set_ylabel("y (um)")
    fig.colorbar(ez_image, ax=axes[1], label="Ez")
    fig.savefig(OUTPUT_DIR / "meep_waveguide_fields.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(wavelengths_um, fluxes, color="#1f77b4", linewidth=1.6)
    ax.set_xlabel("Wavelength (um)")
    ax.set_ylabel("Transmitted flux")
    ax.set_title("Waveguide transmitted flux spectrum")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "meep_waveguide_flux_spectrum.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
