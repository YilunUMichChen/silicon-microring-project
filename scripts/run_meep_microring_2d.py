"""2D effective-index Meep FDTD example for a bus-coupled microring.

This is a small, runnable starting point rather than a final calibrated device
simulation. It produces Lumerical-like quick-look outputs:

- dielectric map
- Ez field map after the pulse decays
- through-port transmission spectrum normalized to a straight-waveguide run
- CSV data for later fitting

The geometry is a 2D effective-index approximation of an SOI microring.
"""

from __future__ import annotations

import csv
import importlib.util
import os
import pathlib

PROJECT_DIR = pathlib.Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_DIR / "outputs" / "meep_microring_2d"
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_DIR / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_DIR / ".cache"))

if importlib.util.find_spec("meep") is None:
    raise SystemExit(
        "Meep is not installed in this Python environment.\n"
        "Use the local Meep environment:\n"
        "  envs/meep-demo/bin/python scripts/run_meep_microring_2d.py"
    )

import matplotlib.pyplot as plt
import meep as mp
import numpy as np


def build_geometry(
    ring: bool,
    ring_radius_um: float,
    waveguide_width_um: float,
    gap_um: float,
    n_eff: float,
) -> tuple[list[mp.GeometricObject], mp.Vector3]:
    """Create straight bus waveguide plus optional annular ring."""

    silicon_eff = mp.Medium(index=n_eff)
    air = mp.Medium(index=1.0)
    outer_radius = ring_radius_um + 0.5 * waveguide_width_um
    inner_radius = ring_radius_um - 0.5 * waveguide_width_um
    ring_center_y = 0.5 * waveguide_width_um + gap_um + outer_radius
    ring_center = mp.Vector3(0, ring_center_y)

    geometry: list[mp.GeometricObject] = [
        mp.Block(
            material=silicon_eff,
            center=mp.Vector3(0, 0),
            size=mp.Vector3(mp.inf, waveguide_width_um, mp.inf),
        )
    ]

    if ring:
        geometry.extend(
            [
                mp.Cylinder(
                    material=silicon_eff,
                    radius=outer_radius,
                    center=ring_center,
                    height=mp.inf,
                ),
                mp.Cylinder(
                    material=air,
                    radius=inner_radius,
                    center=ring_center,
                    height=mp.inf,
                ),
            ]
        )

    return geometry, ring_center


def run_simulation(ring: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, mp.Vector3]:
    """Run reference or ring simulation and return flux plus optional fields."""

    resolution = 18
    pml = 1.0
    sx = 14.0
    sy = 9.0
    cell = mp.Vector3(sx, sy, 0)

    ring_radius_um = 2.2
    waveguide_width_um = 0.45
    gap_um = 0.20
    n_eff = 2.6

    geometry, _ = build_geometry(ring, ring_radius_um, waveguide_width_um, gap_um, n_eff)

    lambda_center_um = 1.55
    fcen = 1.0 / lambda_center_um
    df = 0.18 * fcen
    nfreq = 161

    sources = [
        mp.Source(
            src=mp.GaussianSource(fcen, fwidth=df),
            component=mp.Ez,
            center=mp.Vector3(-5.5, 0),
            size=mp.Vector3(0, 1.2),
        )
    ]

    sim = mp.Simulation(
        cell_size=cell,
        boundary_layers=[mp.PML(pml)],
        geometry=geometry,
        sources=sources,
        resolution=resolution,
    )

    trans_region = mp.FluxRegion(center=mp.Vector3(5.5, 0), size=mp.Vector3(0, 1.6))
    trans = sim.add_flux(fcen, df, nfreq, trans_region)

    sim.run(until_after_sources=mp.stop_when_fields_decayed(60, mp.Ez, mp.Vector3(5.5, 0), 1e-7))

    freqs = np.asarray(mp.get_flux_freqs(trans), dtype=float)
    fluxes = np.asarray(mp.get_fluxes(trans), dtype=float)

    eps_data = None
    ez_data = None
    if ring:
        eps_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Dielectric)
        ez_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Ez)

    return freqs, fluxes, eps_data, ez_data, cell


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Running straight-waveguide reference simulation...")
    ref_freqs, ref_flux, _, _, _ = run_simulation(ring=False)

    print("Running bus-coupled microring simulation...")
    freqs, ring_flux, eps_data, ez_data, cell = run_simulation(ring=True)

    wavelengths_um = 1.0 / freqs
    transmission = ring_flux / np.maximum(ref_flux, 1e-30)
    order = np.argsort(wavelengths_um)
    wavelengths_um = wavelengths_um[order]
    transmission = transmission[order]

    save_csv(wavelengths_um, transmission, OUTPUT_DIR / "microring_transmission.csv")
    save_transmission_plot(wavelengths_um, transmission)
    if eps_data is not None and ez_data is not None:
        save_field_plot(eps_data, ez_data, cell)

    print("Running continuous-wave field visualization...")
    cw_eps, cw_ez, cw_cell = run_cw_field_simulation(wavelength_um=1.55)
    save_cw_field_plot(cw_eps, cw_ez, cw_cell)

    min_idx = int(np.argmin(transmission))
    print("2D microring FDTD example complete.")
    print(f"Minimum through transmission = {transmission[min_idx]:.4f}")
    print(f"Approximate dip wavelength = {wavelengths_um[min_idx]:.4f} um")
    print(f"Saved dielectric/field map: {OUTPUT_DIR / 'microring_fields.png'}")
    print(f"Saved CW field map: {OUTPUT_DIR / 'microring_cw_field.png'}")
    print(f"Saved transmission spectrum: {OUTPUT_DIR / 'microring_transmission.png'}")
    print(f"Saved CSV data: {OUTPUT_DIR / 'microring_transmission.csv'}")


def run_cw_field_simulation(wavelength_um: float) -> tuple[np.ndarray, np.ndarray, mp.Vector3]:
    """Run a short continuous-wave simulation for a more visible field map."""

    resolution = 18
    sx = 14.0
    sy = 9.0
    cell = mp.Vector3(sx, sy, 0)
    geometry, _ = build_geometry(
        ring=True,
        ring_radius_um=2.2,
        waveguide_width_um=0.45,
        gap_um=0.20,
        n_eff=2.6,
    )

    frequency = 1.0 / wavelength_um
    sources = [
        mp.Source(
            src=mp.ContinuousSource(frequency=frequency, width=20),
            component=mp.Ez,
            center=mp.Vector3(-5.5, 0),
            size=mp.Vector3(0, 1.2),
        )
    ]

    sim = mp.Simulation(
        cell_size=cell,
        boundary_layers=[mp.PML(1.0)],
        geometry=geometry,
        sources=sources,
        resolution=resolution,
    )
    sim.run(until=220)

    eps_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Dielectric)
    ez_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Ez)
    return eps_data, ez_data, cell


def save_csv(wavelengths_um: np.ndarray, transmission: np.ndarray, path: pathlib.Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["wavelength_um", "through_transmission"])
        writer.writerows(zip(wavelengths_um, transmission))


def save_transmission_plot(wavelengths_um: np.ndarray, transmission: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(wavelengths_um, transmission, color="#1f77b4", linewidth=1.5)
    ax.set_xlabel("Wavelength (um)")
    ax.set_ylabel("Normalized through transmission")
    ax.set_title("2D Meep microring through-port spectrum")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "microring_transmission.png", dpi=200)
    plt.close(fig)


def save_field_plot(eps_data: np.ndarray, ez_data: np.ndarray, cell: mp.Vector3) -> None:
    extent = [-0.5 * cell.x, 0.5 * cell.x, -0.5 * cell.y, 0.5 * cell.y]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5), constrained_layout=True)

    eps_image = axes[0].imshow(
        eps_data.T,
        cmap="gray_r",
        origin="lower",
        extent=extent,
        interpolation="spline36",
        aspect="equal",
    )
    axes[0].set_title("Dielectric profile")
    axes[0].set_xlabel("x (um)")
    axes[0].set_ylabel("y (um)")
    fig.colorbar(eps_image, ax=axes[0], label="epsilon")

    ez_limit = float(np.max(np.abs(ez_data)))
    ez_image = axes[1].imshow(
        ez_data.T,
        cmap="RdBu",
        origin="lower",
        extent=extent,
        interpolation="spline36",
        aspect="equal",
        vmin=-ez_limit,
        vmax=ez_limit,
    )
    axes[1].set_title("Ez field after source decay")
    axes[1].set_xlabel("x (um)")
    axes[1].set_ylabel("y (um)")
    fig.colorbar(ez_image, ax=axes[1], label="Ez")

    fig.savefig(OUTPUT_DIR / "microring_fields.png", dpi=200)
    plt.close(fig)


def save_cw_field_plot(eps_data: np.ndarray, ez_data: np.ndarray, cell: mp.Vector3) -> None:
    extent = [-0.5 * cell.x, 0.5 * cell.x, -0.5 * cell.y, 0.5 * cell.y]
    fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)

    ez_limit = float(np.percentile(np.abs(ez_data), 99.5))
    image = ax.imshow(
        ez_data.T,
        cmap="RdBu",
        origin="lower",
        extent=extent,
        interpolation="spline36",
        aspect="equal",
        vmin=-ez_limit,
        vmax=ez_limit,
    )
    ax.contour(
        eps_data.T,
        levels=[2.0],
        colors="black",
        linewidths=0.8,
        origin="lower",
        extent=extent,
    )
    ax.set_title("Continuous-wave Ez field at 1.55 um")
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    fig.colorbar(image, ax=ax, label="Ez")
    fig.savefig(OUTPUT_DIR / "microring_cw_field.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
