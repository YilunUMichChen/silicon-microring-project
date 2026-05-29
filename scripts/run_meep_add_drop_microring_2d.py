"""2D effective-index Meep FDTD example for an add-drop microring.

Compared with the single-bus ring, an add-drop ring has a second bus waveguide.
The through port should show resonance dips, while the drop port should show
resonance peaks when the ring couples energy into the upper bus.
"""

from __future__ import annotations

import csv
import importlib.util
import os
import pathlib

PROJECT_DIR = pathlib.Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_DIR / "outputs" / "meep_add_drop_2d"
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_DIR / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_DIR / ".cache"))

if importlib.util.find_spec("meep") is None:
    raise SystemExit(
        "Meep is not installed in this Python environment.\n"
        "Use the local Meep environment:\n"
        "  envs/meep-demo/bin/python scripts/run_meep_add_drop_microring_2d.py"
    )

import matplotlib.pyplot as plt
import meep as mp
import numpy as np


RING_RADIUS_UM = 1.6
WAVEGUIDE_WIDTH_UM = 0.45
GAP_UM = 0.10
N_EFF = 2.6
RESOLUTION = 20
CELL = mp.Vector3(14.0, 11.0, 0)
LAMBDA_CENTER_UM = 1.55
NFREQ = 301


def geometry(add_drop: bool) -> tuple[list[mp.GeometricObject], float]:
    """Return add-drop ring geometry and upper-bus center y."""

    silicon_eff = mp.Medium(index=N_EFF)
    air = mp.Medium(index=1.0)
    outer_radius = RING_RADIUS_UM + 0.5 * WAVEGUIDE_WIDTH_UM
    inner_radius = RING_RADIUS_UM - 0.5 * WAVEGUIDE_WIDTH_UM
    ring_center_y = 0.5 * WAVEGUIDE_WIDTH_UM + GAP_UM + outer_radius
    drop_y = ring_center_y + outer_radius + GAP_UM + 0.5 * WAVEGUIDE_WIDTH_UM

    objects: list[mp.GeometricObject] = [
        mp.Block(
            material=silicon_eff,
            center=mp.Vector3(0, 0),
            size=mp.Vector3(mp.inf, WAVEGUIDE_WIDTH_UM, mp.inf),
        )
    ]

    if add_drop:
        objects.extend(
            [
                mp.Block(
                    material=silicon_eff,
                    center=mp.Vector3(0, drop_y),
                    size=mp.Vector3(mp.inf, WAVEGUIDE_WIDTH_UM, mp.inf),
                ),
                mp.Cylinder(
                    material=silicon_eff,
                    radius=outer_radius,
                    center=mp.Vector3(0, ring_center_y),
                    height=mp.inf,
                ),
                mp.Cylinder(
                    material=air,
                    radius=inner_radius,
                    center=mp.Vector3(0, ring_center_y),
                    height=mp.inf,
                ),
            ]
        )

    return objects, drop_y


def run_reference() -> tuple[np.ndarray, np.ndarray]:
    """Run straight lower-bus reference to normalize port powers."""

    fcen = 1.0 / LAMBDA_CENTER_UM
    df = 0.22 * fcen
    sim = mp.Simulation(
        cell_size=CELL,
        boundary_layers=[mp.PML(1.0)],
        geometry=geometry(add_drop=False)[0],
        sources=source(fcen, df),
        resolution=RESOLUTION,
    )
    ref_region = mp.FluxRegion(center=mp.Vector3(5.7, 0), size=mp.Vector3(0, 1.4))
    ref_monitor = sim.add_flux(fcen, df, NFREQ, ref_region)
    sim.run(until_after_sources=mp.stop_when_fields_decayed(80, mp.Ez, mp.Vector3(5.7, 0), 1e-8))
    return np.asarray(mp.get_flux_freqs(ref_monitor)), np.asarray(mp.get_fluxes(ref_monitor))


def run_add_drop() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, mp.Vector3]:
    """Run add-drop microring and return through/drop fluxes plus fields."""

    fcen = 1.0 / LAMBDA_CENTER_UM
    df = 0.22 * fcen
    objects, drop_y = geometry(add_drop=True)
    sim = mp.Simulation(
        cell_size=CELL,
        boundary_layers=[mp.PML(1.0)],
        geometry=objects,
        sources=source(fcen, df),
        resolution=RESOLUTION,
    )

    through_region = mp.FluxRegion(center=mp.Vector3(5.7, 0), size=mp.Vector3(0, 1.4))
    drop_region = mp.FluxRegion(center=mp.Vector3(5.7, drop_y), size=mp.Vector3(0, 1.4))
    through_monitor = sim.add_flux(fcen, df, NFREQ, through_region)
    drop_monitor = sim.add_flux(fcen, df, NFREQ, drop_region)

    sim.run(until_after_sources=mp.stop_when_fields_decayed(100, mp.Ez, mp.Vector3(5.7, 0), 1e-8))

    freqs = np.asarray(mp.get_flux_freqs(through_monitor))
    through_flux = np.asarray(mp.get_fluxes(through_monitor))
    drop_flux = np.asarray(mp.get_fluxes(drop_monitor))
    eps_data = sim.get_array(center=mp.Vector3(), size=CELL, component=mp.Dielectric)
    ez_data = sim.get_array(center=mp.Vector3(), size=CELL, component=mp.Ez)
    return freqs, through_flux, drop_flux, eps_data, ez_data, CELL


def source(fcen: float, df: float) -> list[mp.Source]:
    """Broadband source injected into the lower bus waveguide."""

    return [
        mp.Source(
            src=mp.GaussianSource(fcen, fwidth=df),
            component=mp.Ez,
            center=mp.Vector3(-5.7, 0),
            size=mp.Vector3(0, 1.2),
        )
    ]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Running lower-bus reference...")
    ref_freqs, ref_flux = run_reference()

    print("Running add-drop microring...")
    freqs, through_flux, drop_flux, eps_data, ez_data, cell = run_add_drop()

    wavelengths_um = 1.0 / freqs
    through = through_flux / np.maximum(ref_flux, 1e-30)
    drop = drop_flux / np.maximum(ref_flux, 1e-30)

    order = np.argsort(wavelengths_um)
    wavelengths_um = wavelengths_um[order]
    through = through[order]
    drop = drop[order]

    save_csv(wavelengths_um, through, drop)
    save_spectrum_plot(wavelengths_um, through, drop)
    save_field_plot(eps_data, ez_data, cell)

    through_min_idx = int(np.argmin(through))
    drop_max_idx = int(np.argmax(drop))
    print("Add-drop microring example complete.")
    print(f"Through minimum = {through[through_min_idx]:.4f} at {wavelengths_um[through_min_idx]:.4f} um")
    print(f"Drop maximum = {drop[drop_max_idx]:.4f} at {wavelengths_um[drop_max_idx]:.4f} um")
    print(f"Saved spectrum: {OUTPUT_DIR / 'add_drop_spectrum.png'}")
    print(f"Saved field map: {OUTPUT_DIR / 'add_drop_fields.png'}")
    print(f"Saved CSV: {OUTPUT_DIR / 'add_drop_spectrum.csv'}")


def save_csv(wavelengths_um: np.ndarray, through: np.ndarray, drop: np.ndarray) -> None:
    with (OUTPUT_DIR / "add_drop_spectrum.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["wavelength_um", "through_transmission", "drop_transmission"])
        writer.writerows(zip(wavelengths_um, through, drop))


def save_spectrum_plot(wavelengths_um: np.ndarray, through: np.ndarray, drop: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(wavelengths_um, through, label="through port", linewidth=1.5)
    ax.plot(wavelengths_um, drop, label="drop port", linewidth=1.5)
    ax.set_xlabel("Wavelength (um)")
    ax.set_ylabel("Normalized transmission")
    ax.set_title("2D Meep add-drop microring spectrum")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "add_drop_spectrum.png", dpi=220)
    plt.close(fig)


def save_field_plot(eps_data: np.ndarray, ez_data: np.ndarray, cell: mp.Vector3) -> None:
    extent = [-0.5 * cell.x, 0.5 * cell.x, -0.5 * cell.y, 0.5 * cell.y]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), constrained_layout=True)

    eps_image = axes[0].imshow(
        eps_data.T,
        cmap="gray_r",
        origin="lower",
        extent=extent,
        interpolation="spline36",
        aspect="equal",
    )
    axes[0].set_title("Add-drop dielectric profile")
    axes[0].set_xlabel("x (um)")
    axes[0].set_ylabel("y (um)")
    fig.colorbar(eps_image, ax=axes[0], label="epsilon")

    ez_limit = float(np.percentile(np.abs(ez_data), 99.5))
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
    axes[1].contour(eps_data.T, levels=[2.0], colors="black", linewidths=0.7, origin="lower", extent=extent)
    axes[1].set_title("Ez field after source decay")
    axes[1].set_xlabel("x (um)")
    axes[1].set_ylabel("y (um)")
    fig.colorbar(ez_image, ax=axes[1], label="Ez")

    fig.savefig(OUTPUT_DIR / "add_drop_fields.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
