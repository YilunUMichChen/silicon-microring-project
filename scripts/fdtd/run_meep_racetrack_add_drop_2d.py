"""2D effective-index Meep FDTD example for a racetrack add-drop resonator.

The racetrack geometry increases straight coupling length compared with a
circular ring, which should make drop-port coupling easier to observe in a
small 2D demo.
"""

from __future__ import annotations

import csv
import importlib.util
import os
import pathlib

PROJECT_DIR = pathlib.Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_DIR / "outputs" / "meep_racetrack_add_drop_2d"
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_DIR / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_DIR / ".cache"))

if importlib.util.find_spec("meep") is None:
    raise SystemExit(
        "Meep is not installed in this Python environment.\n"
        "Use: envs/meep-demo/bin/python scripts/fdtd/run_meep_racetrack_add_drop_2d.py"
    )

import matplotlib.pyplot as plt
import meep as mp
import numpy as np


RADIUS_UM = 1.35
COUPLING_LENGTH_UM = 2.4
WAVEGUIDE_WIDTH_UM = 0.45
GAP_UM = 0.08
N_EFF = 2.6
RESOLUTION = 18
CELL = mp.Vector3(16.0, 10.5, 0)
LAMBDA_CENTER_UM = 1.55
NFREQ = 301


def racetrack_geometry(add_drop: bool) -> tuple[list[mp.GeometricObject], float, float]:
    """Return geometry, top bus y, and racetrack center y."""

    silicon = mp.Medium(index=N_EFF)
    air = mp.Medium(index=1.0)
    outer_r = RADIUS_UM + 0.5 * WAVEGUIDE_WIDTH_UM
    inner_r = RADIUS_UM - 0.5 * WAVEGUIDE_WIDTH_UM
    half_lc = 0.5 * COUPLING_LENGTH_UM
    center_y = 0.5 * WAVEGUIDE_WIDTH_UM + GAP_UM + outer_r
    top_bus_y = center_y + outer_r + GAP_UM + 0.5 * WAVEGUIDE_WIDTH_UM

    objects: list[mp.GeometricObject] = [
        mp.Block(
            material=silicon,
            center=mp.Vector3(0, 0),
            size=mp.Vector3(mp.inf, WAVEGUIDE_WIDTH_UM, mp.inf),
        )
    ]
    if add_drop:
        objects.extend(
            [
                mp.Block(
                    material=silicon,
                    center=mp.Vector3(0, top_bus_y),
                    size=mp.Vector3(mp.inf, WAVEGUIDE_WIDTH_UM, mp.inf),
                ),
                # Outer racetrack envelope.
                mp.Block(
                    material=silicon,
                    center=mp.Vector3(0, center_y),
                    size=mp.Vector3(COUPLING_LENGTH_UM, 2 * outer_r, mp.inf),
                ),
                mp.Cylinder(material=silicon, radius=outer_r, center=mp.Vector3(-half_lc, center_y), height=mp.inf),
                mp.Cylinder(material=silicon, radius=outer_r, center=mp.Vector3(half_lc, center_y), height=mp.inf),
                # Inner air hole, placed after silicon to subtract it.
                mp.Block(
                    material=air,
                    center=mp.Vector3(0, center_y),
                    size=mp.Vector3(COUPLING_LENGTH_UM, 2 * inner_r, mp.inf),
                ),
                mp.Cylinder(material=air, radius=inner_r, center=mp.Vector3(-half_lc, center_y), height=mp.inf),
                mp.Cylinder(material=air, radius=inner_r, center=mp.Vector3(half_lc, center_y), height=mp.inf),
            ]
        )
    return objects, top_bus_y, center_y


def source(fcen: float, df: float) -> list[mp.Source]:
    return [
        mp.Source(
            src=mp.GaussianSource(fcen, fwidth=df),
            component=mp.Ez,
            center=mp.Vector3(-6.4, 0),
            size=mp.Vector3(0, 1.2),
        )
    ]


def run_reference() -> tuple[np.ndarray, np.ndarray]:
    fcen = 1.0 / LAMBDA_CENTER_UM
    df = 0.22 * fcen
    sim = mp.Simulation(
        cell_size=CELL,
        boundary_layers=[mp.PML(1.0)],
        geometry=racetrack_geometry(add_drop=False)[0],
        sources=source(fcen, df),
        resolution=RESOLUTION,
    )
    mon = sim.add_flux(fcen, df, NFREQ, mp.FluxRegion(center=mp.Vector3(6.4, 0), size=mp.Vector3(0, 1.4)))
    sim.run(until_after_sources=mp.stop_when_fields_decayed(80, mp.Ez, mp.Vector3(6.4, 0), 1e-8))
    return np.asarray(mp.get_flux_freqs(mon)), np.asarray(mp.get_fluxes(mon))


def run_racetrack() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, mp.Vector3]:
    fcen = 1.0 / LAMBDA_CENTER_UM
    df = 0.22 * fcen
    objects, top_y, _ = racetrack_geometry(add_drop=True)
    sim = mp.Simulation(
        cell_size=CELL,
        boundary_layers=[mp.PML(1.0)],
        geometry=objects,
        sources=source(fcen, df),
        resolution=RESOLUTION,
    )
    through_mon = sim.add_flux(fcen, df, NFREQ, mp.FluxRegion(center=mp.Vector3(6.4, 0), size=mp.Vector3(0, 1.4)))
    drop_right_mon = sim.add_flux(fcen, df, NFREQ, mp.FluxRegion(center=mp.Vector3(6.4, top_y), size=mp.Vector3(0, 1.4)))
    drop_left_mon = sim.add_flux(fcen, df, NFREQ, mp.FluxRegion(center=mp.Vector3(-6.4, top_y), size=mp.Vector3(0, 1.4)))

    sim.run(until_after_sources=mp.stop_when_fields_decayed(100, mp.Ez, mp.Vector3(6.4, 0), 1e-8))

    freqs = np.asarray(mp.get_flux_freqs(through_mon))
    through = np.asarray(mp.get_fluxes(through_mon))
    drop_right = np.asarray(mp.get_fluxes(drop_right_mon))
    drop_left = np.asarray(mp.get_fluxes(drop_left_mon))
    # Left monitor is opposite-oriented; use magnitude for quick-look energy.
    drop_total = np.abs(drop_right) + np.abs(drop_left)
    eps = sim.get_array(center=mp.Vector3(), size=CELL, component=mp.Dielectric)
    ez = sim.get_array(center=mp.Vector3(), size=CELL, component=mp.Ez)
    return freqs, through, drop_total, eps, ez, CELL


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Running racetrack straight-bus reference...")
    _, ref_flux = run_reference()
    print("Running racetrack add-drop resonator...")
    freqs, through_flux, drop_flux, eps, ez, cell = run_racetrack()

    wavelengths = 1.0 / freqs
    through = through_flux / np.maximum(ref_flux, 1e-30)
    drop = drop_flux / np.maximum(np.abs(ref_flux), 1e-30)
    order = np.argsort(wavelengths)
    wavelengths = wavelengths[order]
    through = through[order]
    drop = drop[order]

    save_csv(wavelengths, through, drop)
    save_spectrum(wavelengths, through, drop)
    save_field(eps, ez, cell)

    t_idx = int(np.argmin(through))
    d_idx = int(np.argmax(drop))
    print("Racetrack add-drop example complete.")
    print(f"Through minimum = {through[t_idx]:.4f} at {wavelengths[t_idx]:.4f} um")
    print(f"Drop maximum = {drop[d_idx]:.4f} at {wavelengths[d_idx]:.4f} um")
    print(f"Geometry: R={RADIUS_UM:.2f} um, Lc={COUPLING_LENGTH_UM:.2f} um, gap={GAP_UM:.2f} um")
    print(f"Saved spectrum: {OUTPUT_DIR / 'racetrack_add_drop_spectrum.png'}")
    print(f"Saved field map: {OUTPUT_DIR / 'racetrack_add_drop_fields.png'}")


def save_csv(wavelengths: np.ndarray, through: np.ndarray, drop: np.ndarray) -> None:
    with (OUTPUT_DIR / "racetrack_add_drop_spectrum.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["wavelength_um", "through_transmission", "drop_transmission_total"])
        writer.writerows(zip(wavelengths, through, drop))


def save_spectrum(wavelengths: np.ndarray, through: np.ndarray, drop: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    ax.plot(wavelengths, through, label="through port")
    ax.plot(wavelengths, drop, label="drop port total")
    ax.set_xlabel("Wavelength (um)")
    ax.set_ylabel("Normalized transmission")
    ax.set_title("2D Meep racetrack add-drop spectrum")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "racetrack_add_drop_spectrum.png", dpi=220)
    plt.close(fig)


def save_field(eps: np.ndarray, ez: np.ndarray, cell: mp.Vector3) -> None:
    extent = [-0.5 * cell.x, 0.5 * cell.x, -0.5 * cell.y, 0.5 * cell.y]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), constrained_layout=True)
    eps_img = axes[0].imshow(eps.T, cmap="gray_r", origin="lower", extent=extent, interpolation="spline36", aspect="equal")
    axes[0].set_title("Racetrack dielectric profile")
    axes[0].set_xlabel("x (um)")
    axes[0].set_ylabel("y (um)")
    fig.colorbar(eps_img, ax=axes[0], label="epsilon")

    ez_lim = float(np.percentile(np.abs(ez), 99.5))
    ez_img = axes[1].imshow(
        ez.T,
        cmap="RdBu",
        origin="lower",
        extent=extent,
        interpolation="spline36",
        aspect="equal",
        vmin=-ez_lim,
        vmax=ez_lim,
    )
    axes[1].contour(eps.T, levels=[2.0], colors="black", linewidths=0.7, origin="lower", extent=extent)
    axes[1].set_title("Ez field after source decay")
    axes[1].set_xlabel("x (um)")
    axes[1].set_ylabel("y (um)")
    fig.colorbar(ez_img, ax=axes[1], label="Ez")
    fig.savefig(OUTPUT_DIR / "racetrack_add_drop_fields.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
