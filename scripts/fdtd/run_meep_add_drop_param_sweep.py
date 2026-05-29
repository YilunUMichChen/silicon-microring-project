"""Small Meep FDTD parameter sweep for add-drop microring candidates.

Use this only after compact optimization narrows the design space. It sweeps a
small set of radius/gap values and records through minima and drop maxima.
"""

from __future__ import annotations

import csv
import importlib.util
import os
import pathlib

PROJECT_DIR = pathlib.Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_DIR / "outputs" / "meep_add_drop_sweep"
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_DIR / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_DIR / ".cache"))

if importlib.util.find_spec("meep") is None:
    raise SystemExit(
        "Meep is not installed in this Python environment.\n"
        "Use: envs/meep-demo/bin/python scripts/fdtd/run_meep_add_drop_param_sweep.py"
    )

import matplotlib.pyplot as plt
import meep as mp
import numpy as np


WAVEGUIDE_WIDTH_UM = 0.45
N_EFF = 2.6
RESOLUTION = 16
CELL = mp.Vector3(14.0, 11.0, 0)
LAMBDA_CENTER_UM = 1.55
NFREQ = 181


def build_geometry(radius_um: float, gap_um: float, add_drop: bool) -> tuple[list[mp.GeometricObject], float]:
    silicon_eff = mp.Medium(index=N_EFF)
    air = mp.Medium(index=1.0)
    outer_radius = radius_um + 0.5 * WAVEGUIDE_WIDTH_UM
    inner_radius = radius_um - 0.5 * WAVEGUIDE_WIDTH_UM
    ring_center_y = 0.5 * WAVEGUIDE_WIDTH_UM + gap_um + outer_radius
    drop_y = ring_center_y + outer_radius + gap_um + 0.5 * WAVEGUIDE_WIDTH_UM

    objects: list[mp.GeometricObject] = [
        mp.Block(material=silicon_eff, center=mp.Vector3(0, 0), size=mp.Vector3(mp.inf, WAVEGUIDE_WIDTH_UM, mp.inf))
    ]
    if add_drop:
        objects.extend(
            [
                mp.Block(material=silicon_eff, center=mp.Vector3(0, drop_y), size=mp.Vector3(mp.inf, WAVEGUIDE_WIDTH_UM, mp.inf)),
                mp.Cylinder(material=silicon_eff, radius=outer_radius, center=mp.Vector3(0, ring_center_y), height=mp.inf),
                mp.Cylinder(material=air, radius=inner_radius, center=mp.Vector3(0, ring_center_y), height=mp.inf),
            ]
        )
    return objects, drop_y


def source(fcen: float, df: float) -> list[mp.Source]:
    return [
        mp.Source(
            mp.GaussianSource(fcen, fwidth=df),
            component=mp.Ez,
            center=mp.Vector3(-5.7, 0),
            size=mp.Vector3(0, 1.2),
        )
    ]


def run_one(radius_um: float, gap_um: float) -> dict[str, float]:
    fcen = 1.0 / LAMBDA_CENTER_UM
    df = 0.22 * fcen
    ref_sim = mp.Simulation(
        cell_size=CELL,
        boundary_layers=[mp.PML(1.0)],
        geometry=build_geometry(radius_um, gap_um, add_drop=False)[0],
        sources=source(fcen, df),
        resolution=RESOLUTION,
    )
    ref_mon = ref_sim.add_flux(fcen, df, NFREQ, mp.FluxRegion(center=mp.Vector3(5.7, 0), size=mp.Vector3(0, 1.4)))
    ref_sim.run(until_after_sources=mp.stop_when_fields_decayed(70, mp.Ez, mp.Vector3(5.7, 0), 1e-7))
    ref_flux = np.asarray(mp.get_fluxes(ref_mon))

    objects, drop_y = build_geometry(radius_um, gap_um, add_drop=True)
    sim = mp.Simulation(
        cell_size=CELL,
        boundary_layers=[mp.PML(1.0)],
        geometry=objects,
        sources=source(fcen, df),
        resolution=RESOLUTION,
    )
    through_mon = sim.add_flux(fcen, df, NFREQ, mp.FluxRegion(center=mp.Vector3(5.7, 0), size=mp.Vector3(0, 1.4)))
    drop_mon = sim.add_flux(fcen, df, NFREQ, mp.FluxRegion(center=mp.Vector3(5.7, drop_y), size=mp.Vector3(0, 1.4)))
    sim.run(until_after_sources=mp.stop_when_fields_decayed(90, mp.Ez, mp.Vector3(5.7, 0), 1e-7))

    freqs = np.asarray(mp.get_flux_freqs(through_mon))
    wavelengths = 1.0 / freqs
    through = np.asarray(mp.get_fluxes(through_mon)) / np.maximum(ref_flux, 1e-30)
    drop = np.asarray(mp.get_fluxes(drop_mon)) / np.maximum(ref_flux, 1e-30)

    through_idx = int(np.argmin(through))
    drop_idx = int(np.argmax(drop))
    return {
        "radius_um": float(radius_um),
        "gap_um": float(gap_um),
        "through_min": float(through[through_idx]),
        "through_min_wavelength_um": float(wavelengths[through_idx]),
        "drop_max": float(drop[drop_idx]),
        "drop_max_wavelength_um": float(wavelengths[drop_idx]),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    radius_values = [1.4, 1.6, 1.8]
    gap_values = [0.08, 0.10, 0.12]
    rows = []

    for radius_um in radius_values:
        for gap_um in gap_values:
            print(f"Running Meep candidate R={radius_um:.2f} um, gap={gap_um:.2f} um")
            rows.append(run_one(radius_um, gap_um))
            print(rows[-1])

    keys = list(rows[0].keys())
    with (OUTPUT_DIR / "meep_add_drop_sweep.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    plot_summary(rows)
    print(f"Saved sweep CSV and plot under: {OUTPUT_DIR}")


def plot_summary(rows: list[dict[str, float]]) -> None:
    labels = [f"R={r['radius_um']:.1f}\ng={r['gap_um']:.2f}" for r in rows]
    through = [r["through_min"] for r in rows]
    drop = [r["drop_max"] for r in rows]
    x = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x - 0.18, through, width=0.36, label="through min")
    ax.bar(x + 0.18, drop, width=0.36, label="drop max")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Normalized transmission")
    ax.set_title("Small Meep add-drop parameter sweep")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "meep_add_drop_sweep.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
