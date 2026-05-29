"""High-Q racetrack add-drop compact optimization.

This script optimizes geometry-like parameters:

- bend radius R
- straight coupling length Lc
- coupling gap

It uses a simple semi-empirical coupling model kappa(gap, Lc), then evaluates a
racetrack add-drop transfer function whose round-trip length is:

    L = 2*pi*R + 2*Lc

The model is intentionally fast and approximate; it is meant to generate
candidates for later Meep/FDTD validation, not replace FDTD.
"""

from __future__ import annotations

import csv
import json
import os
import pathlib

PROJECT_DIR = pathlib.Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_DIR / "outputs" / "racetrack_high_q_optimization"
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_DIR / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_DIR / ".cache"))

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution
from scipy.signal import find_peaks, peak_widths


TARGET_NM = 1550.0
LAMBDA_REF_NM = 1550.0
N_EFF = 2.4
N_G = 4.0
LOSS_DB_CM = 1.0
WAVELENGTHS_NM = np.linspace(1520.0, 1580.0, 24001)


def kappa_from_gap_length(gap_um: float, coupling_length_um: float) -> float:
    """Toy coupling model for racetrack directional couplers.

    Smaller gap and longer coupling length increase kappa. The sine saturation
    mimics directional-coupler power exchange, while the exponential gap factor
    mimics evanescent decay.
    """

    gap_ref = 0.20
    gap_decay = 0.075
    lc_pi = 8.0
    gap_factor = np.exp(-(gap_um - gap_ref) / gap_decay)
    length_factor = np.sin(0.5 * np.pi * np.clip(coupling_length_um / lc_pi, 0.0, 1.0))
    return float(np.clip(0.18 * gap_factor * length_factor, 0.005, 0.60))


def add_drop_racetrack_transfer(
    wavelength_nm: np.ndarray,
    radius_um: float,
    coupling_length_um: float,
    gap_um: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return through/drop fields plus kappa and round-trip length."""

    kappa = kappa_from_gap_length(gap_um, coupling_length_um)
    length_um = 2.0 * np.pi * radius_um + 2.0 * coupling_length_um
    length_cm = length_um * 1e-4

    alpha_power_cm = LOSS_DB_CM * np.log(10.0) / 10.0
    a = np.exp(-alpha_power_cm * length_cm / 2.0)
    t1 = np.sqrt(1.0 - kappa**2)
    t2 = t1

    dneff_dlambda = (N_EFF - N_G) / LAMBDA_REF_NM
    n_eff_lambda = N_EFF + dneff_dlambda * (wavelength_nm - LAMBDA_REF_NM)
    beta = 2.0 * np.pi * n_eff_lambda / wavelength_nm
    phi = beta * length_um * 1e3
    round_trip = a * np.exp(1j * phi)
    half_trip = np.sqrt(a) * np.exp(0.5j * phi)
    denominator = 1.0 - t1 * t2 * round_trip

    through = (t1 - t2 * round_trip) / denominator
    drop = (-kappa * kappa * half_trip) / denominator
    return through, drop, kappa, length_um


def evaluate(radius_um: float, coupling_length_um: float, gap_um: float) -> dict[str, float]:
    through_field, drop_field, kappa, length_um = add_drop_racetrack_transfer(
        WAVELENGTHS_NM, radius_um, coupling_length_um, gap_um
    )
    through = np.abs(through_field) ** 2
    drop = np.abs(drop_field) ** 2

    peak_idx = choose_peak(drop)
    peak_nm = float(WAVELENGTHS_NM[peak_idx])
    drop_peak = float(drop[peak_idx])
    through_at_peak = float(through[peak_idx])
    er_db = 10.0 * np.log10(max(np.percentile(through, 90), 1e-15) / max(through_at_peak, 1e-15))
    q_loaded = estimate_q(drop, peak_idx)
    fsr_nm = TARGET_NM**2 / (N_G * length_um * 1e3)
    detuning_nm = abs(peak_nm - TARGET_NM)

    q_target = 10_000.0
    q_penalty = max(0.0, q_target - q_loaded) / 2_000.0
    detuning_penalty = detuning_nm / 2.0
    weak_drop_penalty = max(0.0, 0.25 - drop_peak) * 4.0

    score = (
        1.8 * drop_peak
        + 1.2 * (1.0 - through_at_peak)
        + 0.08 * min(er_db, 35.0)
        + 0.00008 * min(q_loaded, 80_000.0)
        - q_penalty
        - detuning_penalty
        - weak_drop_penalty
    )

    return {
        "score": float(score),
        "radius_um": float(radius_um),
        "coupling_length_um": float(coupling_length_um),
        "gap_um": float(gap_um),
        "kappa": float(kappa),
        "round_trip_length_um": float(length_um),
        "peak_nm": peak_nm,
        "detuning_nm": float(detuning_nm),
        "drop_peak": drop_peak,
        "through_at_peak": through_at_peak,
        "er_db": float(er_db),
        "q_loaded": float(q_loaded),
        "fsr_nm": float(fsr_nm),
    }


def choose_peak(drop: np.ndarray) -> int:
    peaks, props = find_peaks(drop, prominence=max(1e-5, 0.02 * (np.max(drop) - np.min(drop))))
    if len(peaks) == 0:
        return int(np.argmax(drop))
    scores = drop[peaks] + 0.15 * props["prominences"] - 0.01 * np.abs(WAVELENGTHS_NM[peaks] - TARGET_NM)
    return int(peaks[int(np.argmax(scores))])


def estimate_q(drop: np.ndarray, peak_idx: int) -> float:
    try:
        width_samples = peak_widths(drop, [peak_idx], rel_height=0.5)[0][0]
        fwhm_nm = max(float(width_samples) * float(np.mean(np.diff(WAVELENGTHS_NM))), 1e-6)
        return float(WAVELENGTHS_NM[peak_idx] / fwhm_nm)
    except Exception:
        return float("nan")


def objective(x: np.ndarray) -> float:
    metrics = evaluate(float(x[0]), float(x[1]), float(x[2]))
    return -metrics["score"]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float]] = []

    def callback(xk: np.ndarray, convergence: float) -> bool:
        row = evaluate(float(xk[0]), float(xk[1]), float(xk[2]))
        row["convergence"] = float(convergence)
        history.append(row)
        print(
            f"iter={len(history):02d} score={row['score']:.3f} "
            f"R={row['radius_um']:.2f} Lc={row['coupling_length_um']:.2f} gap={row['gap_um']:.3f} "
            f"k={row['kappa']:.3f} peak={row['peak_nm']:.2f} Q={row['q_loaded']:.0f} "
            f"drop={row['drop_peak']:.3f} through={row['through_at_peak']:.3f}"
        )
        return False

    result = differential_evolution(
        objective,
        bounds=[(5.0, 30.0), (0.5, 10.0), (0.12, 0.35)],
        seed=11,
        maxiter=35,
        popsize=12,
        polish=True,
        callback=callback,
        workers=1,
    )
    best = evaluate(float(result.x[0]), float(result.x[1]), float(result.x[2]))
    write_outputs(best, history, float(result.fun))
    plot_best(best)
    plot_history(history)
    print("High-Q racetrack compact optimization complete.")
    print(json.dumps(best, indent=2))
    print(f"Saved outputs under: {OUTPUT_DIR}")


def write_outputs(best: dict[str, float], history: list[dict[str, float]], fun: float) -> None:
    payload = {
        "optimizer": "scipy.optimize.differential_evolution",
        "objective": "high-Q racetrack add-drop compact optimization",
        "target_nm": TARGET_NM,
        "n_eff": N_EFF,
        "n_g": N_G,
        "loss_db_cm": LOSS_DB_CM,
        "best": best,
        "minimized_objective": fun,
    }
    (OUTPUT_DIR / "best_racetrack_design.json").write_text(json.dumps(payload, indent=2))
    if history:
        keys = list(history[0].keys())
        with (OUTPUT_DIR / "racetrack_optimization_history.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(history)


def plot_best(best: dict[str, float]) -> None:
    through_field, drop_field, _, _ = add_drop_racetrack_transfer(
        WAVELENGTHS_NM, best["radius_um"], best["coupling_length_um"], best["gap_um"]
    )
    through = np.abs(through_field) ** 2
    drop = np.abs(drop_field) ** 2

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.plot(WAVELENGTHS_NM, through, label="through port")
    ax.plot(WAVELENGTHS_NM, drop, label="drop port")
    ax.axvline(TARGET_NM, color="black", linestyle="--", linewidth=1.0, label="target")
    ax.scatter([best["peak_nm"]], [best["drop_peak"]], color="crimson", zorder=3)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Transmission")
    ax.set_title("Optimized high-Q racetrack add-drop compact model")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "best_racetrack_spectrum.png", dpi=220)
    plt.close(fig)


def plot_history(history: list[dict[str, float]]) -> None:
    if not history:
        return
    steps = np.arange(1, len(history) + 1)
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 6.6), sharex=True)
    axes[0].plot(steps, [h["score"] for h in history], marker="o")
    axes[0].set_ylabel("Score")
    axes[1].plot(steps, [h["q_loaded"] for h in history], marker="o", color="#2ca02c")
    axes[1].set_ylabel("Loaded Q")
    axes[2].plot(steps, [h["detuning_nm"] for h in history], marker="o", color="#d62728")
    axes[2].set_ylabel("Detuning (nm)")
    axes[2].set_xlabel("Iteration")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "racetrack_optimization_history.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
