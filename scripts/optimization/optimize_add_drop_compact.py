"""Optimize an add-drop microring using the fast compact model.

This is the first-pass optimizer before expensive FDTD validation. It uses
SciPy differential evolution to tune:

- ring radius
- input-bus coupling coefficient kappa1
- drop-bus coupling coefficient kappa2

The objective favors a drop-port peak near 1550 nm, a through-port dip at the
same wavelength, a useful extinction ratio, and a reasonable loaded Q.
"""

from __future__ import annotations

import csv
import json
import os
import pathlib
import sys

PROJECT_DIR = pathlib.Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_DIR / "src"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "add_drop_optimization"
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_DIR / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_DIR / ".cache"))
sys.path.insert(0, str(SRC_DIR))

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution
from scipy.signal import find_peaks, peak_widths

from microring_model import add_drop_transfer, fsr_nm


TARGET_NM = 1550.0
N_EFF = 2.4
N_G = 4.0
LOSS_DB_CM = 2.0
LAMBDA_REF_NM = 1550.0
WAVELENGTHS_NM = np.linspace(1500.0, 1600.0, 16001)


def evaluate_design(radius_um: float, kappa1: float, kappa2: float) -> dict[str, float]:
    """Return metrics and objective components for one add-drop design."""

    through_field, drop_field = add_drop_transfer(
        WAVELENGTHS_NM,
        radius_um=radius_um,
        lambda_ref_nm=LAMBDA_REF_NM,
        n_eff=N_EFF,
        n_g=N_G,
        loss_db_per_cm=LOSS_DB_CM,
        kappa1=kappa1,
        kappa2=kappa2,
    )
    through = np.abs(through_field) ** 2
    drop = np.abs(drop_field) ** 2

    peak_idx = choose_drop_peak(drop)
    peak_nm = float(WAVELENGTHS_NM[peak_idx])
    drop_peak = float(drop[peak_idx])
    through_at_peak = float(through[peak_idx])
    through_baseline = float(np.percentile(through, 90))
    er_db = 10.0 * np.log10(max(through_baseline, 1e-15) / max(through_at_peak, 1e-15))
    q_loaded = estimate_q_from_peak(drop, peak_idx)
    analytic_fsr = fsr_nm(TARGET_NM, radius_um, N_G)

    detuning_nm = abs(peak_nm - TARGET_NM)
    q_penalty = soft_range_penalty(q_loaded, low=2_000.0, high=80_000.0, scale=1_000.0)
    fsr_penalty = max(0.0, 20.0 - analytic_fsr) / 20.0
    er_reward = min(er_db, 35.0)

    score = (
        3.0 * drop_peak
        + 1.5 * (1.0 - through_at_peak)
        + 0.10 * er_reward
        - 0.25 * detuning_nm
        - q_penalty
        - fsr_penalty
    )

    return {
        "score": float(score),
        "radius_um": float(radius_um),
        "kappa1": float(kappa1),
        "kappa2": float(kappa2),
        "peak_nm": peak_nm,
        "detuning_nm": float(detuning_nm),
        "drop_peak": drop_peak,
        "through_at_peak": through_at_peak,
        "through_baseline": through_baseline,
        "er_db": float(er_db),
        "q_loaded": float(q_loaded),
        "analytic_fsr_nm": float(analytic_fsr),
    }


def choose_drop_peak(drop: np.ndarray) -> int:
    peaks, props = find_peaks(drop, prominence=max(1e-5, 0.01 * (np.max(drop) - np.min(drop))))
    if len(peaks) == 0:
        return int(np.argmax(drop))

    peak_scores = []
    for idx, peak in enumerate(peaks):
        detuning = abs(WAVELENGTHS_NM[peak] - TARGET_NM)
        prominence = props["prominences"][idx]
        peak_scores.append(drop[peak] + 0.2 * prominence - 0.01 * detuning)
    return int(peaks[int(np.argmax(peak_scores))])


def estimate_q_from_peak(drop: np.ndarray, peak_idx: int) -> float:
    try:
        widths = peak_widths(drop, [peak_idx], rel_height=0.5)[0]
        fwhm_nm = max(float(widths[0]) * float(np.mean(np.diff(WAVELENGTHS_NM))), 1e-6)
        return float(WAVELENGTHS_NM[peak_idx] / fwhm_nm)
    except Exception:
        return float("nan")


def soft_range_penalty(value: float, low: float, high: float, scale: float) -> float:
    if not np.isfinite(value):
        return 3.0
    if value < low:
        return (low - value) / scale
    if value > high:
        return (value - high) / scale
    return 0.0


def objective(x: np.ndarray) -> float:
    radius_um, kappa1, kappa2 = [float(v) for v in x]
    metrics = evaluate_design(radius_um, kappa1, kappa2)
    return -metrics["score"]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float]] = []

    def callback(xk: np.ndarray, convergence: float) -> bool:
        metrics = evaluate_design(float(xk[0]), float(xk[1]), float(xk[2]))
        metrics["convergence"] = float(convergence)
        history.append(metrics)
        print(
            f"iter={len(history):02d} score={metrics['score']:.3f} "
            f"R={metrics['radius_um']:.3f} k1={metrics['kappa1']:.3f} k2={metrics['kappa2']:.3f} "
            f"peak={metrics['peak_nm']:.2f} nm drop={metrics['drop_peak']:.3f} "
            f"through={metrics['through_at_peak']:.3f} ER={metrics['er_db']:.2f} dB"
        )
        return False

    bounds = [
        (3.0, 15.0),  # radius_um
        (0.02, 0.50),  # kappa1
        (0.02, 0.50),  # kappa2
    ]
    result = differential_evolution(
        objective,
        bounds=bounds,
        seed=7,
        maxiter=30,
        popsize=12,
        polish=True,
        callback=callback,
        workers=1,
        updating="immediate",
    )

    best = evaluate_design(float(result.x[0]), float(result.x[1]), float(result.x[2]))
    write_summary(best, result.fun)
    write_history(history)
    plot_best_spectrum(best)
    plot_history(history)

    print("Compact add-drop optimization complete.")
    print(json.dumps(best, indent=2))
    print(f"Saved outputs under: {OUTPUT_DIR}")


def write_summary(best: dict[str, float], fun: float) -> None:
    payload = {
        "optimizer": "scipy.optimize.differential_evolution",
        "objective": "maximize drop peak near 1550 nm, through extinction, ER; penalize detuning/Q/low FSR",
        "target_nm": TARGET_NM,
        "n_eff": N_EFF,
        "n_g": N_G,
        "loss_db_cm": LOSS_DB_CM,
        "best": best,
        "minimized_objective": float(fun),
    }
    (OUTPUT_DIR / "best_design.json").write_text(json.dumps(payload, indent=2))


def write_history(history: list[dict[str, float]]) -> None:
    if not history:
        return
    keys = list(history[0].keys())
    with (OUTPUT_DIR / "optimization_history.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(history)


def plot_best_spectrum(best: dict[str, float]) -> None:
    through_field, drop_field = add_drop_transfer(
        WAVELENGTHS_NM,
        radius_um=best["radius_um"],
        lambda_ref_nm=LAMBDA_REF_NM,
        n_eff=N_EFF,
        n_g=N_G,
        loss_db_per_cm=LOSS_DB_CM,
        kappa1=best["kappa1"],
        kappa2=best["kappa2"],
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
    ax.set_title("Optimized compact add-drop microring")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "best_compact_spectrum.png", dpi=220)
    plt.close(fig)


def plot_history(history: list[dict[str, float]]) -> None:
    if not history:
        return
    steps = np.arange(1, len(history) + 1)
    scores = np.asarray([row["score"] for row in history])
    detuning = np.asarray([row["detuning_nm"] for row in history])

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.2), sharex=True)
    axes[0].plot(steps, scores, marker="o", linewidth=1.2)
    axes[0].set_ylabel("Score")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(steps, detuning, marker="o", linewidth=1.2, color="#d62728")
    axes[1].set_xlabel("Optimizer iteration")
    axes[1].set_ylabel("|peak - target| (nm)")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "optimization_history.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
