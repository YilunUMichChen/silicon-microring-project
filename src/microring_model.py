"""Compact models and metrics for silicon microring resonators.

This module intentionally uses only NumPy so it can run in a lightweight
environment before connecting to Lumerical or a full photonics design kit.
All wavelength values are in nm unless otherwise stated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


C_NM_PER_S = 2.99792458e17


@dataclass(frozen=True)
class RingParams:
    """Basic parameters for a single-bus all-pass microring."""

    radius_um: float = 10.0
    lambda_ref_nm: float = 1550.0
    n_eff: float = 2.4
    n_g: float = 4.0
    loss_db_per_cm: float = 2.0
    kappa: float = 0.25


@dataclass(frozen=True)
class ResonanceMetrics:
    """Extracted figures of merit around one resonance dip."""

    resonance_nm: float
    fwhm_nm: float
    q_loaded: float
    extinction_ratio_db: float
    min_transmission: float
    baseline_transmission: float


def round_trip_length_um(radius_um: float) -> float:
    """Return the ring round-trip length in um."""

    _require_positive(radius_um, "radius_um")
    return 2.0 * np.pi * radius_um


def round_trip_length_cm(radius_um: float) -> float:
    """Return the ring round-trip length in cm."""

    return round_trip_length_um(radius_um) * 1e-4


def fsr_nm(lambda_nm: float, radius_um: float, n_g: float) -> float:
    """Approximate wavelength-domain free spectral range.

    FSR_lambda ~= lambda^2 / (n_g * L), where L is the round-trip length.
    The calculation is unit-consistent by converting L from um to nm.
    """

    _require_positive(lambda_nm, "lambda_nm")
    _require_positive(n_g, "n_g")
    length_nm = round_trip_length_um(radius_um) * 1e3
    return lambda_nm**2 / (n_g * length_nm)


def loss_dbcm_to_round_trip_amplitude(loss_db_per_cm: float, radius_um: float) -> float:
    """Convert power loss in dB/cm to one-round-trip field transmission.

    If optical power follows P_out/P_in = exp(-alpha_power * L), then field
    amplitude follows a = exp(-alpha_power * L / 2).
    """

    if loss_db_per_cm < 0:
        raise ValueError("loss_db_per_cm must be non-negative")
    alpha_power_per_cm = loss_db_per_cm * np.log(10.0) / 10.0
    return float(np.exp(-alpha_power_per_cm * round_trip_length_cm(radius_um) / 2.0))


def all_pass_transfer(lambda_nm: np.ndarray, params: RingParams) -> np.ndarray:
    """Return complex through-port field E_out/E_in for an all-pass ring."""

    wavelength = _as_wavelength_array(lambda_nm)
    _validate_kappa(params.kappa)
    _require_positive(params.lambda_ref_nm, "lambda_ref_nm")
    a = loss_dbcm_to_round_trip_amplitude(params.loss_db_per_cm, params.radius_um)
    t = np.sqrt(1.0 - params.kappa**2)
    beta = _beta_with_linear_dispersion(wavelength, params.lambda_ref_nm, params.n_eff, params.n_g)
    phi = beta * round_trip_length_um(params.radius_um) * 1e3
    g = a * np.exp(1j * phi)
    return (t - g) / (1.0 - t * g)


def all_pass_response(lambda_nm: np.ndarray, params: RingParams) -> tuple[np.ndarray, np.ndarray]:
    """Return intensity transmission and unwrapped phase."""

    field = all_pass_transfer(lambda_nm, params)
    return np.abs(field) ** 2, np.unwrap(np.angle(field))


def add_drop_transfer(
    lambda_nm: np.ndarray,
    radius_um: float,
    lambda_ref_nm: float,
    n_eff: float,
    n_g: float,
    loss_db_per_cm: float,
    kappa1: float,
    kappa2: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return through/drop complex fields for a two-bus add-drop ring.

    The sign convention is chosen for intensity analysis; phase references can
    be adjusted later if matching a specific simulator or PDK compact model.
    """

    wavelength = _as_wavelength_array(lambda_nm)
    _require_positive(lambda_ref_nm, "lambda_ref_nm")
    _validate_kappa(kappa1, "kappa1")
    _validate_kappa(kappa2, "kappa2")

    a = loss_dbcm_to_round_trip_amplitude(loss_db_per_cm, radius_um)
    t1 = np.sqrt(1.0 - kappa1**2)
    t2 = np.sqrt(1.0 - kappa2**2)
    beta = _beta_with_linear_dispersion(wavelength, lambda_ref_nm, n_eff, n_g)
    phi = beta * round_trip_length_um(radius_um) * 1e3
    half_trip = np.sqrt(a) * np.exp(0.5j * phi)
    round_trip = a * np.exp(1j * phi)
    denominator = 1.0 - t1 * t2 * round_trip

    through = (t1 - t2 * round_trip) / denominator
    drop = (-kappa1 * kappa2 * half_trip) / denominator
    return through, drop


def find_resonance_dips(lambda_nm: np.ndarray, transmission: np.ndarray) -> np.ndarray:
    """Return indices of local transmission minima, sorted by wavelength."""

    wavelength = _as_wavelength_array(lambda_nm)
    intensity = _as_intensity_array(transmission)
    if len(wavelength) != len(intensity):
        raise ValueError("lambda_nm and transmission must have the same length")

    candidates = np.where((intensity[1:-1] < intensity[:-2]) & (intensity[1:-1] <= intensity[2:]))[0] + 1
    if candidates.size == 0:
        return candidates

    span = float(np.nanmax(intensity) - np.nanmin(intensity))
    if span <= 0:
        return np.array([], dtype=int)

    baseline = float(np.nanpercentile(intensity, 90))
    depth = baseline - intensity[candidates]
    return candidates[depth > 0.05 * span]


def extract_fsr_from_dips(lambda_nm: np.ndarray, dip_indices: np.ndarray) -> float | None:
    """Estimate FSR from adjacent resonance dips."""

    wavelength = _as_wavelength_array(lambda_nm)
    if len(dip_indices) < 2:
        return None
    spacings = np.diff(wavelength[np.asarray(dip_indices, dtype=int)])
    return float(np.median(spacings))


def extract_resonance_metrics(
    lambda_nm: np.ndarray,
    transmission: np.ndarray,
    resonance_index: int | None = None,
) -> ResonanceMetrics:
    """Extract resonance wavelength, FWHM, loaded Q, and extinction ratio."""

    wavelength = _as_wavelength_array(lambda_nm)
    intensity = _as_intensity_array(transmission)
    if len(wavelength) != len(intensity):
        raise ValueError("lambda_nm and transmission must have the same length")

    if resonance_index is None:
        dips = find_resonance_dips(wavelength, intensity)
        if len(dips) == 0:
            resonance_index = int(np.argmin(intensity))
        else:
            resonance_index = int(dips[np.argmin(intensity[dips])])

    min_t = float(intensity[resonance_index])
    baseline_t = float(np.nanpercentile(intensity, 90))
    half_level = min_t + 0.5 * (baseline_t - min_t)

    left_nm = _half_width_crossing(wavelength, intensity, resonance_index, half_level, direction=-1)
    right_nm = _half_width_crossing(wavelength, intensity, resonance_index, half_level, direction=1)
    if left_nm is None or right_nm is None or right_nm <= left_nm:
        raise ValueError("Could not determine FWHM; increase wavelength resolution or scan span")

    fwhm = right_nm - left_nm
    resonance_nm = float(wavelength[resonance_index])
    q_loaded = resonance_nm / fwhm
    extinction_ratio_db = 10.0 * np.log10(max(baseline_t, 1e-15) / max(min_t, 1e-15))

    return ResonanceMetrics(
        resonance_nm=resonance_nm,
        fwhm_nm=float(fwhm),
        q_loaded=float(q_loaded),
        extinction_ratio_db=float(extinction_ratio_db),
        min_transmission=min_t,
        baseline_transmission=baseline_t,
    )


def resonance_shift_nm(lambda_res_nm: float, delta_neff: float, n_g: float) -> float:
    """First-order resonance wavelength shift from effective-index change."""

    _require_positive(lambda_res_nm, "lambda_res_nm")
    _require_positive(n_g, "n_g")
    return lambda_res_nm * delta_neff / n_g


def _beta_with_linear_dispersion(
    wavelength_nm: np.ndarray,
    lambda_ref_nm: float,
    n_eff_ref: float,
    n_g_ref: float,
) -> np.ndarray:
    """Return beta using a first-order n_eff(lambda) around lambda_ref.

    n_g = n_eff - lambda * d(n_eff)/d(lambda), so this keeps n_eff and n_g
    separate while preserving the expected local FSR.
    """

    _require_positive(n_eff_ref, "n_eff_ref")
    _require_positive(n_g_ref, "n_g_ref")
    dneff_dlambda = (n_eff_ref - n_g_ref) / lambda_ref_nm
    n_eff_lambda = n_eff_ref + dneff_dlambda * (wavelength_nm - lambda_ref_nm)
    return 2.0 * np.pi * n_eff_lambda / wavelength_nm


def _half_width_crossing(
    wavelength: np.ndarray,
    intensity: np.ndarray,
    start_idx: int,
    level: float,
    direction: int,
) -> float | None:
    idx_range = range(start_idx, 0, -1) if direction < 0 else range(start_idx, len(intensity) - 1)
    for idx in idx_range:
        i0, i1 = (idx - 1, idx) if direction < 0 else (idx, idx + 1)
        y0 = intensity[i0] - level
        y1 = intensity[i1] - level
        if y0 == 0:
            return float(wavelength[i0])
        if y0 * y1 <= 0 and y0 != y1:
            frac = -y0 / (y1 - y0)
            return float(wavelength[i0] + frac * (wavelength[i1] - wavelength[i0]))
    return None


def _as_wavelength_array(lambda_nm: np.ndarray) -> np.ndarray:
    wavelength = np.asarray(lambda_nm, dtype=float)
    if wavelength.ndim != 1:
        raise ValueError("lambda_nm must be a 1D array")
    if np.any(wavelength <= 0):
        raise ValueError("wavelength values must be positive")
    return wavelength


def _as_intensity_array(transmission: np.ndarray) -> np.ndarray:
    intensity = np.asarray(transmission, dtype=float)
    if intensity.ndim != 1:
        raise ValueError("transmission must be a 1D array")
    return intensity


def _validate_kappa(kappa: float, name: str = "kappa") -> None:
    if not 0.0 <= kappa < 1.0:
        raise ValueError(f"{name} must satisfy 0 <= {name} < 1")


def _require_positive(value: float, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")
