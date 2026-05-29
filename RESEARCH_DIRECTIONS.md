# Research Directions After the Racetrack Add-Drop Demo

## 1. Curvature / Bend Radius Study

Goal: quantify how local curvature affects mode confinement, bend radiation,
through/drop spectra, loaded Q, and coupling efficiency. For racetrack rings,
the curved sides do not need to be perfect semicircles. They can be circular
arcs, Euler/clothoid bends, Bezier curves, spline curves, or other adiabatic
curvature profiles.

Suggested workflow:

1. Keep waveguide width, gap, and coupling length fixed.
2. Sweep bend radius, for example `R = 1.2, 1.5, 2, 3, 5, 10 um`.
3. For each radius, run:
   - compact model estimate of FSR and Q
   - 2D effective-index Meep FDTD for field maps and spectra
   - optional directional-coupler extraction for coupling efficiency
4. Extract metrics:
   - resonance wavelength
   - FSR
   - through extinction
   - drop efficiency
   - loaded Q
   - field leakage around bends
   - bend-loss proxy from power outside waveguide region

Interpretation:

- Smaller radius usually means larger FSR and smaller footprint.
- Smaller radius also increases bend radiation and mode distortion.
- Racetrack geometry separates bend radius from coupling length, which is why
  it is more controllable than a pure circular ring.
- Non-circular bends can reduce radiation and transition loss by avoiding an
  abrupt curvature jump between straight and curved sections.
- Euler/clothoid bends use gradually varying curvature. Bezier/spline bends can
  be optimized numerically for compactness, mode purity, and low bend loss.

Useful references:

- Bogaerts et al., "Silicon microring resonators", Laser & Photonics Reviews,
  2012.
- Ansys ring resonator parameter extraction tutorial.
- Ultra-high-Q racetrack microring work using modified Euler bends.
- "Racetrack microring resonator with improved quality factor based on
  asymmetric waveguide bend", Optics Communications, 2022. Uses asymmetric
  Bezier waveguide bends to improve Q in compact racetrack resonators.
- "Ultrahigh-Q silicon racetrack resonators", Photonics Research, 2020. Uses
  multimode waveguide bends based on modified Euler curves.
- "Analysis of silicon nitride partial Euler waveguide bends", arXiv:1910.07257.
  Discusses partial Euler bends for reducing bend loss at small effective radii.
- "Ultra-high-Q racetrack microring based on silicon-nitride", arXiv:2209.01097.
  Uses modified Euler curves and directional coupler design for ultra-high-Q
  racetrack resonators.

## 2. Thermal Effects

Goal: add temperature-dependent resonance shift and optional heater tuning.

Compact model:

```text
Delta lambda_res / Delta T ~= lambda_res / n_g * (dn_eff/dT + n_eff * alpha_expansion)
```

For silicon near telecom wavelengths, the thermo-optic term is usually much
larger than thermal expansion, so a first model can use:

```text
Delta n_eff ~= Gamma_Si * (dn_Si/dT) * Delta T
```

Simulation levels:

1. Compact model:
   - sweep temperature
   - shift resonance
   - compute detuning penalty, ER, OMA, insertion loss
2. FDTD approximation:
   - rerun Meep with `n_eff(T) = n_eff0 + dneff_dT * DeltaT`
   - compare field and spectrum before/after thermal shift
3. Multiphysics:
   - solve temperature distribution using FEM or thermal solver
   - feed spatially varying index back into optical simulation

Extra effects to include later:

- free-carrier absorption
- two-photon absorption
- self-heating
- thermal crosstalk between nearby rings
- fabrication variation in width, gap, and thickness

Useful references:

- Thermal effect analysis of silicon microring optical switch for on-chip
  interconnect.
- Photonic and thermal modelling of microrings in silicon, diamond and GaN for
  temperature sensing.
- Wide-range and fast thermally tunable silicon photonic microring resonators.
- Thermo-optic multistability in silicon microring resonators with lateral
  diodes.

## 3. Toward 3D Simulation

Goal: move from a 2D effective-index approximation to geometry closer to SOI.

3D geometry:

```text
Si core: height ~220 nm
waveguide width ~450 nm
SiO2 lower cladding/substrate
air or oxide upper cladding
finite-height bus and ring waveguides
```

Recommended steps:

1. Use a 2D or 3D mode solver first to get the TE0 mode.
2. Use an eigenmode source instead of a raw `Ez` line source.
3. Use mode-expansion monitors for cleaner through/drop coupling metrics.
4. Start with a straight waveguide and directional coupler before full ring.
5. Use symmetry and reduced spans where possible.
6. Validate against compact model:
   - `n_eff`
   - `n_g`
   - propagation loss
   - coupling coefficient
   - Q and FSR

Warnings:

- 3D FDTD is much slower than 2D effective-index FDTD.
- High-Q rings require long simulation times.
- Mesh resolution near sub-100-nm gaps strongly affects coupling.
- Material dispersion and mode-source setup matter.

Useful references:

- Ansys ring resonator tutorial, especially 3D FDTD parameter extraction.
- Meep documentation on Python FDTD setup, flux spectra, PML, and scaling.
- Silicon photonics review literature on silicon wire waveguides and tight bend
  radii.
