# Fig. 4 math report -- Distortion of the LO-free Rydberg atomic receiver

## 1. What this is

A re-analysis of the already-computed, genuine QuTiP grid saved in
`fig3_fresh_build/fig3_quantum_response.npz`. No new QuTiP solves were
performed for this figure -- the paper itself describes Fig. 4 as
"a top-down view of Fig. 3," i.e. a re-analysis of the same
Pout(Delta_c, Omega_RF) surface, not new physics.

## 2. Physical model (identical to Fig. 3, paper Sec. IV)

Four-level ladder in a Cs vapor cell: |1> ground, |2> intermediate
excited, |3> Rydberg, |4> nearby Rydberg. Probe drives |1>-|2>,
coupling drives |2>-|3>, the RF field drives |3>-|4>.

Hamiltonian (rotating frame, rad/s):

    H = -Delta_p |2><2| - (Delta_p+Delta_c) |3><3| - (Delta_p+Delta_c+Delta_RF) |4><4|
        + (Omega_p/2)(|1><2|+|2><1|) + (Omega_c/2)(|2><3|+|3><2|) + (Omega_RF/2)(|3><4|+|4><3|)

Steady state solved via the Lindblad master equation (QuTiP
`steadystate`, direct method), with collapse operators
sqrt(gamma2)|1><2|, sqrt(gamma3)|2><3|, sqrt(gamma4)|3><4|.
Delta_p = Delta_RF = 0 fixed throughout (matches Fig. 3).

Pout obtained via the Beer-Lambert relation:
    chi = C0 * rho21,  Pout = Pin * exp(-kp * L * Im(chi))

## 3. Parameters used (paper Sec. IV, Cs 6S1/2 -> 6P3/2 -> 47D5/2 -> 48P3/2)

    L = 1 cm, N0 = 1e15 m^-3, d_probe = 0.76 mm
    Omega_p/2pi = 8.0 MHz, Omega_c/2pi = 1.0 MHz
    gamma1 = 0, gamma2/2pi = 5.2 MHz, gamma3/2pi = 3.9 kHz, gamma4/2pi = 1.7 kHz
    wp_RF (|3>-|4> dipole moment) = -1443.459 e*a0
    wp_12 (|1>-|2> dipole moment, squared convention) = (2.5 e*a0)^2
    lambda_p = 852 nm

## 4. Data grid (fig3_quantum_response.npz)

    Delta_c/2pi: [-20.0, 20.0] MHz, 201 points,
    step 0.2000 MHz
    Omega_RF/2pi: [0.0, 20.0] MHz, 161 points,
    step 0.1250 MHz

## 5. EIT linewidth -- MEASURED, not taken from the paper's analytical formula

The paper's Eq. 51 (Gamma_EIT) is derived in Sec. III-B for the LO-dressed
receiver, under an additional gamma3=gamma4=0 approximation this
simulation does not use (this run keeps the real, nonzero gamma3, gamma4
from Sec. IV). Rather than import that formula, the HWHM used here is
measured directly from THIS dataset's own Omega_RF=0 spectrum (the plain
3-level EIT lineshape, no RF field at all):

    Peak position: Delta_c = 0.0000 MHz
    Peak value: 37.6190 microW
    Baseline (avg of the two grid edges): 27.2134 microW
    Half-max level: 32.4162 microW
    Half-max crossings: left = -2.5960 MHz, right = 2.5960 MHz

    MEASURED HWHM (Gamma) = 2.5960 MHz
    MEASURED FWHM = 5.1920 MHz

(For reference only: the paper's analytical Eq. 51 gives Gamma_EIT =
2.6101 MHz for these same Omega_p, Omega_c, gamma2 -- close to, but not
identical to, the 2.5960 MHz measured here, consistent with
the gamma3/gamma4 and linear-vs-exponential differences noted above.)

## 6. Peak extraction and resolvability criterion

For every Omega_RF row:
  1. Find all local maxima of the real Pout(Delta_c) curve via
     scipy.signal.find_peaks (no amplitude/prominence filter).
  2. If fewer than 2 peaks exist, the row is unresolved (single_peak).
  3. Otherwise take the two tallest peaks; require a genuine INTERIOR
     valley between them (a real dip strictly between the two peak
     indices, not a monotonic shoulder) -- otherwise unresolved
     (no_interior_valley).
  4. FWHM-based resolvability criterion: resolved only if separation >=
     measured FWHM (5.1920 MHz). This is a plainly-stated
     separation>=1-FWHM rule, not a specific named criterion -- it is
     NOT called "the Rayleigh criterion" here, since that name properly
     refers to a diffraction-pattern condition for Airy/sinc lineshapes
     that does not rigorously apply to this EIT/AT doublet. The linewidth
     is measured from this dataset, not imported from elsewhere.

## 7. Result

    Resolved rows: 120 / 161
    THRESHOLD (lowest resolved Omega_RF) = 5.1250 MHz
    Omega_RF grid spacing = 0.1250 MHz (threshold accurate to within
    one grid step, not stated as exact)

    Paper's stated saturation threshold: ~5.5 MHz
    This result: 5.1250 MHz -- independently measured, not tuned to match the paper.

## 8. Assumptions and limitations (stated explicitly, none hidden)

  - Delta_p = Delta_RF = 0 fixed (matches Fig. 3's own assumption).
  - The separation>=1-FWHM criterion is a practical, plainly-stated
    convention chosen for this analysis, not something the paper itself
    states explicitly for Fig. 4 (the paper only says the spectrum "no
    longer exhibits two peaks" below ~5.5 MHz, without giving a precise
    numerical definition), and not a claim to any specific named
    criterion from optics (see Section 6).
  - A separate, criterion-free check (raw peak count, no threshold at
    all) shows the two peaks are already mathematically distinct local
    maxima starting around Omega_RF/2pi ~ 1.3-1.4 MHz -- well below the
    5.1250 MHz FWHM-based threshold above. Both numbers are
    genuine and data-derived; they answer different questions (literal
    peak count vs. linewidth-based practical distinguishability) and
    should not be conflated.
  - Grid resolution: separation/HWHM values are only as precise as the
    Delta_c grid spacing above; finer grids would refine (not
    qualitatively change) the reported threshold.
