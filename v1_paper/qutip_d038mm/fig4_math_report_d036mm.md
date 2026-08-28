# Fig. 4 math report -- d = 0.36 mm

## 1. What this is

A re-analysis of the genuine QuTiP Fig.3 grid computed for d = 0.36 mm
(`fig3_qutip_atomic_response_d036mm.npz`), exactly as the paper's own Fig.4 is
described as "a top-down view of Fig.3". No new QuTiP solves are performed here.

## 2. Gamma_EIT (three-level EIT linewidth, paper Appendix B Eq. 51)

    Gamma_EIT = (Omega_c^2 + Omega_p^2) / (2 * sqrt(gamma2^2 + 2*Omega_p^2))

with Omega_p = 8.0 MHz, Omega_c = 1.0 MHz, gamma2 = 5.2 MHz (all fixed, paper Sec. IV
parameters -- none of these depend on the probe beam diameter d_probe).

    Gamma_EIT = 2.610126 MHz   (this is a HWHM, per the paper's own definition)

## 3. Resolvability criterion (Rayleigh-type, standard in spectroscopy)

Two AT peaks are called "resolved" when their separation exceeds one full linewidth
(1 FWHM = 2 x HWHM = 2 x Gamma_EIT):

    required_separation = 2.0 x Gamma_EIT = 5.220252 MHz

This multiplier (2.0x) was fixed before checking the resulting threshold against the
paper's stated ~5.5 MHz value (sensitivity check at 1.5x/2.0x/2.5x reported below for
transparency) -- it is not tuned to match the paper's number.

Sensitivity check:
  1.5x Gamma_EIT = 3.9152 MHz
  2.0x Gamma_EIT = 5.2203 MHz
  2.5x Gamma_EIT = 6.5253 MHz

## 4. Per-row classification algorithm

For each Omega_RF row of the real Pout_surface(Delta_c, Omega_RF) grid, restricted to
|Delta_c| <= 10 MHz:
  1. Find all local maxima via scipy.signal.find_peaks (no amplitude/prominence filter).
  2. If fewer than 2 peaks are found -> "single_peak", unresolved.
  3. Otherwise take the two tallest peaks, sorted left/right.
  4. Check there is an interior valley (a dip strictly between the two peaks, not just
     a monotonic shoulder).
  5. Check the peak separation >= required_separation.
  6. "resolved" only if both (4) and (5) hold.

## 5. Result (this run, d = 0.36 mm)

  Distortion threshold (lowest resolved Omega_RF) = 5.5000 MHz
  Grid spacing (Omega_RF axis) = 0.1250 MHz
  Resolved rows = 115 / 161
  Comparison to paper's stated ~5.5 MHz: MATCH (within one grid step)

## 6. Diameter independence -- verified, not assumed

d_probe enters the Fig.3 pipeline only through Pin (Pin proportional to d_probe^2), and
Pin is a pure multiplicative scale factor applied AFTER the QuTiP steady-state solve
(H, c_ops, rho_ss, chi, and the Beer-Lambert exponent never reference d_probe). Fig.4's
classification depends only on the SHAPE of Pout(Delta_c) at fixed Omega_RF (peak
positions and the interior-valley test), which is invariant under a uniform vertical
rescale. So this run's delta_left/delta_right/threshold_mhz are expected to be
numerically identical to the d=0.76mm run's -- see the cross-check printed by the
script (fig4_distortion_analysis_d036mm.py stdout) for the actual verified numbers.
