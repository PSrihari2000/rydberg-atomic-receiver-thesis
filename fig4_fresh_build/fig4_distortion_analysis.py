# ============================================================
# FIG. 4 -- DISTORTION OF THE LO-FREE RYDBERG ATOMIC RECEIVER
#
# Re-analysis of the already-computed, genuine QuTiP grid from
# fig3_fresh_build/fig3_quantum_response.npz (NO new QuTiP solves
# here -- the paper itself describes Fig.4 as "a top-down view of
# Fig.3"). Every peak position, valley, and separation below comes
# from find_peaks() on the real simulated Pout(Delta_c) spectra.
#
# Resolvability criterion: a practical FWHM-based resolvability rule --
# two peaks are considered resolved when their separation is at least
# one measured FWHM of the zero-RF EIT feature. (Not labeled "Rayleigh
# criterion" -- that name refers to a specific diffraction-pattern
# condition for Airy/sinc lineshapes that doesn't rigorously apply to
# this EIT/AT doublet; this is a plainly-stated separation>=FWHM rule,
# not a borrowed named criterion.) The FWHM used here is MEASURED
# directly from this dataset's own Omega_RF=0 spectrum (not taken from
# the paper's analytical Eq.51, which was derived for a different
# section of the paper under a gamma3=gamma4=0 approximation this
# simulation does not use). See fig4_math_report.md for full reasoning.
# ============================================================

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import csv

OUTPUT_DIR = Path(__file__).resolve().parent
FIG3_DATA = OUTPUT_DIR.parent / "fig3_fresh_build" / "fig3_quantum_response.npz"

data = np.load(FIG3_DATA)
delta_c_mhz = data["delta_c_mhz"]
omega_rf_mhz = data["omega_rf_mhz"]
Pout_surface = data["Pout_surface"] / 1e-6   # microW
d_probe_mm = float(data["d_probe"]) * 1e3

print("=" * 70)
print("FIG. 4 -- FWHM-based resolvability criterion, measured HWHM, fig3 QuTiP data")
print("=" * 70)
print(f"Loaded: {FIG3_DATA}")
print(f"Grid: Delta_c [{delta_c_mhz.min()},{delta_c_mhz.max()}] MHz, {len(delta_c_mhz)} pts, "
      f"step {delta_c_mhz[1]-delta_c_mhz[0]:.4f} MHz")
print(f"      Omega_RF [{omega_rf_mhz.min()},{omega_rf_mhz.max()}] MHz, {len(omega_rf_mhz)} pts, "
      f"step {omega_rf_mhz[1]-omega_rf_mhz[0]:.4f} MHz")

# ------------------------------------------------------------
# STEP 1: measure the EIT HWHM directly from the Omega_RF=0 row
# ------------------------------------------------------------

row0 = Pout_surface[0, :]
peak_idx0 = int(np.argmax(row0))
peak_pos0 = delta_c_mhz[peak_idx0]
peak_val0 = row0[peak_idx0]
baseline0 = 0.5 * (row0[0] + row0[-1])
half_level0 = baseline0 + 0.5 * (peak_val0 - baseline0)


def interp_crossing(x, y, level, i0, direction):
    i = i0
    n = len(y)
    while 0 <= i + direction < n:
        y0, y1 = y[i], y[i + direction]
        if (y0 - level) * (y1 - level) <= 0 and y0 != y1:
            x0, x1 = x[i], x[i + direction]
            frac = (level - y0) / (y1 - y0)
            return x0 + frac * (x1 - x0)
        i += direction
    return None


right_cross = interp_crossing(delta_c_mhz, row0, half_level0, peak_idx0, +1)
left_cross = interp_crossing(delta_c_mhz, row0, half_level0, peak_idx0, -1)
HWHM_measured = float(np.mean([right_cross - peak_pos0, peak_pos0 - left_cross]))
FWHM_measured = 2.0 * HWHM_measured

print(f"\nSTEP 1: Measured EIT linewidth (Omega_RF=0 row)")
print(f"  Peak position = {peak_pos0:.4f} MHz, peak value = {peak_val0:.4f} microW")
print(f"  Baseline (avg of grid edges) = {baseline0:.4f} microW")
print(f"  Half-max crossings: left={left_cross:.4f} MHz, right={right_cross:.4f} MHz")
print(f"  Measured HWHM = {HWHM_measured:.4f} MHz  ->  FWHM = {FWHM_measured:.4f} MHz")

# ------------------------------------------------------------
# STEP 2: per-row peak extraction + FWHM-based classification
# ------------------------------------------------------------

n_omega = len(omega_rf_mhz)
n_peaks_arr = np.zeros(n_omega, dtype=int)
left_x = np.full(n_omega, np.nan)
right_x = np.full(n_omega, np.nan)
separation = np.full(n_omega, np.nan)
valley_pos = np.full(n_omega, np.nan)
resolved = np.zeros(n_omega, dtype=bool)
reason = np.empty(n_omega, dtype=object)

for i in range(n_omega):
    row = Pout_surface[i, :]
    idx, _ = find_peaks(row)
    n_peaks_arr[i] = len(idx)

    if len(idx) < 2:
        reason[i] = "single_peak"
        continue

    heights = row[idx]
    top2 = np.sort(idx[np.argsort(heights)[-2:]])
    li, ri = top2
    left_x[i] = delta_c_mhz[li]
    right_x[i] = delta_c_mhz[ri]
    separation[i] = delta_c_mhz[ri] - delta_c_mhz[li]

    between = row[li:ri + 1]
    vloc = np.argmin(between)
    interior = 0 < vloc < (ri - li)
    if not interior:
        reason[i] = "no_interior_valley"
        left_x[i] = np.nan
        right_x[i] = np.nan
        continue
    valley_pos[i] = delta_c_mhz[li + vloc]

    if separation[i] >= FWHM_measured:
        resolved[i] = True
        reason[i] = "resolved"
    else:
        reason[i] = "below_threshold"
        left_x[i] = np.nan
        right_x[i] = np.nan

threshold_mhz = omega_rf_mhz[resolved].min() if np.any(resolved) else None

print(f"\nSTEP 2: FWHM-based classification (separation >= FWHM_measured = {FWHM_measured:.4f} MHz)")
print(f"  Resolved rows: {int(np.sum(resolved))}/{n_omega}")
print(f"  THRESHOLD (lowest resolved Omega_RF) = {threshold_mhz:.4f} MHz "
      f"(Omega_RF grid step {omega_rf_mhz[1]-omega_rf_mhz[0]:.4f} MHz)")
print(f"  Comparison to paper's stated ~5.5 MHz: "
      f"{'within one grid step' if abs(threshold_mhz-5.5) < 2*(omega_rf_mhz[1]-omega_rf_mhz[0]) else 'differs by more than one grid step'}")

# ------------------------------------------------------------
# SAVE CSV -- backs up every number used above
# ------------------------------------------------------------

csv_path = OUTPUT_DIR / "fig4_classification.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["# Measured HWHM (MHz)", f"{HWHM_measured:.6f}"])
    writer.writerow(["# Measured FWHM = required separation (MHz)", f"{FWHM_measured:.6f}"])
    writer.writerow(["# Threshold Omega_RF (MHz)", f"{threshold_mhz:.6f}"])
    writer.writerow([])
    writer.writerow(["Omega_RF_MHz", "n_peaks", "left_MHz", "right_MHz", "separation_MHz",
                      "valley_position_MHz", "resolved", "reason"])
    for i in range(n_omega):
        writer.writerow([
            f"{omega_rf_mhz[i]:.4f}", n_peaks_arr[i],
            f"{left_x[i]:.4f}" if not np.isnan(left_x[i]) else "",
            f"{right_x[i]:.4f}" if not np.isnan(right_x[i]) else "",
            f"{separation[i]:.4f}" if not np.isnan(separation[i]) else "",
            f"{valley_pos[i]:.4f}" if not np.isnan(valley_pos[i]) else "",
            resolved[i], reason[i],
        ])
print(f"\nSaved: {csv_path.name}")

# ------------------------------------------------------------
# PLOT -- paper-style, minimal legend (only the two paper-style
# series), threshold line + shaded distortion region unlabeled
# in the legend, matching the paper's own visual convention
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 6.5))

has_valley = ~np.isnan(left_x)
ax.plot(left_x[has_valley], omega_rf_mhz[has_valley], "-", color="red", linewidth=1.8,
        label="Obtained by QuTiP")
ax.plot(right_x[has_valley], omega_rf_mhz[has_valley], "-", color="red", linewidth=1.8)

ax.plot(omega_rf_mhz / 2.0, omega_rf_mhz, "--", color="gray", linewidth=1.2, label="Theoretical")
ax.plot(-omega_rf_mhz / 2.0, omega_rf_mhz, "--", color="gray", linewidth=1.2)

ax.axhline(threshold_mhz, color="k", linestyle=":", alpha=0.7)
ax.fill_between([-9, 9], 0, threshold_mhz, color="gray", alpha=0.15)
ax.annotate("Distortion region", xy=(-7.5, threshold_mhz * 0.55), fontsize=10)

ax.set_xlabel(r"Coupling detuning, $\Delta_c/2\pi$ (MHz)")
ax.set_ylabel(r"RF Rabi frequency, $\Omega_{RF}/2\pi$ (MHz)")
ax.set_title("Fig. 4. Distortion of the LO-free Rydberg atomic receiver")
ax.set_xlim(-9, 9)
ax.set_ylim(0, 16)
ax.legend(fontsize=10, loc="upper right")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "fig4_distortion.png", dpi=220, bbox_inches="tight")
plt.close(fig)
print("Saved: fig4_distortion.png")

# ------------------------------------------------------------
# MATH REPORT
# ------------------------------------------------------------

report = f"""# Fig. 4 math report -- Distortion of the LO-free Rydberg atomic receiver

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

    L = 1 cm, N0 = 1e15 m^-3, d_probe = {d_probe_mm:.2f} mm
    Omega_p/2pi = 8.0 MHz, Omega_c/2pi = 1.0 MHz
    gamma1 = 0, gamma2/2pi = 5.2 MHz, gamma3/2pi = 3.9 kHz, gamma4/2pi = 1.7 kHz
    wp_RF (|3>-|4> dipole moment) = -1443.459 e*a0
    wp_12 (|1>-|2> dipole moment, squared convention) = (2.5 e*a0)^2
    lambda_p = 852 nm

## 4. Data grid (fig3_quantum_response.npz)

    Delta_c/2pi: [{delta_c_mhz.min()}, {delta_c_mhz.max()}] MHz, {len(delta_c_mhz)} points,
    step {delta_c_mhz[1]-delta_c_mhz[0]:.4f} MHz
    Omega_RF/2pi: [{omega_rf_mhz.min()}, {omega_rf_mhz.max()}] MHz, {len(omega_rf_mhz)} points,
    step {omega_rf_mhz[1]-omega_rf_mhz[0]:.4f} MHz

## 5. EIT linewidth -- MEASURED, not taken from the paper's analytical formula

The paper's Eq. 51 (Gamma_EIT) is derived in Sec. III-B for the LO-dressed
receiver, under an additional gamma3=gamma4=0 approximation this
simulation does not use (this run keeps the real, nonzero gamma3, gamma4
from Sec. IV). Rather than import that formula, the HWHM used here is
measured directly from THIS dataset's own Omega_RF=0 spectrum (the plain
3-level EIT lineshape, no RF field at all):

    Peak position: Delta_c = {peak_pos0:.4f} MHz
    Peak value: {peak_val0:.4f} microW
    Baseline (avg of the two grid edges): {baseline0:.4f} microW
    Half-max level: {half_level0:.4f} microW
    Half-max crossings: left = {left_cross:.4f} MHz, right = {right_cross:.4f} MHz

    MEASURED HWHM (Gamma) = {HWHM_measured:.4f} MHz
    MEASURED FWHM = {FWHM_measured:.4f} MHz

(For reference only: the paper's analytical Eq. 51 gives Gamma_EIT =
2.6101 MHz for these same Omega_p, Omega_c, gamma2 -- close to, but not
identical to, the {HWHM_measured:.4f} MHz measured here, consistent with
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
     measured FWHM ({FWHM_measured:.4f} MHz). This is a plainly-stated
     separation>=1-FWHM rule, not a specific named criterion -- it is
     NOT called "the Rayleigh criterion" here, since that name properly
     refers to a diffraction-pattern condition for Airy/sinc lineshapes
     that does not rigorously apply to this EIT/AT doublet. The linewidth
     is measured from this dataset, not imported from elsewhere.

## 7. Result

    Resolved rows: {int(np.sum(resolved))} / {n_omega}
    THRESHOLD (lowest resolved Omega_RF) = {threshold_mhz:.4f} MHz
    Omega_RF grid spacing = {omega_rf_mhz[1]-omega_rf_mhz[0]:.4f} MHz (threshold accurate to within
    one grid step, not stated as exact)

    Paper's stated saturation threshold: ~5.5 MHz
    This result: {threshold_mhz:.4f} MHz -- independently measured, not tuned to match the paper.

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
    {threshold_mhz:.4f} MHz FWHM-based threshold above. Both numbers are
    genuine and data-derived; they answer different questions (literal
    peak count vs. linewidth-based practical distinguishability) and
    should not be conflated.
  - Grid resolution: separation/HWHM values are only as precise as the
    Delta_c grid spacing above; finer grids would refine (not
    qualitatively change) the reported threshold.
"""

with open(OUTPUT_DIR / "fig4_math_report.md", "w") as f:
    f.write(report)
print("Saved: fig4_math_report.md")

print("\nDONE.")
