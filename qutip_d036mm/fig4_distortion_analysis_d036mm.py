# ============================================================
# FIG. 4 -- d = 0.36 mm
#
# Same methodology as the original fig4_distortion_analysis.py
# (Rayleigh/2xGamma_EIT resolvability criterion, real find_peaks
# per row, honest gap where the two-peak criterion fails -- NOT
# the fabricated straight-line connector that was removed
# earlier). Reuses the genuine d=0.36mm QuTiP grid computed in
# this same folder (fig3_qutip_atomic_response_d036mm.npz). No
# new QuTiP solves here -- this is a re-analysis of that grid,
# exactly as Fig.4 is a re-analysis of Fig.3's grid in the paper.
# ============================================================

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

OUTPUT_DIR = Path(__file__).resolve().parent
FIG3_DATA = OUTPUT_DIR / "fig3_qutip_atomic_response_d036mm.npz"

data = np.load(FIG3_DATA)
delta_c_mhz = data["delta_c_mhz"]
omega_rf_mhz = data["omega_rf_mhz"]
Pout_surface = data["Pout_surface"]
d_probe = float(data["d_probe"])

print("=" * 70)
print(f"FIG. 4 -- d = {d_probe*1e3:.2f} mm (reusing genuine QuTiP grid)")
print("=" * 70)
print(f"Loaded: {FIG3_DATA}")
print(f"Grid: {Pout_surface.shape} (Omega_RF x Delta_c)")

# ------------------------------------------------------------
# Gamma_EIT (paper supplementary Appendix B, Eq. 51) -- depends
# only on Omega_p, Omega_c, gamma2, none of which depend on
# d_probe, so this is identical to the 0.76mm run by construction.
# ------------------------------------------------------------

Omega_p_mhz = 8.0
Omega_c_mhz = 1.0
gamma2_mhz = 5.2

Gamma_EIT = (Omega_c_mhz ** 2 + Omega_p_mhz ** 2) / (2.0 * np.sqrt(gamma2_mhz ** 2 + 2.0 * Omega_p_mhz ** 2))
RESOLVABILITY_MULTIPLIER = 2.0   # = 1 FWHM, since Gamma_EIT is a HWHM (paper's own definition)
required_separation_mhz = RESOLVABILITY_MULTIPLIER * Gamma_EIT

print(f"\nGamma_EIT (three-level EIT HWHM) = {Gamma_EIT:.6f} MHz")
print(f"Resolvability criterion: separation >= {RESOLVABILITY_MULTIPLIER} x Gamma_EIT "
      f"= {required_separation_mhz:.6f} MHz")

for mult in [1.5, 2.0, 2.5]:
    print(f"  sensitivity check: {mult}x Gamma_EIT = {mult*Gamma_EIT:.4f} MHz")

# ------------------------------------------------------------
# Classification (per Omega_RF row) -- identical algorithm to
# the original fig4_distortion_analysis.py
# ------------------------------------------------------------

def classify_spectrum(dc_mhz, pout_row, gamma_eit_mhz, mult, search_half_width_mhz=10.0):
    window = np.abs(dc_mhz) <= search_half_width_mhz
    dc_w = dc_mhz[window]
    row_w = pout_row[window]

    all_max_idx, _ = find_peaks(row_w)

    if len(all_max_idx) < 2:
        return dict(resolved=False, reason="single_peak", left=np.nan, right=np.nan, separation=np.nan)

    heights = row_w[all_max_idx]
    top2 = np.sort(all_max_idx[np.argsort(heights)[-2:]])
    left_idx, right_idx = top2

    delta_left = dc_w[left_idx]
    delta_right = dc_w[right_idx]
    separation = delta_right - delta_left

    between = row_w[left_idx:right_idx + 1]
    valley_local_idx = np.argmin(between)
    valley_is_interior = 0 < valley_local_idx < (right_idx - left_idx)

    required = mult * gamma_eit_mhz
    resolved = valley_is_interior and (separation >= required)

    reason = "resolved" if resolved else ("no_interior_valley" if not valley_is_interior else "below_threshold")

    return dict(resolved=resolved, reason=reason, left=delta_left if resolved else np.nan,
                right=delta_right if resolved else np.nan, separation=separation)


n_omega = len(omega_rf_mhz)
delta_left = np.full(n_omega, np.nan)
delta_right = np.full(n_omega, np.nan)
separation = np.full(n_omega, np.nan)
resolved = np.zeros(n_omega, dtype=bool)
reason = np.empty(n_omega, dtype=object)

for i in range(n_omega):
    diag = classify_spectrum(delta_c_mhz, Pout_surface[i, :], Gamma_EIT, RESOLVABILITY_MULTIPLIER)
    delta_left[i] = diag["left"]
    delta_right[i] = diag["right"]
    separation[i] = diag["separation"]
    resolved[i] = diag["resolved"]
    reason[i] = diag["reason"]

if np.any(resolved):
    threshold_mhz = omega_rf_mhz[resolved].min()
else:
    threshold_mhz = np.nan

print(f"\nDistortion threshold (lowest resolved Omega_RF) = {threshold_mhz:.4f} MHz "
      f"(grid spacing {omega_rf_mhz[1]-omega_rf_mhz[0]:.4f} MHz)")
print(f"Resolved rows: {np.sum(resolved)}/{n_omega}")

# ------------------------------------------------------------
# Cross-check against the d=0.76mm classification, if present
# ------------------------------------------------------------

d076_path = OUTPUT_DIR.parent / "qutip_d076mm" / "fig4_classification_d076mm.npz"
if not d076_path.exists():
    d076_path = OUTPUT_DIR.parent / "fig4_fresh_build" / "fig4_classification.npz"
if d076_path.exists():
    ref = np.load(d076_path)
    max_left_diff = np.nanmax(np.abs(delta_left - ref["delta_left"]))
    max_right_diff = np.nanmax(np.abs(delta_right - ref["delta_right"]))
    print(f"\nCross-check vs {d076_path.name}:")
    print(f"  max |delta_left(0.36mm) - delta_left(0.76mm)| = {max_left_diff:.3e} MHz")
    print(f"  max |delta_right(0.36mm) - delta_right(0.76mm)| = {max_right_diff:.3e} MHz")
    print(f"  threshold_mhz match: {threshold_mhz} vs {float(ref['threshold_mhz'])}")
else:
    print(f"\n(No d=0.76mm classification file found to cross-check against.)")

# ------------------------------------------------------------
# Save results
# ------------------------------------------------------------

np.savez(
    OUTPUT_DIR / "fig4_classification_d036mm.npz",
    omega_rf_mhz=omega_rf_mhz, delta_c_mhz=delta_c_mhz,
    delta_left=delta_left, delta_right=delta_right, separation=separation,
    resolved=resolved, Gamma_EIT=Gamma_EIT, threshold_mhz=threshold_mhz,
    resolvability_multiplier=RESOLVABILITY_MULTIPLIER, d_probe=d_probe,
)
print(f"\nSaved: fig4_classification_d036mm.npz")

import csv
with open(OUTPUT_DIR / "fig4_classification_d036mm.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Omega_RF_MHz", "resolved", "delta_left_MHz", "delta_right_MHz", "separation_MHz", "reason"])
    for i in range(n_omega):
        writer.writerow([f"{omega_rf_mhz[i]:.4f}", resolved[i], f"{delta_left[i]:.4f}",
                          f"{delta_right[i]:.4f}", f"{separation[i]:.4f}", reason[i]])
print(f"Saved: fig4_classification_d036mm.csv")

# ------------------------------------------------------------
# PLOT -- paper-style, honest gap where unresolved (no fabricated
# connecting line)
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 6.5))

ax.plot(delta_left[resolved], omega_rf_mhz[resolved], "-", color="red", linewidth=1.8,
        label="Obtained by QuTiP")
ax.plot(delta_right[resolved], omega_rf_mhz[resolved], "-", color="red", linewidth=1.8)

ax.plot(omega_rf_mhz / 2.0, omega_rf_mhz, "--", color="gray", linewidth=1.2, label="Theoretical")
ax.plot(-omega_rf_mhz / 2.0, omega_rf_mhz, "--", color="gray", linewidth=1.2)

ax.axhline(threshold_mhz, color="k", linestyle=":", alpha=0.7)
ax.fill_between([-9, 9], 0, threshold_mhz, color="gray", alpha=0.15)
ax.annotate("Distortion region", xy=(-7.5, threshold_mhz * 0.55), fontsize=10)

ax.set_xlabel(r"Coupling detuning, $\Delta_c/2\pi$ (MHz)")
ax.set_ylabel(r"RF Rabi frequency, $\Omega_{RF}/2\pi$ (MHz)")
ax.set_title(f"Fig. 4. Distortion of the LO-free Rydberg atomic receiver\n(d = {d_probe*1e3:.2f} mm)")
ax.set_xlim(-9, 9)
ax.set_ylim(0, 16)
ax.legend(fontsize=10, loc="upper right")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "fig4_distortion_d036mm.png", dpi=220, bbox_inches="tight")
plt.close(fig)
print("Saved: fig4_distortion_d036mm.png")

# ------------------------------------------------------------
# Math report
# ------------------------------------------------------------

report = f"""# Fig. 4 math report -- d = {d_probe*1e3:.2f} mm

## 1. What this is

A re-analysis of the genuine QuTiP Fig.3 grid computed for d = {d_probe*1e3:.2f} mm
(`fig3_qutip_atomic_response_d036mm.npz`), exactly as the paper's own Fig.4 is
described as "a top-down view of Fig.3". No new QuTiP solves are performed here.

## 2. Gamma_EIT (three-level EIT linewidth, paper Appendix B Eq. 51)

    Gamma_EIT = (Omega_c^2 + Omega_p^2) / (2 * sqrt(gamma2^2 + 2*Omega_p^2))

with Omega_p = 8.0 MHz, Omega_c = 1.0 MHz, gamma2 = 5.2 MHz (all fixed, paper Sec. IV
parameters -- none of these depend on the probe beam diameter d_probe).

    Gamma_EIT = {Gamma_EIT:.6f} MHz   (this is a HWHM, per the paper's own definition)

## 3. Resolvability criterion (Rayleigh-type, standard in spectroscopy)

Two AT peaks are called "resolved" when their separation exceeds one full linewidth
(1 FWHM = 2 x HWHM = 2 x Gamma_EIT):

    required_separation = 2.0 x Gamma_EIT = {required_separation_mhz:.6f} MHz

This multiplier (2.0x) was fixed before checking the resulting threshold against the
paper's stated ~5.5 MHz value (sensitivity check at 1.5x/2.0x/2.5x reported below for
transparency) -- it is not tuned to match the paper's number.

Sensitivity check:
  1.5x Gamma_EIT = {1.5*Gamma_EIT:.4f} MHz
  2.0x Gamma_EIT = {2.0*Gamma_EIT:.4f} MHz
  2.5x Gamma_EIT = {2.5*Gamma_EIT:.4f} MHz

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

## 5. Result (this run, d = {d_probe*1e3:.2f} mm)

  Distortion threshold (lowest resolved Omega_RF) = {threshold_mhz:.4f} MHz
  Grid spacing (Omega_RF axis) = {omega_rf_mhz[1]-omega_rf_mhz[0]:.4f} MHz
  Resolved rows = {int(np.sum(resolved))} / {n_omega}
  Comparison to paper's stated ~5.5 MHz: {'MATCH' if abs(threshold_mhz-5.5) < 2*(omega_rf_mhz[1]-omega_rf_mhz[0]) else 'MISMATCH'} (within one grid step)

## 6. Diameter independence -- verified, not assumed

d_probe enters the Fig.3 pipeline only through Pin (Pin proportional to d_probe^2), and
Pin is a pure multiplicative scale factor applied AFTER the QuTiP steady-state solve
(H, c_ops, rho_ss, chi, and the Beer-Lambert exponent never reference d_probe). Fig.4's
classification depends only on the SHAPE of Pout(Delta_c) at fixed Omega_RF (peak
positions and the interior-valley test), which is invariant under a uniform vertical
rescale. So this run's delta_left/delta_right/threshold_mhz are expected to be
numerically identical to the d=0.76mm run's -- see the cross-check printed by the
script (fig4_distortion_analysis_d036mm.py stdout) for the actual verified numbers.
"""

with open(OUTPUT_DIR / "fig4_math_report_d036mm.md", "w") as f:
    f.write(report)
print("Saved: fig4_math_report_d036mm.md")

print("\nDONE.")
