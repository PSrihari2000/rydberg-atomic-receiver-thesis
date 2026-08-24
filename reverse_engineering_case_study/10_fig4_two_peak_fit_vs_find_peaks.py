# ============================================================
# *** CASE STUDY -- NOT a validated result ***
#
# QUESTION: the paper's own Fig.4 red curve is described as "jagged"
# due to "perturbations in the quantum system during the solution of
# the master equation, reflecting the STEP-WISE PROCESS INVOLVED IN
# LOCATING THE EIT SPECTRUM PEAKS" -- and it NATURALLY saturates around
# 5.5MHz as Omega_RF decreases, rather than the two branches simply
# meeting at Omega_RF=0. Our fig4_fresh_build instead applies an
# EXTERNAL classification rule (separation >= 1 measured FWHM) to
# find_peaks()'s literal local-maxima positions. These are two
# genuinely different methodologies. The paper's own cited source for
# this exact technique, Jing et al. 2020 (ref [20]), explicitly says
# they "performed a two-peak fit of the EIT spectrum ... using Voigt
# functions F_+, F_-, that is P(Delta_c)=F_+(Delta_c-Omega_+)+
# F_-(Delta_c-Omega_-)" -- a FIT, not simple peak-counting. This tests
# whether a genuine two-peak-fit approach on our REAL, already-computed
# grid reproduces a naturally-saturating curve shape, unlike
# find_peaks()'s hard binary (2-peaks-or-1) transition.
#
# Uses ONLY fig3_fresh_build's real, already-computed Pout grid -- no
# new QuTiP solves. Does not touch any frozen fresh_build file.
# ============================================================

from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

OUTPUT_DIR = Path(__file__).resolve().parent
FIG3_NPZ = OUTPUT_DIR.parent / "fig3_fresh_build" / "fig3_quantum_response.npz"

print("=" * 70)
print("*** CASE STUDY 10 -- Fig.4: two-peak fit vs find_peaks() ***")
print("=" * 70)

data = np.load(FIG3_NPZ)
delta_c_mhz = data["delta_c_mhz"]
omega_rf_mhz = data["omega_rf_mhz"]
Pout_surface = data["Pout_surface"] / 1e-6  # microW

def two_lorentzian(x, baseline, A1, c1, w1, A2, c2, w2):
    return (baseline
            + A1 * (w1**2) / ((x - c1)**2 + w1**2)
            + A2 * (w2**2) / ((x - c2)**2 + w2**2))

print("\nFitting a genuine two-Lorentzian model (matching Jing et al. 2020's own")
print("two-peak-fit methodology) to every real row of the Delta_c sweep...")

fit_separation = np.full(len(omega_rf_mhz), np.nan)
fit_success = np.zeros(len(omega_rf_mhz), dtype=bool)
findpeaks_separation = np.full(len(omega_rf_mhz), np.nan)
findpeaks_n = np.zeros(len(omega_rf_mhz), dtype=int)

row0 = Pout_surface[0, :]
baseline0_guess = 0.5*(row0[0]+row0[-1])
peak0_guess = row0.max()

for i, w_rf in enumerate(omega_rf_mhz):
    row = Pout_surface[i, :]

    # find_peaks() reference (same as fig4_fresh_build's own method)
    idx, _ = find_peaks(row)
    findpeaks_n[i] = len(idx)
    if len(idx) >= 2:
        heights = row[idx]
        top2 = np.sort(idx[np.argsort(heights)[-2:]])
        findpeaks_separation[i] = delta_c_mhz[top2[1]] - delta_c_mhz[top2[0]]

    # Two-Lorentzian fit, seeded near +/- Omega_RF/2 (the paper's own
    # theoretical extrema, Eq.48) regardless of whether find_peaks sees 2 peaks
    c1_guess, c2_guess = -w_rf/2.0, w_rf/2.0
    if abs(c1_guess - c2_guess) < 0.3:
        c1_guess, c2_guess = -0.3, 0.3  # avoid a degenerate zero-separation seed
    p0 = [baseline0_guess, peak0_guess-baseline0_guess, c1_guess, 1.5,
          peak0_guess-baseline0_guess, c2_guess, 1.5]
    try:
        popt, _ = curve_fit(two_lorentzian, delta_c_mhz, row, p0=p0, maxfev=8000)
        c1, c2 = popt[2], popt[5]
        fit_separation[i] = abs(c2 - c1)
        fit_success[i] = True
    except Exception:
        fit_success[i] = False

print(f"Fit succeeded for {np.sum(fit_success)}/{len(omega_rf_mhz)} rows")

print(f"\n{'Omega_RF(MHz)':>14} {'find_peaks_n':>13} {'find_peaks_sep':>15} {'2peakfit_sep':>13}")
for i in range(0, len(omega_rf_mhz), 8):
    fp_sep = f"{findpeaks_separation[i]:.4f}" if not np.isnan(findpeaks_separation[i]) else "--"
    fit_sep = f"{fit_separation[i]:.4f}" if fit_success[i] else "FAIL"
    print(f"{omega_rf_mhz[i]:14.4f} {findpeaks_n[i]:13d} {fp_sep:>15} {fit_sep:>13}")

# Where does the 2-peak-fit separation approach zero / saturate, vs find_peaks' hard cutoff?
valid_fit = fit_success & ~np.isnan(fit_separation)
print(f"\n2-peak-fit separation range: [{fit_separation[valid_fit].min():.4f},"
      f"{fit_separation[valid_fit].max():.4f}]MHz")
print(f"2-peak-fit separation AT Omega_RF=0: {fit_separation[0]:.4f}MHz "
      f"(does the fit itself refuse to converge to near-zero separation, or does it?)")

# find_peaks' hard transition point (existing fig4_fresh_build threshold, for reference)
resolved_fp = ~np.isnan(findpeaks_separation)
print(f"find_peaks(): {int(np.sum(resolved_fp))}/{len(omega_rf_mhz)} rows show >=2 local maxima at all")
if np.any(resolved_fp):
    print(f"find_peaks() lowest Omega_RF with 2 distinct local maxima: "
          f"{omega_rf_mhz[resolved_fp].min():.4f}MHz")

fig, ax = plt.subplots(figsize=(8, 6.5))
ax.plot(fit_separation[valid_fit], omega_rf_mhz[valid_fit], "-", color="purple", linewidth=1.5,
        label="Two-Lorentzian FIT separation (Jing et al.-style)")
ax.plot(findpeaks_separation[resolved_fp], omega_rf_mhz[resolved_fp], "o", color="red",
        markersize=3, label="find_peaks() separation (this project's method)")
ax.plot(omega_rf_mhz, omega_rf_mhz, "--", color="gray", linewidth=1.0, label="Theoretical (Eq.48): separation=Omega_RF")
ax.axhline(5.5, color="k", linestyle=":", alpha=0.6, label="Paper's stated ~5.5MHz")
ax.axhline(5.125, color="green", linestyle=":", alpha=0.6, label="fig4_fresh_build's 5.125MHz")
ax.set_xlabel("Separation (MHz)")
ax.set_ylabel("Omega_RF/2pi (MHz)")
ax.set_title("Case study: does a genuine 2-peak fit naturally saturate,\nunlike find_peaks()'s hard binary transition?")
ax.set_xlim(-2, 10)
ax.set_ylim(0, 16)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "10_fig4_two_peak_fit_comparison.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("\nSaved: 10_fig4_two_peak_fit_comparison.png")
print("\nDONE.")
