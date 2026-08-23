# ============================================================
# FIG. 4 -- PEAK TRACE DIRECTLY FROM DATA
#
# Replaces fig4_continuous_boundary.py, which drew a straight
# line between the two branches' endpoints -- that line was NOT
# derived from any computed Pout value in the interior region,
# so it was fabricated (correctly flagged by the user).
#
# This script instead re-runs find_peaks() on every row of the
# real fig3 Pout_surface, with NO resolvability threshold and NO
# separation/interior-valley gate. Whatever peak count and
# position the data actually has at each Omega_RF is what gets
# plotted -- 2 points where two local maxima exist, 1 point
# (both branches coincide) where they've merged into one. The
# resulting curve shape is not assumed in advance.
#
# The 2xGamma_EIT Rayleigh-resolvability threshold from
# fig4_distortion_analysis.py is kept ONLY as a separate shaded
# reference band (a legitimate, pre-committed physical criterion
# for "practically distinguishable given EIT linewidth") -- it
# does not gate what the red curve shows.
# ============================================================

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

OUTPUT_DIR = Path(__file__).resolve().parent
FIG3_DATA = OUTPUT_DIR.parent / "fig3_fresh_build" / "fig3_quantum_response.npz"

data = np.load(FIG3_DATA)
delta_c_mhz = data["delta_c_mhz"]
omega_rf_mhz = data["omega_rf_mhz"]
Pout_surface = data["Pout_surface"]

print("=" * 70)
print("FIG. 4 -- PEAK TRACE DIRECTLY FROM DATA (no resolvability gate)")
print("=" * 70)
print(f"Loaded: {FIG3_DATA}")
print(f"Grid: {Pout_surface.shape} (Omega_RF x Delta_c)")

# ------------------------------------------------------------
# Gamma_EIT -- kept only for the reference shading, same formula
# as fig4_distortion_analysis.py (paper Appendix B, Eq. 51)
# ------------------------------------------------------------

Omega_p_mhz = 8.0
Omega_c_mhz = 1.0
gamma2_mhz = 5.2
Gamma_EIT = (Omega_c_mhz ** 2 + Omega_p_mhz ** 2) / (2.0 * np.sqrt(gamma2_mhz ** 2 + 2.0 * Omega_p_mhz ** 2))
RESOLVABILITY_MULTIPLIER = 2.0
required_separation_mhz = RESOLVABILITY_MULTIPLIER * Gamma_EIT
print(f"\n(Reference only) Gamma_EIT = {Gamma_EIT:.6f} MHz, "
      f"2xGamma_EIT = {required_separation_mhz:.6f} MHz")

# ------------------------------------------------------------
# Raw peak extraction -- every row, no filters beyond the same
# +/-10 MHz search window used previously (keeps grid edges out).
# ------------------------------------------------------------

SEARCH_HALF_WIDTH_MHZ = 10.0
window = np.abs(delta_c_mhz) <= SEARCH_HALF_WIDTH_MHZ
dc_w = delta_c_mhz[window]

n_omega = len(omega_rf_mhz)
left_x = np.full(n_omega, np.nan)
right_x = np.full(n_omega, np.nan)
n_peaks_found = np.zeros(n_omega, dtype=int)

for i in range(n_omega):
    row_w = Pout_surface[i, window]
    peak_idx, _ = find_peaks(row_w)
    n_peaks_found[i] = len(peak_idx)

    if len(peak_idx) == 0:
        continue
    elif len(peak_idx) == 1:
        x = dc_w[peak_idx[0]]
        left_x[i] = x
        right_x[i] = x
    else:
        heights = row_w[peak_idx]
        top2 = np.sort(peak_idx[np.argsort(heights)[-2:]])
        left_x[i] = dc_w[top2[0]]
        right_x[i] = dc_w[top2[1]]

print(f"\nRows with 0 peaks found: {np.sum(n_peaks_found == 0)}")
print(f"Rows with 1 peak found:  {np.sum(n_peaks_found == 1)}")
print(f"Rows with 2+ peaks found: {np.sum(n_peaks_found >= 2)}")

merge_idx = np.where(n_peaks_found == 1)[0]
if len(merge_idx) > 0:
    merge_omega_max = omega_rf_mhz[merge_idx].max()
    print(f"\nHighest Omega_RF at which peaks are still merged (1 peak only): "
          f"{merge_omega_max:.4f} MHz")
    print(f"Delta_c of the merged peak there: {left_x[merge_idx[np.argmax(omega_rf_mhz[merge_idx])]]:.4f} MHz")

split_idx = np.where(n_peaks_found >= 2)[0]
if len(split_idx) > 0:
    split_omega_min = omega_rf_mhz[split_idx].min()
    print(f"Lowest Omega_RF at which 2 peaks are already found: {split_omega_min:.4f} MHz")

# ------------------------------------------------------------
# Save results
# ------------------------------------------------------------

np.savez(
    OUTPUT_DIR / "fig4_peak_trace.npz",
    omega_rf_mhz=omega_rf_mhz, delta_c_mhz=delta_c_mhz,
    left_x=left_x, right_x=right_x, n_peaks_found=n_peaks_found,
    Gamma_EIT=Gamma_EIT, required_separation_mhz=required_separation_mhz,
)
print(f"\nSaved: fig4_peak_trace.npz")

import csv
with open(OUTPUT_DIR / "fig4_peak_trace.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Omega_RF_MHz", "n_peaks_found", "left_x_MHz", "right_x_MHz"])
    for i in range(n_omega):
        writer.writerow([f"{omega_rf_mhz[i]:.4f}", n_peaks_found[i],
                          f"{left_x[i]:.4f}", f"{right_x[i]:.4f}"])
print(f"Saved: fig4_peak_trace.csv")

# ------------------------------------------------------------
# PLOT
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 6.5))

valid = ~np.isnan(left_x)
ax.plot(left_x[valid], omega_rf_mhz[valid], "-", color="red", linewidth=1.8,
        label="Obtained by QuTiP (raw peak positions)")
ax.plot(right_x[valid], omega_rf_mhz[valid], "-", color="red", linewidth=1.8)

ax.plot(omega_rf_mhz / 2.0, omega_rf_mhz, "--", color="gray", linewidth=1.2, label="Theoretical")
ax.plot(-omega_rf_mhz / 2.0, omega_rf_mhz, "--", color="gray", linewidth=1.2)

ax.axhline(required_separation_mhz, color="k", linestyle=":", alpha=0.7,
           label="2xGamma_EIT Rayleigh threshold (reference)")
ax.fill_between([-9, 9], 0, required_separation_mhz, color="gray", alpha=0.15)

ax.set_xlabel(r"Coupling detuning, $\Delta_c/2\pi$ (MHz)")
ax.set_ylabel(r"RF Rabi frequency, $\Omega_{RF}/2\pi$ (MHz)")
ax.set_title("Fig. 4 (data-driven) -- peak positions with no resolvability gate")
ax.set_xlim(-9, 9)
ax.set_ylim(0, 16)
ax.legend(fontsize=9, loc="upper right")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "fig4_peak_trace.png", dpi=220, bbox_inches="tight")
plt.close(fig)
print("Saved: fig4_peak_trace.png")
print("\nDONE.")
