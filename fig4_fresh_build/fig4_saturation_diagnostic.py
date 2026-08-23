# ============================================================
# FIG. 4 -- POST-PROCESSING DIAGNOSTIC ONLY
#
# Loads the already-completed fig4_data.npz (32,361 genuine
# QuTiP steady-state solves, Delta_c in [-10,10] MHz, Omega_RF
# in [0,20] MHz). NO new QuTiP solves. NO Hamiltonian changes.
# NO data modification, fitting, or forcing toward the paper.
#
# Purpose: investigate why the raw QuTiP peak-separation data
# tracks the theoretical Delta_nu_AT = Omega_RF relation almost
# exactly, rather than showing a low-Omega_RF saturation plateau
# the way the published Fig.4 appears to.
# ============================================================

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

OUTPUT_DIR = Path(__file__).resolve().parent
data = np.load(OUTPUT_DIR / "fig4_data.npz")

delta_c_mhz = data["delta_c_mhz"]
omega_rf_mhz = data["omega_rf_mhz"]
Pout_surface = data["Pout_surface"] / 1e-6   # microW, for readability

print("=" * 70)
print("FIG. 4 -- SATURATION DIAGNOSTIC (post-processing of fig4_data.npz)")
print("=" * 70)
print(f"Loaded: fig4_data.npz  (NOT re-solving QuTiP)")
print(f"Delta_c grid: [{delta_c_mhz.min()}, {delta_c_mhz.max()}] MHz, "
      f"{len(delta_c_mhz)} pts, step {delta_c_mhz[1]-delta_c_mhz[0]:.4f} MHz")
print(f"Omega_RF grid: [{omega_rf_mhz.min()}, {omega_rf_mhz.max()}] MHz, "
      f"{len(omega_rf_mhz)} pts, step {omega_rf_mhz[1]-omega_rf_mhz[0]:.4f} MHz")

# ------------------------------------------------------------
# 2. Full per-row peak extraction (ALL local maxima, no filter)
# ------------------------------------------------------------

n_omega = len(omega_rf_mhz)
n_peaks_total = np.zeros(n_omega, dtype=int)
top2_left = np.full(n_omega, np.nan)
top2_right = np.full(n_omega, np.nan)
top2_separation = np.full(n_omega, np.nan)
valley_pos = np.full(n_omega, np.nan)
valley_height = np.full(n_omega, np.nan)
peak_heights_top2 = np.full((n_omega, 2), np.nan)

all_peaks_record = {}   # omega index -> list of (position_MHz, height)

for i in range(n_omega):
    row = Pout_surface[i, :]
    idx, _ = find_peaks(row)
    n_peaks_total[i] = len(idx)
    all_peaks_record[i] = [(delta_c_mhz[k], row[k]) for k in idx]

    if len(idx) >= 2:
        heights = row[idx]
        top2 = np.sort(idx[np.argsort(heights)[-2:]])
        li, ri = top2
        top2_left[i] = delta_c_mhz[li]
        top2_right[i] = delta_c_mhz[ri]
        top2_separation[i] = delta_c_mhz[ri] - delta_c_mhz[li]
        peak_heights_top2[i, 0] = row[li]
        peak_heights_top2[i, 1] = row[ri]
        between = row[li:ri + 1]
        vloc = np.argmin(between)
        valley_pos[i] = delta_c_mhz[li + vloc]
        valley_height[i] = between[vloc]

# ------------------------------------------------------------
# 1. Representative Pout vs Delta_c spectra at requested Omega_RF values
# ------------------------------------------------------------

requested_omega = [2.0, 5.0, 5.25, 5.375, 5.5, 6.0, 10.0, 15.0]
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.flatten()

print("\n" + "=" * 70)
print("REQUESTED SPECTRA")
print("=" * 70)
for ax, w in zip(axes, requested_omega):
    i = int(np.argmin(np.abs(omega_rf_mhz - w)))
    row = Pout_surface[i, :]
    ax.plot(delta_c_mhz, row, color="tab:blue", linewidth=1.3)
    for pos, ht in all_peaks_record[i]:
        ax.axvline(pos, color="red", linestyle=":", alpha=0.6)
    ax.set_title(f"Omega_RF/2pi={omega_rf_mhz[i]:.3f} MHz\n"
                 f"n_peaks={n_peaks_total[i]}, sep(top2)={top2_separation[i]:.4f} MHz",
                 fontsize=9)
    ax.set_xlabel(r"$\Delta_c/2\pi$ (MHz)", fontsize=8)
    ax.set_ylabel(r"$P_{out}$ ($\mu$W)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.3)
    print(f"\nOmega_RF/2pi = {omega_rf_mhz[i]:.4f} MHz (row {i}):")
    print(f"  n_peaks (all local maxima, no filter) = {n_peaks_total[i]}")
    print(f"  all peak positions/heights: {[(f'{p:.3f}', f'{h:.4f}') for p, h in all_peaks_record[i]]}")
    print(f"  top-2 separation = {top2_separation[i]:.4f} MHz  "
          f"(left={top2_left[i]:.4f}, right={top2_right[i]:.4f})")
    print(f"  valley position = {valley_pos[i]:.4f} MHz, valley height = {valley_height[i]:.4f} microW")

fig.suptitle("Fig.4 diagnostic -- Pout vs Delta_c at requested Omega_RF values "
             "(from existing fig4_data.npz, no new solves)", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUTPUT_DIR / "fig4_diagnostic_requested_spectra.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("\nSaved: fig4_diagnostic_requested_spectra.png")

# ------------------------------------------------------------
# 3. Delta_nu_AT vs Omega_RF, with theoretical Delta_nu_AT = Omega_RF
# ------------------------------------------------------------

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

valid = ~np.isnan(top2_separation)
ax1.plot(omega_rf_mhz[valid], top2_separation[valid], "-", color="red", linewidth=1.5,
         label=r"$\Delta\nu_{AT}$ from QuTiP (top-2 peak separation)")
ax1.plot(omega_rf_mhz, omega_rf_mhz, "--", color="gray", linewidth=1.2,
         label=r"Theoretical $\Delta\nu_{AT}=\Omega_{RF}/2\pi$")
ax1.set_xlabel(r"$\Omega_{RF}/2\pi$ (MHz)")
ax1.set_ylabel(r"$\Delta\nu_{AT}$ (MHz)")
ax1.set_title("Full range")
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)

# Zoom into the 0-6 MHz region specifically requested
zoom_mask = omega_rf_mhz <= 6.0
ax2.plot(omega_rf_mhz[zoom_mask & valid], top2_separation[zoom_mask & valid], "-o",
         color="red", markersize=3, linewidth=1.5, label=r"$\Delta\nu_{AT}$ from QuTiP")
ax2.plot(omega_rf_mhz[zoom_mask], omega_rf_mhz[zoom_mask], "--", color="gray", linewidth=1.2,
         label=r"Theoretical $\Delta\nu_{AT}=\Omega_{RF}/2\pi$")
merged_mask = zoom_mask & (n_peaks_total < 2)
if np.any(merged_mask):
    ax2.axvspan(0, omega_rf_mhz[merged_mask].max(), color="gray", alpha=0.15,
                label="Only 1 peak found (merged)")
ax2.set_xlabel(r"$\Omega_{RF}/2\pi$ (MHz)")
ax2.set_ylabel(r"$\Delta\nu_{AT}$ (MHz)")
ax2.set_title("Zoom: 0-6 MHz region")
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(OUTPUT_DIR / "fig4_diagnostic_deltanu_vs_omegarf.png", dpi=220, bbox_inches="tight")
plt.close(fig)
print("Saved: fig4_diagnostic_deltanu_vs_omegarf.png")

# ------------------------------------------------------------
# 5/6. Detailed investigation of the 0-6 MHz region
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DETAILED 0-6 MHz REGION (every grid row)")
print("=" * 70)
print(f"{'Omega_RF':>9} {'n_peaks':>8} {'sep(top2)':>10} {'theory':>8} {'diff':>8} {'valley_ht':>10}")
zoom_idx = np.where(omega_rf_mhz <= 6.0)[0]
for i in zoom_idx:
    sep = top2_separation[i]
    theory = omega_rf_mhz[i]
    diff = sep - theory if not np.isnan(sep) else np.nan
    vh = valley_height[i]
    print(f"{omega_rf_mhz[i]:9.4f} {n_peaks_total[i]:8d} "
          f"{sep if not np.isnan(sep) else float('nan'):10.4f} {theory:8.4f} "
          f"{diff if not np.isnan(diff) else float('nan'):8.4f} "
          f"{vh if not np.isnan(vh) else float('nan'):10.4f}")

# ------------------------------------------------------------
# 4. Classification of the observed behavior (report only, no
# modification of data)
# ------------------------------------------------------------

merged_rows = np.where(n_peaks_total < 2)[0]
split_rows = np.where(n_peaks_total >= 2)[0]

# The LOW-Omega_RF merge block is the contiguous run of merged rows
# starting at index 0 (the physically relevant "saturation" transition).
# A separate, unrelated artifact occurs at the very top of the grid
# (Omega_RF=20 MHz) where the theoretical peak position sits exactly at
# the +/-10 MHz Delta_c boundary, so find_peaks cannot register it as an
# interior local maximum -- that is a grid-edge effect, not part of the
# low-Omega_RF story, and is reported separately below.
low_merged = []
for idx in range(n_omega):
    if n_peaks_total[idx] < 2:
        low_merged.append(idx)
    else:
        break
merge_point = omega_rf_mhz[low_merged].max() if low_merged else None
first_split = omega_rf_mhz[len(low_merged)] if len(low_merged) < n_omega else None

high_edge_artifact_rows = [omega_rf_mhz[i] for i in merged_rows if i > (low_merged[-1] if low_merged else -1)]

# RMS deviation from theory, restricted to rows where 2 peaks exist
theory_all = omega_rf_mhz.copy()
dev = top2_separation - theory_all
dev_valid = dev[valid]
rms_dev = np.sqrt(np.nanmean(dev_valid ** 2))
max_dev = np.nanmax(np.abs(dev_valid))

print("\n" + "=" * 70)
print("SUMMARY / CLASSIFICATION OF OBSERVED BEHAVIOR")
print("=" * 70)
print(f"Highest Omega_RF with only 1 peak (merged): {merge_point} MHz")
print(f"Lowest Omega_RF with 2+ peaks found: {first_split} MHz")
print(f"RMS deviation of QuTiP separation from theoretical Omega_RF "
      f"(rows with 2+ peaks): {rms_dev:.4f} MHz")
print(f"Max deviation: {max_dev:.4f} MHz")
print(f"\nObserved behavior in this dataset:")
print(f"  - Two peaks are present for essentially the ENTIRE swept range above "
      f"{first_split} MHz, all the way to {omega_rf_mhz.max()} MHz.")
print(f"  - In that range, separation tracks the theoretical Delta_nu_AT=Omega_RF "
      f"line closely (RMS deviation {rms_dev:.4f} MHz, i.e. within about one "
      f"Delta_c grid step of {delta_c_mhz[1]-delta_c_mhz[0]:.4f} MHz) -- consistent "
      f"with simple theoretical linear AT splitting, NOT a separate linewidth-limited "
      f"plateau/saturation region.")
print(f"  - The only genuine 'disappearance of two peaks' in the raw data happens "
      f"below {merge_point} MHz, NOT around 5.5 MHz.")
print(f"  - No linewidth-limited unresolved region distinct from this literal merge "
      f"point was found in the raw peak count -- i.e. the data itself does not show "
      f"a low-Omega_RF SATURATION plateau the way the published Fig.4 appears to; "
      f"it shows theory-following separation down to {first_split} MHz, then an "
      f"abrupt drop to a single peak.")
if high_edge_artifact_rows:
    print(f"\n  (Unrelated grid-edge note: {len(high_edge_artifact_rows)} row(s) at the TOP "
          f"of the sweep -- Omega_RF = {high_edge_artifact_rows} MHz -- also show <2 peaks, "
          f"but this is because the theoretical peak position sits exactly at the +/-10 MHz "
          f"Delta_c boundary there, so find_peaks cannot register a peak at the array edge. "
          f"This is a grid-window artifact, not a physical merge, and is unrelated to the "
          f"low-Omega_RF transition reported above.)")

with open(OUTPUT_DIR / "fig4_saturation_diagnostic_report.txt", "w") as f:
    f.write("FIG.4 SATURATION DIAGNOSTIC -- POST-PROCESSING ONLY, NO NEW QUTIP SOLVES\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Source: fig4_data.npz (32361 pre-existing QuTiP steady-state solves)\n")
    f.write(f"Delta_c grid: [{delta_c_mhz.min()},{delta_c_mhz.max()}] MHz, "
            f"{len(delta_c_mhz)} pts, step {delta_c_mhz[1]-delta_c_mhz[0]:.4f} MHz\n")
    f.write(f"Omega_RF grid: [{omega_rf_mhz.min()},{omega_rf_mhz.max()}] MHz, "
            f"{len(omega_rf_mhz)} pts, step {omega_rf_mhz[1]-omega_rf_mhz[0]:.4f} MHz\n\n")
    f.write(f"Highest Omega_RF with only 1 peak (merged): {merge_point} MHz\n")
    f.write(f"Lowest Omega_RF with 2+ peaks found: {first_split} MHz\n")
    f.write(f"RMS deviation of QuTiP separation from theory (rows with 2+ peaks): "
            f"{rms_dev:.4f} MHz\n")
    f.write(f"Max deviation: {max_dev:.4f} MHz\n\n")
    f.write("CONCLUSION:\n")
    f.write("The raw QuTiP data shows theoretical linear AT splitting (Delta_nu_AT = \n")
    f.write("Omega_RF) essentially everywhere two peaks exist, with only grid-quantization\n")
    f.write("level deviation. There is no separate, linewidth-limited saturation region\n")
    f.write("in the raw peak-count data -- the two-peak structure persists down to\n")
    f.write(f"{first_split} MHz, then disappears abruptly. This is a smaller Omega_RF than\n")
    f.write("the paper's reported ~5.5 MHz saturation threshold. See fig4_saturation_diagnostic.py\n")
    f.write("output above (and the printed 0-6 MHz row-by-row table) for full detail.\n")

print("\nSaved: fig4_saturation_diagnostic_report.txt")
print("\nDONE. No QuTiP solves were performed; this was a pure re-analysis of fig4_data.npz.")
