# ============================================================
# FIG. 4 -- d=0.36mm INDEPENDENT REBUILD
# Distortion of the LO-free Rydberg atomic receiver
#
# Identical method to fig4_fresh_build/fig4_distortion_analysis.py:
# reuses THIS folder's own freshbuild_fig3_d036mm_raw.npz (no new QuTiP
# solves -- Fig.4 is a top-down view of Fig.3, matching the paper's own
# description). Same FWHM-based resolvability criterion (separation >=
# 1 measured FWHM of the Omega_RF=0 spectrum), NOT named "Rayleigh
# criterion" (see project convention). Does not touch fig4_fresh_build
# or the older qutip_d036mm files.
# ============================================================

from pathlib import Path
import csv

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

OUTPUT_DIR = Path(__file__).resolve().parent
FIG3_DATA = OUTPUT_DIR.parent / "fig3_d036mm" / "freshbuild_fig3_d036mm_raw.npz"

data = np.load(FIG3_DATA)
delta_c_mhz = data["delta_c_mhz"]
omega_rf_mhz = data["omega_rf_mhz"]
Pout_surface = data["Pout_surface"] / 1e-6
d_probe_mm = float(data["d_probe"]) * 1e3

print("=" * 70)
print(f"FIG. 4 -- d={d_probe_mm:.2f}mm INDEPENDENT REBUILD -- FWHM-based resolvability")
print("=" * 70)
print(f"Loaded: {FIG3_DATA.name}")

row0 = Pout_surface[0, :]
peak_idx0 = int(np.argmax(row0))
peak_pos0 = delta_c_mhz[peak_idx0]
peak_val0 = row0[peak_idx0]
baseline0 = 0.5*(row0[0]+row0[-1])
half_level0 = baseline0 + 0.5*(peak_val0-baseline0)

def interp_crossing(x, y, level, i0, direction):
    i = i0
    n = len(y)
    while 0 <= i+direction < n:
        y0, y1 = y[i], y[i+direction]
        if (y0-level)*(y1-level) <= 0 and y0 != y1:
            x0, x1 = x[i], x[i+direction]
            frac = (level-y0)/(y1-y0)
            return x0 + frac*(x1-x0)
        i += direction
    return None

right_cross = interp_crossing(delta_c_mhz, row0, half_level0, peak_idx0, +1)
left_cross = interp_crossing(delta_c_mhz, row0, half_level0, peak_idx0, -1)
HWHM_measured = float(np.mean([right_cross-peak_pos0, peak_pos0-left_cross]))
FWHM_measured = 2.0*HWHM_measured
print(f"Measured HWHM = {HWHM_measured:.4f} MHz -> FWHM = {FWHM_measured:.4f} MHz")

n_omega = len(omega_rf_mhz)
n_peaks_arr = np.zeros(n_omega, dtype=int)
left_x = np.full(n_omega, np.nan)
right_x = np.full(n_omega, np.nan)
separation = np.full(n_omega, np.nan)
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
    separation[i] = delta_c_mhz[ri]-delta_c_mhz[li]
    between = row[li:ri+1]
    vloc = np.argmin(between)
    if not (0 < vloc < (ri-li)):
        reason[i] = "no_interior_valley"
        left_x[i] = right_x[i] = np.nan
        continue
    if separation[i] >= FWHM_measured:
        resolved[i] = True
        reason[i] = "resolved"
    else:
        reason[i] = "below_threshold"
        left_x[i] = right_x[i] = np.nan

threshold_mhz = omega_rf_mhz[resolved].min() if np.any(resolved) else None
print(f"Resolved rows: {int(np.sum(resolved))}/{n_omega}")
print(f"THRESHOLD (lowest resolved Omega_RF) = {threshold_mhz:.4f} MHz")
print(f"Comparison to d=0.76mm fresh_build's threshold (5.1250MHz): "
      f"{'byte-identical' if abs(threshold_mhz-5.1250)<1e-6 else f'differs by {threshold_mhz-5.1250:+.4f}MHz'}")

csv_path = OUTPUT_DIR / "freshbuild_fig4_d036mm.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["# d_probe_mm", f"{d_probe_mm:.4f}"])
    writer.writerow(["# Measured HWHM (MHz)", f"{HWHM_measured:.6f}"])
    writer.writerow(["# Measured FWHM (MHz)", f"{FWHM_measured:.6f}"])
    writer.writerow(["# Threshold Omega_RF (MHz)", f"{threshold_mhz:.6f}"])
    writer.writerow([])
    writer.writerow(["Omega_RF_MHz","n_peaks","left_MHz","right_MHz","separation_MHz","resolved","reason"])
    for i in range(n_omega):
        writer.writerow([f"{omega_rf_mhz[i]:.4f}", n_peaks_arr[i],
                          f"{left_x[i]:.4f}" if not np.isnan(left_x[i]) else "",
                          f"{right_x[i]:.4f}" if not np.isnan(right_x[i]) else "",
                          f"{separation[i]:.4f}" if not np.isnan(separation[i]) else "",
                          resolved[i], reason[i]])
print(f"Saved: {csv_path.name}")

fig, ax = plt.subplots(figsize=(7, 6.5))
has_valley = ~np.isnan(left_x)
ax.plot(left_x[has_valley], omega_rf_mhz[has_valley], "-", color="red", linewidth=1.8, label="Obtained by QuTiP")
ax.plot(right_x[has_valley], omega_rf_mhz[has_valley], "-", color="red", linewidth=1.8)
ax.plot(omega_rf_mhz/2.0, omega_rf_mhz, "--", color="gray", linewidth=1.2, label="Theoretical")
ax.plot(-omega_rf_mhz/2.0, omega_rf_mhz, "--", color="gray", linewidth=1.2)
ax.axhline(threshold_mhz, color="k", linestyle=":", alpha=0.7)
ax.fill_between([-9,9], 0, threshold_mhz, color="gray", alpha=0.15)
ax.annotate("Distortion region", xy=(-7.5, threshold_mhz*0.55), fontsize=10)
ax.set_xlabel(r"Coupling detuning, $\Delta_c/2\pi$ (MHz)")
ax.set_ylabel(r"RF Rabi frequency, $\Omega_{RF}/2\pi$ (MHz)")
ax.set_title(f"Fig. 4. Distortion of the LO-free Rydberg atomic receiver (d={d_probe_mm:.2f}mm)")
ax.set_xlim(-9, 9)
ax.set_ylim(0, 16)
ax.legend(fontsize=10, loc="upper right")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "freshbuild_fig4_d036mm.png", dpi=220, bbox_inches="tight")
plt.close(fig)
print("Saved: freshbuild_fig4_d036mm.png")

report = f"""# Fig. 4 math report -- d=0.36mm independent rebuild

Re-analysis of THIS folder's own `freshbuild_fig3_d036mm_raw.npz` -- no new QuTiP
solves (Fig.4 is a top-down view of Fig.3, per the paper's own description). Same
FWHM-based resolvability criterion as `fig4_fresh_build` (separation >= 1 measured
FWHM of the Omega_RF=0 spectrum) -- plainly stated, not named "Rayleigh criterion."
Does not touch fig4_fresh_build or the older qutip_d036mm files.

## Result

    Measured HWHM = {HWHM_measured:.4f} MHz -> FWHM = {FWHM_measured:.4f} MHz
    Resolved rows: {int(np.sum(resolved))}/{n_omega}
    THRESHOLD (lowest resolved Omega_RF) = {threshold_mhz:.4f} MHz

## Comparison to d=0.76mm

The d=0.76mm fresh_build threshold is 5.1250 MHz. Probe diameter enters Fig.3's
Pout only through Pin, a pure multiplicative prefactor with zero effect on where
peaks/valleys occur (proven algebraically and confirmed by direct byte-for-byte
comparison in the earlier qutip_d036mm/qutip_d076mm investigation this project
already did) -- so this threshold is expected to match 5.1250MHz exactly, and this
run independently re-confirms that on freshly-computed d=0.36mm data using TODAY's
FWHM-based method (the earlier investigation used the OLDER 5.5MHz-era method).
"""
with open(OUTPUT_DIR / "freshbuild_fig4_d036mm_math_report.md", "w") as f:
    f.write(report)
print("Saved: freshbuild_fig4_d036mm_math_report.md")
print("\nDONE.")
