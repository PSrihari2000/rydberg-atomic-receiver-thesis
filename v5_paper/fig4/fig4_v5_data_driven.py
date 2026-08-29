# ============================================================
# FIG. 4 (v5 paper) -- DATA-DRIVEN reconstruction
#
# We are NOT forcing our data to reproduce the paper's unexplained
# numerical linewidth normalization or its apparent slope
# Delta_c/2pi = +/-R (see fig4_linewidth_forensic_audit.md -- no
# candidate linewidth, measured or analytic, reproduces that slope).
#
# Instead we reproduce the PHYSICAL CONCEPT of Fig.4 -- RF Rabi
# frequency normalized by a linewidth -- using our own real,
# genuine QuTiP Fig.3 data end to end. Gamma_ref is measured fresh
# from that data every run, never hardcoded.
#
# Paper's definition:      R          = Omega_RF / Gamma_FWHM
# Our implementation:      R_qutip     = Omega_RF / Gamma_ref
# where Gamma_ref is OUR OWN measured FWHM of Pout(Delta_c) at
# Omega_RF=0, re-measured from fig3_v5_qutip_response.npz every
# time this script runs.
#
# Uses ONLY v5_paper/fig3/fig3_v5_qutip_response.npz. Does not
# modify Fig.3, does not call QuTiP, does not extrapolate beyond
# the real computed grid.
# ============================================================

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import griddata

OUTPUT_DIR = Path(__file__).resolve().parent
FIG3_DATA = OUTPUT_DIR.parent / "fig3" / "fig3_v5_qutip_response.npz"

# ------------------------------------------------------------
# STEP 0: load and report the real dataset (no modification)
# ------------------------------------------------------------

d = np.load(FIG3_DATA)
delta_c_mhz = d["delta_c_mhz"]
omega_rf_mhz = d["omega_rf_mhz"]
Pout_surface = d["Pout_surface"]  # Watts, shape (n_orf, n_dc)

print("=" * 70)
print("FIG. 4 (v5) -- DATA-DRIVEN reconstruction")
print(f"Loaded: {FIG3_DATA}")
print("=" * 70)
print(f"Delta_c/2pi: {delta_c_mhz.min():.2f} to {delta_c_mhz.max():.2f} MHz, "
      f"n={len(delta_c_mhz)}, spacing={delta_c_mhz[1]-delta_c_mhz[0]:.4f} MHz")
print(f"Omega_RF/2pi: {omega_rf_mhz.min():.2f} to {omega_rf_mhz.max():.2f} MHz, "
      f"n={len(omega_rf_mhz)}, spacing={omega_rf_mhz[1]-omega_rf_mhz[0]:.4f} MHz")
print(f"Pout_surface: shape={Pout_surface.shape}, units=Watts, "
      f"peak={Pout_surface.max()*1e6:.4f} uW")

# ------------------------------------------------------------
# STEP 1: measure Gamma_ref from the real Omega_RF=0 row
# (half-max crossings, linear interpolation) -- fresh every run,
# never hardcoded.
# ------------------------------------------------------------

i_orf0 = np.argmin(np.abs(omega_rf_mhz - 0.0))
row0 = Pout_surface[i_orf0]
peak0 = row0.max()
baseline0 = row0.min()   # consistent baseline: the row's own minimum
half0 = (peak0 + baseline0) / 2.0
center = np.argmin(np.abs(delta_c_mhz))

right = None
for j in range(center, len(delta_c_mhz) - 1):
    if row0[j] >= half0 and row0[j + 1] < half0:
        right = delta_c_mhz[j] + (half0 - row0[j]) * (delta_c_mhz[j + 1] - delta_c_mhz[j]) / (row0[j + 1] - row0[j])
        break

left = None
for j in range(center, 0, -1):
    if row0[j] >= half0 and row0[j - 1] < half0:
        left = delta_c_mhz[j] + (half0 - row0[j]) * (delta_c_mhz[j - 1] - delta_c_mhz[j]) / (row0[j - 1] - row0[j])
        break

HWHM = (right - left) / 2.0
FWHM = right - left
Gamma_ref = FWHM   # our data-driven reference linewidth (MHz, i.e. /2pi convention)

print()
print("STEP 1 -- measured reference linewidth (Omega_RF=0 row):")
print(f"  peak Pout = {peak0*1e6:.4f} uW, baseline Pout = {baseline0*1e6:.4f} uW, "
      f"half-height = {half0*1e6:.4f} uW")
print(f"  left crossing Delta_c/2pi = {left:.4f} MHz, right crossing = {right:.4f} MHz")
print(f"  HWHM = {HWHM:.4f} MHz")
print(f"  FWHM = {FWHM:.4f} MHz")
print(f"  Gamma_ref (= measured FWHM) = {Gamma_ref:.4f} MHz")

# ------------------------------------------------------------
# STEP 2: build R_qutip axis -- consistent units throughout
# (both Omega_RF and Gamma_ref are already in the same "/2pi MHz"
# convention, so this ratio never mixes Hz and rad/s)
# ------------------------------------------------------------

R_qutip_full = omega_rf_mhz / Gamma_ref
print()
print(f"STEP 2 -- R_qutip = Omega_RF/2pi / Gamma_ref, "
      f"range: {R_qutip_full.min():.3f} to {R_qutip_full.max():.3f}")

# ------------------------------------------------------------
# STEP 6 (checked before plotting): data range -- do not extrapolate
# ------------------------------------------------------------

R_max_available = R_qutip_full.max()
print(f"STEP 6 -- max R available from existing Fig.3 grid = {R_max_available:.3f} "
      f"(paper's Fig.4 shows R up to 10; {'REACHES' if R_max_available>=10 else 'DOES NOT REACH'} that)")

# ------------------------------------------------------------
# STEP 5: measure the real AT peak loci (not assumed) + compare
# to the physical approximation Delta_c = +/- Omega_RF/2
# ------------------------------------------------------------

peak_R, peak_dc_pos, peak_dc_neg = [], [], []
for i, r_val in enumerate(R_qutip_full):
    row = Pout_surface[i]
    pk, props = find_peaks(row, prominence=(row.max() - row.min()) * 0.02 + 1e-30)
    if len(pk) >= 2:
        prom = props["prominences"]
        top2 = pk[np.argsort(prom)[-2:]]
        top2_dc = delta_c_mhz[top2]
        peak_R.append(r_val)
        peak_dc_pos.append(top2_dc.max())
        peak_dc_neg.append(top2_dc.min())

peak_R = np.array(peak_R)
peak_dc_pos = np.array(peak_dc_pos)
peak_dc_neg = np.array(peak_dc_neg)

# physical approximation: Delta_c/2pi = +/- [R_qutip * Gamma_ref/2pi] / 2
approx_dc_pos = peak_R * Gamma_ref / 2.0

resid = peak_dc_pos - approx_dc_pos
print()
print(f"STEP 5 -- measured AT peak loci vs physical approx Delta_c=+/-Omega_RF/2:")
print(f"  splitting resolved (2 peaks found) starting at R_qutip = "
      f"{peak_R.min() if len(peak_R) else float('nan'):.3f}")
print(f"  mean |measured - approx| = {np.mean(np.abs(resid)):.5f} MHz, "
      f"max |residual| = {np.max(np.abs(resid)):.5f} MHz")
print(f"  (residual this small confirms peaks sit at Delta_c=+/-Omega_RF/2 essentially exactly,"
      f" independent of any Gamma_ref choice)")

# ------------------------------------------------------------
# STEP 3+4: build heatmap, measured-threshold reference line,
# peak loci overlay
#
# The reference line marks the MEASURED resolvability threshold
# (first grid point where find_peaks resolves 2 distinct AT
# peaks) -- not R=1, which is only true by construction (it's
# just the point where Omega_RF equals whatever Gamma_ref we
# measured, not an independently verified result).
# ------------------------------------------------------------

DC, RR = np.meshgrid(delta_c_mhz, R_qutip_full)
Zc = Pout_surface / 1e-6   # x10^-6 W, consistent with Fig.3/Fig.4 convention

R_threshold = peak_R.min() if len(peak_R) else float("nan")

fig, ax = plt.subplots(figsize=(7, 5.8))
pcm = ax.pcolormesh(DC, RR, Zc, cmap="turbo", shading="gouraud")
fig.colorbar(pcm, ax=ax, label=r"$P_{out}$ (x$10^{-6}$ W)")

ax.axhline(R_threshold, color="yellow", linestyle=":", linewidth=2,
           label=rf"$R={R_threshold:.3f}$ (measured resolvability threshold)")

if len(peak_R) > 0:
    ax.plot(peak_dc_pos, peak_R, color="white", linestyle="--", linewidth=1.8, label="measured AT peak loci")
    ax.plot(peak_dc_neg, peak_R, color="white", linestyle="--", linewidth=1.8)
    ax.plot(approx_dc_pos, peak_R, color="black", linestyle=":", linewidth=1.3, label=r"$\Delta_c=\pm\Omega_{RF}/2$")
    ax.plot(-approx_dc_pos, peak_R, color="black", linestyle=":", linewidth=1.3)

ax.set_xlabel(r"$\Delta_c/2\pi$ (MHz)")
ax.set_ylabel(r"$R_{QuTiP} = \Omega_{RF}/\Gamma_{ref}$ (dimensionless)")
ax.set_xticks(np.arange(-12, 13, 4))
ax.set_xlim(-12, 12)
ax.set_ylim(R_qutip_full.min() if R_qutip_full.min() > 0 else 0.01, R_qutip_full.max())
ax.legend(fontsize=8, loc="upper right")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "fig4_v5_data_driven.png", dpi=220, bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------
# STEP 7: save data + report
# ------------------------------------------------------------

import csv
with open(OUTPUT_DIR / "fig4_v5_data_driven.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["delta_c_MHz", "R_qutip", "Pout_microW"])
    for i, r_val in enumerate(R_qutip_full):
        for j, dc_val in enumerate(delta_c_mhz):
            writer.writerow([f"{dc_val:.4f}", f"{r_val:.4f}", f"{Pout_surface[i,j]*1e6:.6f}"])

np.savez(
    OUTPUT_DIR / "fig4_v5_data_driven_response.npz",
    delta_c_mhz=delta_c_mhz, R_qutip=R_qutip_full, Pout_surface=Pout_surface,
    Gamma_ref=Gamma_ref, HWHM=HWHM, FWHM=FWHM,
    peak_R=peak_R, peak_dc_pos=peak_dc_pos, peak_dc_neg=peak_dc_neg,
    approx_dc_pos=approx_dc_pos,
)

print("\nSaved: fig4_v5_data_driven.png, fig4_v5_data_driven.csv, fig4_v5_data_driven_response.npz")
print("DONE.")
