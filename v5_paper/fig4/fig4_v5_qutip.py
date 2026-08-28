# ============================================================
# FIG. 4 (v5 paper) -- Probe transmission heatmap vs Delta_c
# and the dimensionless ratio R = Omega_RF / Gamma_FWHM (Eq.48)
#
# REUSES the real, frozen Fig.3 data (fig3_v5_qutip_response.npz)
# -- NO new QuTiP sweep. Fig.4 is the same physical quantity as
# Fig.3 (identical Hamiltonian, no Doppler, Pin=20.7uW direct,
# real qutip.steadystate() throughout), just re-axised from raw
# Omega_RF/2pi (MHz) to R, plus two overlay annotations:
#   - yellow dotted line at R=1 (paper's resolvability threshold)
#   - white dashed lines tracing the AT peak positions vs R
#
# Gamma_FWHM is not given as a closed-form formula anywhere in the
# v5 main text -- it is only ever called "the FWHM of the EIT
# spectrum". Measured directly from our own real Omega_RF=0 row
# (half-max crossings, linear interpolation) rather than assumed
# from an analytic small-signal formula -- same practice already
# established for this project's v1 Fig.4 rebuild.
#
# The white dashed peak-loci lines are MEASURED from our own real
# data (via scipy.signal.find_peaks per R-row), not assumed from
# the paper's caption text "Delta_c/2pi = +/-R" -- this lets us
# honestly check whether that relationship actually holds here.
# ============================================================

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

OUTPUT_DIR = Path(__file__).resolve().parent
FIG3_DATA = OUTPUT_DIR.parent / "fig3" / "fig3_v5_qutip_response.npz"

d = np.load(FIG3_DATA)
delta_c_mhz = d["delta_c_mhz"]
omega_rf_mhz = d["omega_rf_mhz"]
Pout_surface = d["Pout_surface"]   # shape (n_orf, n_dc), Watts

print("=" * 70)
print("FIG. 4 (v5) -- heatmap vs Delta_c and R = Omega_RF/Gamma_FWHM")
print(f"Reusing real data from: {FIG3_DATA.name}")
print("=" * 70)

# ------------------------------------------------------------
# STEP 1: measure Gamma_FWHM from the real Omega_RF=0 row
# ------------------------------------------------------------

i_orf0 = np.argmin(np.abs(omega_rf_mhz - 0.0))
row0 = Pout_surface[i_orf0]
peak0 = row0.max()
half0 = peak0 / 2.0
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

gamma_fwhm = right - left
print(f"Measured Gamma_FWHM (from Omega_RF=0 row, half-max crossings) = {gamma_fwhm:.4f} MHz")

# ------------------------------------------------------------
# STEP 2: build R axis, crop to the paper's plotted window
# (Delta_c/2pi in [-8,8], R in [1, max available])
# ------------------------------------------------------------

R_full = omega_rf_mhz / gamma_fwhm
r_mask = R_full >= 1.0
dc_mask = np.abs(delta_c_mhz) <= 8.0

R = R_full[r_mask]
dc = delta_c_mhz[dc_mask]
Z = Pout_surface[np.ix_(r_mask, dc_mask)]   # (n_R, n_dc), Watts

print(f"R range in cropped window: {R.min():.3f} to {R.max():.3f}")
print(f"Delta_c/2pi range: {dc.min():.2f} to {dc.max():.2f} MHz")
print(f"Peak Pout in this window = {Z.max()/1e-6:.4f} microW")

# ------------------------------------------------------------
# STEP 3: measure the real AT peak loci vs R (not assumed)
# ------------------------------------------------------------

peak_R = []
peak_dc_pos = []
peak_dc_neg = []
for i, r_val in enumerate(R):
    row = Z[i]
    pk, props = find_peaks(row, prominence=row.max() * 0.02)
    if len(pk) >= 2:
        # take the two most prominent peaks, split by sign of Delta_c
        prom = props["prominences"]
        top2 = pk[np.argsort(prom)[-2:]]
        top2_dc = dc[top2]
        peak_R.append(r_val)
        peak_dc_pos.append(top2_dc.max())
        peak_dc_neg.append(top2_dc.min())

peak_R = np.array(peak_R)
peak_dc_pos = np.array(peak_dc_pos)
peak_dc_neg = np.array(peak_dc_neg)

print(f"Two-peak splitting resolved (measured) starting at R = {peak_R.min() if len(peak_R) else float('nan'):.3f}")

# Cross-check against the paper's own caption claim "Delta_c/2pi = +/-R"
if len(peak_R) > 5:
    # simple linear fit slope of |measured peak position| vs R
    slope = np.polyfit(peak_R, peak_dc_pos, 1)[0]
    print(f"Measured slope of peak locus vs R (linear fit) = {slope:.4f} "
          f"(paper's caption literally states slope=1)")

# ------------------------------------------------------------
# STEP 4: save cropped data + plot
# ------------------------------------------------------------

import csv
with open(OUTPUT_DIR / "fig4_v5_data.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["delta_c_MHz", "R", "Pout_microW"])
    for i, r_val in enumerate(R):
        for j, dc_val in enumerate(dc):
            writer.writerow([f"{dc_val:.4f}", f"{r_val:.4f}", f"{Z[i,j]*1e6:.6f}"])

np.savez(
    OUTPUT_DIR / "fig4_v5_qutip_response.npz",
    delta_c_mhz=dc, R=R, Pout_surface=Z, gamma_fwhm=gamma_fwhm,
    peak_R=peak_R, peak_dc_pos=peak_dc_pos, peak_dc_neg=peak_dc_neg,
)

DC, RR = np.meshgrid(dc, R)
Zc = Z / 1e-6   # match paper's own x10^-6 colorbar convention

fig, ax = plt.subplots(figsize=(6.5, 5.5))
pcm = ax.pcolormesh(DC, RR, Zc, cmap="turbo", shading="gouraud")
fig.colorbar(pcm, ax=ax, label=r"$P_{out}$ (x$10^{-6}$ W)")

ax.axhline(1.0, color="yellow", linestyle=":", linewidth=2)
ax.text(6.0, 1.3, r"$\mathcal{R}=1$", color="yellow", fontsize=10)

if len(peak_R) > 0:
    ax.plot(peak_dc_pos, peak_R, color="white", linestyle="--", linewidth=1.8)
    ax.plot(peak_dc_neg, peak_R, color="white", linestyle="--", linewidth=1.8)

ax.set_xlabel(r"$\Delta_c/2\pi$ (MHz)")
ax.set_ylabel(r"$\mathcal{R} = \Omega_{RF}/\Gamma_{FWHM}$")
ax.set_xlim(-8, 8)
ax.set_ylim(1.0, R.max())
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "fig4_v5_heatmap.png", dpi=220, bbox_inches="tight")
plt.close(fig)

print("\nSaved: fig4_v5_qutip_response.npz, fig4_v5_data.csv, fig4_v5_heatmap.png")
print("DONE.")
