# ============================================================
# FIG. 3 -- d = 0.38 mm COMPANION PLOTS (for direct comparison
# against the d = 0.76 mm fresh-build plots in this same folder)
#
# PROVENANCE (important, read before using these plots):
# This data was NOT recomputed in this session. It is the
# original, real QuTiP output from this project's very first
# Fig.3 reproduction (commit 6fafe3e, "Regenerate Fig. 3 from
# real from-scratch QuTiP rerun with corrected Pin"), which the
# user later deleted from the working tree as part of this
# session's cleanup. It was recovered read-only from git history
# (`git show 6fafe3e:fig3_results/fig3_quantum_response.npz`) --
# not re-run, not rescaled, not fabricated. Verified on recovery:
# Pin=9.7690 microW (exact match to the d=0.38mm legacy
# calibration used throughout this project), full 161x201 grid,
# same resolution as the d=0.76mm fresh-build data.
#
# This script only re-PLOTS that recovered real data with clean,
# separate, clearly-labeled outputs -- it performs no new QuTiP
# solves and does not merge d=0.38mm and d=0.76mm into the same
# plot (per the user's explicit request to keep them visually
# separate for the guide).
# ============================================================

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

OUTPUT_DIR = Path(__file__).resolve().parent
RECOVERED_DATA = OUTPUT_DIR / "d038mm_recovered_from_git" / "fig3_quantum_response_d038mm.npz"

data = np.load(RECOVERED_DATA)
delta_c_mhz = data["delta_c_mhz"]
omega_rf_mhz = data["omega_rf_mhz"]
Pout_surface = data["Pout_surface"]
Pin = float(data["Pin"])

print("=" * 70)
print("FIG. 3 -- d=0.38mm COMPANION PLOTS (recovered real data, see header)")
print("=" * 70)
print(f"Pin = {Pin/1e-6:.4f} microW")
print(f"Grid: {Pout_surface.shape}")
print(f"Peak Pout = {Pout_surface.max()/1e-6:.4f} microW")

# ------------------------------------------------------------
# 3D surface
# ------------------------------------------------------------

X, Y = np.meshgrid(delta_c_mhz, omega_rf_mhz)
Z = Pout_surface / 1e-6

fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")
surf = ax.plot_surface(X, Y, Z, cmap="turbo", linewidth=0, antialiased=True, alpha=0.93)
dc0 = np.argmin(np.abs(delta_c_mhz))
ax.plot(np.full_like(omega_rf_mhz, delta_c_mhz[dc0]), omega_rf_mhz, Z[:, dc0],
        color="red", linestyle="--", linewidth=2, marker="x", markersize=5,
        label=r"$P_{out}$ vs $\Omega_{RF}$ ($\Delta_c=0$)")
ax.set_xlabel(r"Coupling detuning, $\Delta_c/2\pi$ (MHz)")
ax.set_ylabel(r"RF Rabi frequency, $\Omega_{RF}/2\pi$ (MHz)")
ax.set_zlabel(r"$P_{out}$ ($\times10^{-6}$ W)")
ax.set_title(r"Fig. 3. $P_{out}$ versus $\Delta_c$ and $\Omega_{RF}$" "\n(d = 0.38 mm)")
ax.view_init(elev=28, azim=100)
fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1, label=r"$P_{out}$ ($\mu$W)")
ax.legend(fontsize=9, loc="upper left")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "fig3_pout_surface_d038mm.png", dpi=220, bbox_inches="tight")
plt.close(fig)
print("Saved: fig3_pout_surface_d038mm.png")

# ------------------------------------------------------------
# Top-down
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 7))
contour = ax.contourf(X, Y, Z, levels=60, cmap="turbo")
cbar = fig.colorbar(contour, ax=ax)
cbar.set_label(r"$P_{out}$ ($\times10^{-6}$ W)", fontsize=12)
ax.set_xlabel(r"Coupling detuning $\Delta_c/2\pi$ (MHz)", fontsize=12)
ax.set_ylabel(r"RF Rabi frequency $\Omega_{RF}/2\pi$ (MHz)", fontsize=12)
ax.set_title("Top-down view of Fig. 3 quantum response\n(d = 0.38 mm)", fontsize=14)
ax.grid(True, alpha=0.25)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "fig3_topdown_d038mm.png", dpi=220, bbox_inches="tight")
plt.close(fig)
print("Saved: fig3_topdown_d038mm.png")

# ------------------------------------------------------------
# Pout vs Omega_RF at Delta_c=0
# ------------------------------------------------------------

pout_vs_omega = Pout_surface[:, dc0]
fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(omega_rf_mhz, pout_vs_omega / 1e-6, color="red", linewidth=2.2, marker="x",
        markersize=5, markeredgewidth=1.4, markevery=max(1, len(omega_rf_mhz) // 35),
        label=r"$P_{out}$ vs $\Omega_{RF}$ ($\Delta_c=0$)")
ax.set_xlabel(r"$\Omega_{RF}/2\pi$ (MHz)")
ax.set_ylabel(r"$P_{out}$ ($\times10^{-6}$ W)")
ax.set_title(r"$P_{out}$ versus $\Omega_{RF}$ at $\Delta_c=0$" "\n(d = 0.38 mm)")
ax.grid(True, alpha=0.35)
ax.legend(fontsize=11)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "fig3_pout_vs_omegarf_d038mm.png", dpi=220, bbox_inches="tight")
plt.close(fig)
print("Saved: fig3_pout_vs_omegarf_d038mm.png")

# ------------------------------------------------------------
# EIT/AT spectrum slices
# ------------------------------------------------------------

selected_omega_mhz = [0.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0]
fig, ax = plt.subplots(figsize=(11, 6.5))
for omega_target in selected_omega_mhz:
    idx = int(np.argmin(np.abs(omega_rf_mhz - omega_target)))
    actual_omega = omega_rf_mhz[idx]
    ax.plot(delta_c_mhz, Pout_surface[idx, :] / 1e-6, linewidth=2.0,
            label=rf"$\Omega_{{RF}}/2\pi = {actual_omega:g}$ MHz")
ax.set_xlabel(r"Coupling detuning $\Delta_c/2\pi$ (MHz)")
ax.set_ylabel(r"$P_{out}$ ($\times10^{-6}$ W)")
ax.set_title("EIT/AT spectrum slices at fixed $\\Omega_{RF}$\n(d = 0.38 mm)")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "fig3_eit_at_slices_d038mm.png", dpi=220, bbox_inches="tight")
plt.close(fig)
print("Saved: fig3_eit_at_slices_d038mm.png")

print("\nDONE. All 4 plots saved as separate, d=0.38mm-labeled files (not merged with d=0.76mm plots).")
