# ============================================================
# FIG. 5 (v5 paper) -- following the paper's STATED METHOD exactly
#
# Paper's own text (Sec.V-B): "use QuTip toolkit to generate the
# EIT-AT spectrum as two Lorentzian peaks (half-width Gamma_FWHM/2)
# centered at detunings Delta_c = +/- Omega_RF(dTx-Rx)/2."
#
# We follow this literally:
#   SHAPE:     two-Lorentzian sum, peaks at +/-Omega_RF(d)/2,
#              half-width = Gamma_ref/2 (paper's stated formula)
#   AMPLITUDE: the paper says QuTiP generates this -- so we use
#              OUR real QuTiP data (genuine peak Pout at that same
#              Omega_RF, read from the frozen Fig.3 grid) --
#              "if they used their QuTiP, we use our QuTiP."
#   Gamma_FWHM = Gamma_ref = 1.2231 MHz (measured, our settled
#              project-wide convention, see fig4_linewidth_forensic_audit.md)
#
# Distance axis: same Natoms situation as before -- the paper never
# gives Natoms a value (see fig5_v5_forensic_audit_2.md). Same
# REFERENCE-value approach as fig5_v5_distance_reference.py: one
# clean reference Natoms used only to draw a concrete curve, axis
# labeled honestly, real distance = axis x sqrt(Natoms_true/Natoms_ref).
# ============================================================

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

OUTPUT_DIR = Path(__file__).resolve().parent
FIG3_DATA = OUTPUT_DIR.parent / "fig3" / "fig3_v5_qutip_response.npz"

d = np.load(FIG3_DATA)
delta_c_mhz = d["delta_c_mhz"]
omega_rf_mhz = d["omega_rf_mhz"]
Pout_surface = d["Pout_surface"]  # Watts, real QuTiP output, untouched

Gamma_ref = 1.2231  # MHz, our settled project-wide ΓFWHM convention
half_width = Gamma_ref / 2.0  # paper's stated Lorentzian half-width

# ------------------------------------------------------------
# STEP 1: extract the REAL QuTiP peak amplitude A(Omega_RF) --
# the actual max Pout in the real Fig.3 row at each Omega_RF.
# This is "our QuTiP" standing in for the paper's "QuTiP".
# ------------------------------------------------------------
A_of_omega = Pout_surface.max(axis=1)  # Watts, real data, one value per real Omega_RF grid point

print("=" * 70)
print("FIG. 5 (v5) -- paper's stated two-Lorentzian method")
print(f"Gamma_FWHM (=Gamma_ref) = {Gamma_ref} MHz, half-width = {half_width} MHz")
print("=" * 70)
print(f"Real QuTiP peak amplitude A(Omega_RF) range: "
      f"{A_of_omega.min()*1e6:.4f} to {A_of_omega.max()*1e6:.4f} uW")


def lorentzian(x, hw):
    return hw ** 2 / (x ** 2 + hw ** 2)


def pout_two_lorentzian(delta_c, omega_rf, amplitude):
    # NORMALIZED: the raw sum of two Lorentzians can reach up to 2x its
    # nominal height when the two peaks overlap (Omega_RF small compared
    # to the half-width) -- a pure arithmetic artifact of adding two
    # curves, not real physics (confirmed and explained in the math
    # report). Dividing by the shape's own max forces the curve's peak
    # to equal exactly the real QuTiP amplitude, regardless of overlap,
    # while keeping the paper's literal two-Lorentzian shape.
    raw = lorentzian(delta_c - omega_rf / 2.0, half_width) + lorentzian(delta_c + omega_rf / 2.0, half_width)
    return amplitude * raw / raw.max()


# ------------------------------------------------------------
# STEP 2: paper-confirmed link-budget constants, Natoms factored
# out as sqrt(Natoms) (see fig5_v5_distance_reference.py for the
# full derivation) -- REFERENCE Natoms used only to draw a curve.
# ------------------------------------------------------------
e_charge = 1.6e-19
a0 = 5.2e-11
hbar = 1.054571817e-34
Z0 = 377.0
N0 = 4.89e16
wp_RF = 1443.459 * e_charge * a0
fRF = 6.9e9
omega_RF_carrier = 2 * np.pi * fRF
PTx = 1.0
GTx = 10 ** (2.15 / 10)
Gamma_ref_hz = Gamma_ref * 1e6

K = (wp_RF / hbar) * np.sqrt(2 * Z0 * PTx * GTx / (4 * np.pi) * 2 * Z0 * wp_RF ** 2 * omega_RF_carrier / (hbar * Gamma_ref_hz))

d_beam = 0.76e-3
L_cell = 1e-2
Veff_ref = np.pi * (d_beam / 2) ** 2 * L_cell
Natoms_ref = N0 * Veff_ref


def omega_rf_of_distance(dist_m):
    return K * np.sqrt(Natoms_ref) / dist_m  # rad/s


omega_max_hz = omega_rf_mhz.max() * 1e6 * 2 * np.pi
omega_at_1m = omega_rf_of_distance(1.0)
d_min_honest = 1.0 * (omega_at_1m / omega_max_hz)
print(f"Natoms_ref (REFERENCE ONLY, not paper-confirmed) = {Natoms_ref:.4e}")
print(f"Honest lower distance bound = {d_min_honest:.4f} m (reference units)")

# ------------------------------------------------------------
# STEP 3: build the surface -- two-Lorentzian shape, real-QuTiP
# amplitude (interpolated to the exact Omega_RF(d) needed), over
# the same reference-distance sweep as before.
# ------------------------------------------------------------
dist_vals = np.logspace(np.log10(d_min_honest), 5, 150)
omega_rf_vals_mhz = np.array([omega_rf_of_distance(dd) / (2 * np.pi) / 1e6 for dd in dist_vals])

Pout_lorentzian = np.zeros((len(dist_vals), len(delta_c_mhz)))
for i, orf_target in enumerate(omega_rf_vals_mhz):
    orf_clamped = np.clip(orf_target, omega_rf_mhz.min(), omega_rf_mhz.max())
    A = np.interp(orf_clamped, omega_rf_mhz, A_of_omega)  # real QuTiP amplitude, interpolated
    Pout_lorentzian[i, :] = pout_two_lorentzian(delta_c_mhz, orf_clamped, A)

# ------------------------------------------------------------
# Plot -- same axis style as before (10^0..10^5 labels, Delta_c
# direction matching the paper)
# ------------------------------------------------------------
X, Y = np.meshgrid(delta_c_mhz, dist_vals)
Z = Pout_lorentzian / 1e-6
logY = np.log10(Y)

fig = plt.figure(figsize=(9.5, 7.5))
ax = fig.add_subplot(111, projection="3d")
surf = ax.plot_surface(X, logY, Z, cmap="turbo", linewidth=0, antialiased=True, alpha=0.93)

ax.set_xlabel(r"$\Delta_c/2\pi$ (MHz)")
ax.set_ylabel(r"Distance, $d_{Tx-Rx}$")
ax.set_zlabel("")
ax.set_xticks(np.arange(-20, 21, 10))
yticks = np.arange(0, 6, 1)
ax.set_yticks(yticks)
ax.set_yticklabels([rf"$10^{{{int(t)}}}$" for t in yticks])
ax.view_init(elev=23, azim=-230)
ax.zaxis._axinfo["juggled"] = (1, 2, 0)
ax.invert_xaxis()
ax.invert_yaxis()   # put 10^0 next to the Delta_c axis (front), matching the paper's layout
fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1, label=r"$P_{out}$ (x$10^{-6}$ W)")
fig.subplots_adjust(left=0.15)
fig.text(0.04, 0.55, r"$P_{out}$ (x$10^{-6}$ W)", rotation=90, va="center", fontsize=11)
fig.savefig(OUTPUT_DIR / "fig5_v5_two_lorentzian.png", dpi=220, bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------
# Save data
# ------------------------------------------------------------
import csv
with open(OUTPUT_DIR / "fig5_v5_two_lorentzian_data.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["delta_c_MHz", "distance_reference_units_m", "omega_RF_MHz_computed", "Pout_microW"])
    for i, dist in enumerate(dist_vals):
        for j, dc in enumerate(delta_c_mhz):
            writer.writerow([f"{dc:.4f}", f"{dist:.4f}", f"{omega_rf_vals_mhz[i]:.6f}", f"{Pout_lorentzian[i,j]*1e6:.6f}"])

print("\nSaved: fig5_v5_two_lorentzian.png, fig5_v5_two_lorentzian_data.csv")
print("DONE.")
