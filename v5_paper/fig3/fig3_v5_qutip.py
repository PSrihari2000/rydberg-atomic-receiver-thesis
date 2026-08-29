# ============================================================
# FIG. 3 (v5 paper) -- Pout(Delta_c, Omega_RF), FINAL BUILD
# REAL QuTiP, NO Doppler averaging.
#
# History (see fig3_v5_math_report.md for full detail):
#   1. First built WITH Doppler averaging (Eqs.9-11) -- initially
#      looked closer to the paper's peak scale in a quick,
#      under-converged (N_VZ=21-31) test.
#   2. Convergence check (this session) proved that test was
#      badly wrong -- Doppler integral only converges around
#      N_VZ~400+ (single-photon Doppler width ~158MHz vs the
#      ~MHz-scale EIT/AT structure being integrated over).
#   3. Redone at full convergence (N_VZ=401): the fast numpy
#      solver and REAL qutip.steadystate() then agreed to
#      <0.12% (genuine cross-validation, see
#      fig3_v5_qutip_validation_results.txt) -- but the
#      converged result erases the AT-splitting entirely
#      (0.0% dip at Delta_c=0 for ORF=1/4/7MHz) -- contradicts
#      the paper's own plotted Fig.3, which shows clear splitting.
#   4. Checked the paper's own text directly: "Doppler"/
#      "velocity"/"Maxwell-Boltzmann" appear ONLY once, in
#      Sec.II-C (Eqs.9-11) -- nowhere in Sec.V-A's simulation
#      parameters, and Gamma_FWHM (used everywhere else, Eq.17/
#      48/50) is defined purely from Appendix B's analytic/
#      natural linewidth, no Doppler dependence. Same pattern as
#      the paper's own explicit "Remark 1" (BBR decoherence
#      acknowledged physically, deliberately NOT built into the
#      master equation, called future work) -- Doppler broadening
#      reads the same way: theoretical completeness, not applied.
#   5. CONCLUSION: revert to no-Doppler, matching v1's original
#      approach and the paper's own actual (undocumented but
#      inferrable) methodology. This file is that final build,
#      using REAL qutip.steadystate() for every grid point (no
#      Doppler integral needed now, so the full grid is cheap
#      enough to do with genuine QuTiP throughout).
#
# Other decisions unchanged: Pin=20.7uW direct (Eq.3/diameter
# bypassed), wp_12=(2.5*e*a0)^2 (paper's pre-squared convention).
# ============================================================

import time
from pathlib import Path

import numpy as np
import qutip as qt
import matplotlib.pyplot as plt

OUTPUT_DIR = Path(__file__).resolve().parent

# ------------------------------------------------------------
# PHYSICAL CONSTANTS
# ------------------------------------------------------------

e_charge = 1.6e-19
a0 = 5.2e-11
hbar = 1.054571817e-34
eps0 = 8.854e-12
Z0 = 377.0

# ------------------------------------------------------------
# ATOMIC / LASER PARAMETERS (paper Sec. V-A, v5 values)
# ------------------------------------------------------------

L_cell = 1.0e-2
N0 = 4.89e10 * 1e6

gamma2 = 2.0 * np.pi * 5.2e6
gamma3 = 2.0 * np.pi * 3.9e3
gamma4 = 2.0 * np.pi * 1.7e3

Omega_p = 2.0 * np.pi * 8.0e6
Omega_c = 2.0 * np.pi * 1.0e6

wp_12 = (2.5 * e_charge * a0) ** 2

lambda_p = 852e-9
kp = 2.0 * np.pi / lambda_p

Pin = 20.7e-6
C0 = -2.0 * N0 * wp_12 / (eps0 * hbar * Omega_p)

print("=" * 70)
print("FIG. 3 (v5) -- FINAL build: REAL QuTiP, no Doppler averaging")
print("=" * 70)
print(f"qutip version = {qt.__version__}")
print(f"N0    = {N0:.4e} m^-3")
print(f"Pin   = {Pin*1e6:.4f} microW (paper-stated, direct)")
print(f"C0    = {C0:.6e}")
print("=" * 70)


def basis4():
    return [qt.basis(4, i) for i in range(4)]


def hamiltonian(Omega_RF, Delta_p=0.0, Delta_c=0.0, Delta_RF=0.0):
    k1, k2, k3, k4 = basis4()
    H = qt.qzero(4)
    H += -Delta_p * k2 * k2.dag()
    H += -(Delta_p + Delta_c) * k3 * k3.dag()
    H += -(Delta_p + Delta_c + Delta_RF) * k4 * k4.dag()
    H += (Omega_p / 2.0) * (k1 * k2.dag() + k2 * k1.dag())
    H += (Omega_c / 2.0) * (k2 * k3.dag() + k3 * k2.dag())
    H += (Omega_RF / 2.0) * (k3 * k4.dag() + k4 * k3.dag())
    return H


def collapse_ops():
    k1, k2, k3, k4 = basis4()
    return [
        np.sqrt(gamma2) * k1 * k2.dag(),
        np.sqrt(gamma3) * k2 * k3.dag(),
        np.sqrt(gamma4) * k3 * k4.dag(),
    ]


C_OPS = collapse_ops()


def pout_at(Omega_RF, Delta_c, Delta_p=0.0, Delta_RF=0.0):
    H = hamiltonian(Omega_RF, Delta_p, Delta_c, Delta_RF)
    rho_ss = qt.steadystate(H, C_OPS, method="direct")
    rho21 = complex(rho_ss[1, 0])
    chi = C0 * rho21
    return Pin * np.exp(-kp * L_cell * np.imag(chi))


# ------------------------------------------------------------
# SANITY CHECK
# ------------------------------------------------------------

t0 = time.time()
Pout_test = pout_at(2 * np.pi * 6e6, 0.0)
print(f"\nSanity check @ Omega_RF/2pi=6MHz, Delta_c=0 (REAL qutip.steadystate): "
      f"Pout = {Pout_test/1e-6:.4f} microW ({time.time()-t0:.3f}s)")

# ------------------------------------------------------------
# FULL GRID SWEEP -- real QuTiP throughout, no Doppler
# ------------------------------------------------------------

delta_c_mhz = np.linspace(-20.0, 20.0, 201)
omega_rf_mhz = np.linspace(0.0, 12.0, 97)

delta_c_vals = 2.0 * np.pi * delta_c_mhz * 1e6
omega_rf_vals = 2.0 * np.pi * omega_rf_mhz * 1e6

n_total = len(delta_c_mhz) * len(omega_rf_mhz)
print(f"\nGrid: {len(omega_rf_mhz)} x {len(delta_c_mhz)} = {n_total} REAL qutip.steadystate() solves")

Pout_surface = np.zeros((len(omega_rf_mhz), len(delta_c_mhz)))

t0 = time.time()
for i, Omega_RF in enumerate(omega_rf_vals):
    for j, Delta_c in enumerate(delta_c_vals):
        Pout_surface[i, j] = pout_at(Omega_RF, Delta_c)
    elapsed = time.time() - t0
    rate = (i + 1) / elapsed
    eta = (len(omega_rf_vals) - i - 1) / rate if rate > 0 else float("nan")
    print(f"\rRow {i+1:3d}/{len(omega_rf_vals)}  Omega_RF/2pi={omega_rf_mhz[i]:6.2f} MHz  "
          f"elapsed={elapsed:6.1f}s  ETA={eta:6.1f}s", end="")
dt = time.time() - t0
print(f"\n\nSweep completed in {dt:.1f}s ({n_total} REAL qutip solves, {dt/n_total*1000:.2f} ms/solve avg)")

peak_val = Pout_surface.max()
peak_idx = np.unravel_index(np.argmax(Pout_surface), Pout_surface.shape)
edge_val = Pout_surface[0, 0]
print(f"Peak Pout = {peak_val/1e-6:.4f} microW at "
      f"Omega_RF/2pi={omega_rf_mhz[peak_idx[0]]:.2f} MHz, Delta_c/2pi={delta_c_mhz[peak_idx[1]]:.2f} MHz")
print(f"Edge Pout (Omega_RF=0, Delta_c=-20MHz) = {edge_val/1e-6:.4f} microW")

# ------------------------------------------------------------
# SAVE DATA
# ------------------------------------------------------------

np.savez(
    OUTPUT_DIR / "fig3_v5_qutip_response.npz",
    delta_c_mhz=delta_c_mhz, omega_rf_mhz=omega_rf_mhz,
    Pout_surface=Pout_surface, Pin=Pin, C0=C0, kp=kp, L=L_cell,
    N0=N0, sweep_seconds=dt,
)

import csv
with open(OUTPUT_DIR / "fig3_v5_data.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["delta_c_MHz", "omega_RF_MHz", "Pout_microW"])
    for i, orf in enumerate(omega_rf_mhz):
        for j, dc in enumerate(delta_c_mhz):
            writer.writerow([f"{dc:.4f}", f"{orf:.4f}", f"{Pout_surface[i,j]*1e6:.6f}"])

# ------------------------------------------------------------
# PLOT -- paper style: no title, Pout axis forced to the left
# (matches v1_paper/fig3_fresh_build/fig3_pout_surface_paperstyle.py's
# camera trick: view_init + zaxis._axinfo['juggled'] override)
# ------------------------------------------------------------

X, Y = np.meshgrid(delta_c_mhz, omega_rf_mhz)
Z = Pout_surface / 1e-5

fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")
surf = ax.plot_surface(X, Y, Z, cmap="turbo", linewidth=0, antialiased=True, alpha=0.93, rcount=97, ccount=201)

for orf_target, style in [(1.0, "--"), (4.0, ":"), (8.0, "-."), (12.0, "-")]:
    i = np.argmin(np.abs(omega_rf_mhz - orf_target))
    ax.plot(delta_c_mhz, np.full_like(delta_c_mhz, omega_rf_mhz[i]), Z[i, :],
            color="blue", linestyle=style, linewidth=1.6)

dc0 = np.argmin(np.abs(delta_c_mhz))
ax.plot(np.full_like(omega_rf_mhz, delta_c_mhz[dc0]), omega_rf_mhz, Z[:, dc0],
        color="red", linestyle="--", linewidth=2, marker="x", markersize=5,
        label=r"$P_{out}$ vs $\Omega_{RF}$ ($\Delta_c=0$)")

ax.set_xlabel(r"Coupling detuning, $\Delta_c/2\pi$ (MHz)")
ax.set_ylabel(r"RF Rabi frequency, $\Omega_{RF}/2\pi$ (MHz)")
ax.set_zlabel("")  # set_zlabel gets clipped with the juggled axis pane -- use fig.text instead
ax.set_xticks(np.arange(-20, 21, 10))
ax.set_yticks(np.arange(0, 13, 4))
ax.set_xlim(-20, 20)
ax.set_ylim(0, 12)
ax.set_zticks(np.arange(0, 0.81, 0.2))
ax.view_init(elev=23, azim=-230)
ax.zaxis._axinfo["juggled"] = (1, 2, 0)   # force the z (Pout) axis pane onto the left side
ax.invert_xaxis()   # match paper's Delta_c axis direction: negative on left, positive on right
fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1, label=r"$P_{out}$ (x$10^{-5}$ W)")
ax.legend(fontsize=9, loc="upper left")
fig.subplots_adjust(left=0.15)
fig.text(0.04, 0.55, r"$P_{out}$ (x$10^{-5}$ W)", rotation=90, va="center", fontsize=11)
fig.savefig(OUTPUT_DIR / "fig3_v5_pout_surface.png", dpi=220, bbox_inches="tight")
plt.close(fig)

print("\nSaved: fig3_v5_qutip_response.npz, fig3_v5_data.csv, fig3_v5_pout_surface.png")
print("DONE.")
