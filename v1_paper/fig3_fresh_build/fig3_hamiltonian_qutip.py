# ============================================================
# FIG. 3 -- FRESH, SELF-CONTAINED REBUILD
# Probe laser transmission Pout(Delta_c, Omega_RF)
#
# Does NOT import fig3.py, fig4.py, or anything under
# fig7_reinvestigation/. Every parameter is re-typed from the
# paper text (Sec. IV, Eqs. 2-7) below. Every grid point is a
# fresh QuTiP steady-state solve -- no cached data reused.
#
# Confirmed decisions (see fig3_math_report.md):
#   d      = 0.76 mm   (paper-literal 1/e^2 diameter, Eq.3)
#   e, a0  = paper's own rounded values (1.6e-19 C, 5.2e-11 m)
#   N0     = +1e15 m^-3 (sign-corrected, see report Flag 2)
#   Delta_p = Delta_RF = 0 (fixed; assumption, see report)
#
# No fitting, tuning, or rescaling of any kind is performed.
# ============================================================

import time
from pathlib import Path

import numpy as np
import qutip as qt
import matplotlib.pyplot as plt

OUTPUT_DIR = Path(__file__).resolve().parent

# ------------------------------------------------------------
# PHYSICAL CONSTANTS (paper's own rounded values, Sec. IV)
# ------------------------------------------------------------

e_charge = 1.6e-19          # C
a0 = 5.2e-11                 # m
hbar = 1.054571817e-34       # J s   (not stated rounded by the paper; standard CODATA value)
eps0 = 8.854e-12             # F/m
eta0 = 377.0                 # Ohm

# ------------------------------------------------------------
# ATOMIC SYSTEM PARAMETERS (paper Sec. IV)
# ------------------------------------------------------------

L = 1.0e-2                   # vapor cell length, m
N0 = 1.0e15                  # atom density, m^-3 (sign-corrected, see math report)

gamma1 = 0.0
gamma2 = 2.0 * np.pi * 5.2e6
gamma3 = 2.0 * np.pi * 3.9e3
gamma4 = 2.0 * np.pi * 1.7e3

Omega_p = 2.0 * np.pi * 8.0e6
Omega_c = 2.0 * np.pi * 1.0e6

wp_RF = -1443.459 * e_charge * a0        # |3>-|4> dipole moment
wp_12 = (2.5 * e_charge * a0) ** 2       # |1>-|2> dipole moment, SQUARED (paper's own convention)

lambda_p = 852e-9
kp = 2.0 * np.pi / lambda_p

# ------------------------------------------------------------
# PROBE BEAM DIAMETER -- confirmed: paper-literal, 0.76mm
# ------------------------------------------------------------

d_probe = 0.76e-3   # m -- used DIRECTLY in Eq.(3), no radius substitution

Pin = (
    np.pi / (2.0 * eta0)
    * (d_probe * Omega_p * hbar / (2.0 * np.sqrt(wp_12))) ** 2
)

# ------------------------------------------------------------
# SUSCEPTIBILITY COEFFICIENT C0 (Eq. 4)
# ------------------------------------------------------------

C0 = -2.0 * N0 * wp_12 / (eps0 * hbar * Omega_p)

print("=" * 70)
print("FIG. 3 -- FRESH REBUILD -- PARAMETERS")
print("=" * 70)
print(f"d (probe beam diameter, paper-literal) = {d_probe*1e3:.4f} mm")
print(f"Pin  = {Pin:.6e} W = {Pin/1e-6:.4f} microW")
print(f"C0   = {C0:.6e}")
print(f"kp   = {kp:.6e} rad/m")
print("=" * 70)


# ------------------------------------------------------------
# HAMILTONIAN (Eq. 6) -- fresh, from scratch
# ------------------------------------------------------------

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


def pout_at(Omega_RF, Delta_c, Delta_p=0.0, Delta_RF=0.0):
    H = hamiltonian(Omega_RF, Delta_p, Delta_c, Delta_RF)
    c_ops = collapse_ops()
    rho_ss = qt.steadystate(H, c_ops, method="direct")
    rho21 = complex(rho_ss[1, 0])
    chi = C0 * rho21
    exponent = -kp * L * np.imag(chi)
    return Pin * np.exp(exponent)


# ------------------------------------------------------------
# SANITY CHECK -- single point, verify trace/hermiticity
# ------------------------------------------------------------

H_test = hamiltonian(2.0 * np.pi * 6e6, Delta_c=0.0)
rho_test = qt.steadystate(H_test, collapse_ops(), method="direct")
print(f"\nSanity check @ Omega_RF/2pi=6MHz, Delta_c=0:")
print(f"  Trace(rho) = {rho_test.tr()}")
print(f"  Hermiticity error = {(rho_test - rho_test.dag()).norm():.3e}")
Pout_test = pout_at(2.0 * np.pi * 6e6, 0.0)
print(f"  Pout = {Pout_test/1e-6:.6f} microW")

# ------------------------------------------------------------
# FULL GRID SWEEP (Delta_c x Omega_RF), fresh QuTiP solves
# ------------------------------------------------------------

delta_c_mhz = np.linspace(-20.0, 20.0, 201)
omega_rf_mhz = np.linspace(0.0, 20.0, 161)

delta_c_vals = 2.0 * np.pi * delta_c_mhz * 1e6
omega_rf_vals = 2.0 * np.pi * omega_rf_mhz * 1e6

n_total = len(delta_c_mhz) * len(omega_rf_mhz)
print(f"\nGrid: {len(omega_rf_mhz)} x {len(delta_c_mhz)} = {n_total} fresh QuTiP steady-state solves")

Pout_surface = np.zeros((len(omega_rf_mhz), len(delta_c_mhz)))

t0 = time.time()
for i, Omega_RF in enumerate(omega_rf_vals):
    for j, Delta_c in enumerate(delta_c_vals):
        Pout_surface[i, j] = pout_at(Omega_RF, Delta_c)
    elapsed = time.time() - t0
    rate = (i + 1) / elapsed
    eta = (len(omega_rf_vals) - i - 1) / rate if rate > 0 else float("nan")
    print(f"\rRow {i+1:4d}/{len(omega_rf_vals)}  Omega_RF/2pi={omega_rf_mhz[i]:7.3f} MHz  "
          f"elapsed={elapsed:6.1f}s  ETA={eta:6.1f}s", end="")
dt = time.time() - t0
print(f"\n\nSweep completed in {dt:.1f}s ({n_total} solves, {dt/n_total*1000:.2f} ms/solve avg)")

peak_val = Pout_surface.max()
peak_idx = np.unravel_index(np.argmax(Pout_surface), Pout_surface.shape)
print(f"Peak Pout = {peak_val/1e-6:.4f} microW at "
      f"Omega_RF/2pi={omega_rf_mhz[peak_idx[0]]:.3f} MHz, Delta_c/2pi={delta_c_mhz[peak_idx[1]]:.3f} MHz")

# ------------------------------------------------------------
# SAVE RAW DATA
# ------------------------------------------------------------

np.savez(
    OUTPUT_DIR / "fig3_quantum_response.npz",
    delta_c_mhz=delta_c_mhz, omega_rf_mhz=omega_rf_mhz,
    Pout_surface=Pout_surface, Pin=Pin, C0=C0, kp=kp, L=L,
    d_probe=d_probe, Omega_p=Omega_p, Omega_c=Omega_c,
    sweep_seconds=dt,
)
print(f"\nSaved: fig3_quantum_response.npz")

# ------------------------------------------------------------
# PLOT -- paper-style 3D surface, clean caption
# ------------------------------------------------------------

X, Y = np.meshgrid(delta_c_mhz, omega_rf_mhz)
Z = Pout_surface / 1e-6

fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")
surf = ax.plot_surface(X, Y, Z, cmap="turbo", linewidth=0, antialiased=True, alpha=0.93)

dc0 = np.argmin(np.abs(delta_c_mhz))
ax.plot(
    np.full_like(omega_rf_mhz, delta_c_mhz[dc0]), omega_rf_mhz, Z[:, dc0],
    color="red", linestyle="--", linewidth=2, marker="x", markersize=5,
    label=r"$P_{out}$ vs $\Omega_{RF}$ ($\Delta_c=0$)"
)

ax.set_xlabel(r"Coupling detuning, $\Delta_c/2\pi$ (MHz)")
ax.set_ylabel(r"RF Rabi frequency, $\Omega_{RF}/2\pi$ (MHz)")
ax.set_zlabel(r"$P_{out}$ ($\times10^{-6}$ W)")
ax.set_title(r"Fig. 3. $P_{out}$ versus $\Delta_c$ and $\Omega_{RF}$" "\n(d = 0.76 mm, fresh rebuild)")
ax.view_init(elev=28, azim=100)
fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1, label=r"$P_{out}$ ($\mu$W)")
ax.legend(fontsize=9, loc="upper left")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "fig3_pout_surface.png", dpi=220, bbox_inches="tight")
plt.close(fig)
print("Saved: fig3_pout_surface.png")

# ------------------------------------------------------------
# Append numeric results to the math report
# ------------------------------------------------------------

with open(OUTPUT_DIR / "fig3_math_report.md", "a") as f:
    f.write("\n\n## 7. Actual numeric results (this run)\n\n")
    f.write(f"- Pin = {Pin:.6e} W = {Pin/1e-6:.4f} microW\n")
    f.write(f"- C0 = {C0:.6e}\n")
    f.write(f"- Sanity check point (Omega_RF/2pi=6MHz, Delta_c=0): "
            f"Pout = {Pout_test/1e-6:.6f} microW, Tr(rho)={rho_test.tr()}, "
            f"hermiticity error={(rho_test - rho_test.dag()).norm():.3e}\n")
    f.write(f"- Grid: {len(omega_rf_mhz)} x {len(delta_c_mhz)} = {n_total} points\n")
    f.write(f"- Sweep time: {dt:.1f}s ({dt/n_total*1000:.2f} ms/solve average)\n")
    f.write(f"- Peak Pout = {peak_val/1e-6:.4f} microW at "
            f"Omega_RF/2pi={omega_rf_mhz[peak_idx[0]]:.3f} MHz, "
            f"Delta_c/2pi={delta_c_mhz[peak_idx[1]]:.3f} MHz\n")
    f.write(f"- Comparison to paper's published Fig.3 peak (~9.6-9.8 microW): "
            f"ratio = {peak_val/1e-6/9.7:.2f}x "
            f"(Flag 1 in Section 5 confirmed/refuted by this number)\n")

print("\nAppended numeric results to fig3_math_report.md")
print("\nDONE.")
