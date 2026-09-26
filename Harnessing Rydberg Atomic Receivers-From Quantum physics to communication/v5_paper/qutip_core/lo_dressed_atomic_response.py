# ============================================================
# CORE (v5 paper) -- REUSABLE LO-DRESSED atomic response, REAL QuTiP.
#
# Same physics as the frozen fig6/fig6_v5_qutip.py panel (a) (identical
# 4-level Hamiltonian, but the coupling on |3>-|4> is Omega_total -- the
# LO-dressed phasor sum Omega_LO + Omega_RF*exp(j*theta), whose magnitude
# is what this sweep scans directly, with all detunings held at zero).
# Extracted here as a standalone, well-documented source so later figures
# (Fig.7/8/9/10) can load Pout(Omega_total) directly instead of re-running
# a QuTiP sweep or reaching into fig6's own folder. Fig.6 itself is DONE
# and untouched -- this is a parallel, not a replacement, and uses the
# identical grid so the two are numerically interchangeable.
# ============================================================

import csv
import time
from pathlib import Path

import numpy as np
import qutip as qt

OUTPUT_DIR = Path(__file__).resolve().parent

# ------------------------------------------------------------
# PHYSICAL CONSTANTS
# ------------------------------------------------------------

e_charge = 1.6e-19
a0 = 5.2e-11
hbar = 1.054571817e-34
eps0 = 8.854e-12

# ------------------------------------------------------------
# ATOMIC / LASER PARAMETERS (paper Sec. V-A, v5 values -- same as
# fig3_v5_qutip.py / fig6_v5_qutip.py, kept identical for consistency)
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

# Analytic four-level Gamma (Eq.78, Appendix C) -- same derivation as
# fig6_v5_qutip.py, kept here since it's a property of this LO-dressed
# response curve (not of any one figure).
gamma2_mhz = 5.2
Omega_p_mhz = 8.0
Omega_c_mhz = 1.0
Gamma3_HWHM_mhz = (Omega_c_mhz**2 + Omega_p_mhz**2) / (2 * np.sqrt(gamma2_mhz**2 + 2 * Omega_p_mhz**2))
Gamma4_mhz = Omega_p_mhz * np.sqrt(2 * (Omega_c_mhz**2 + Omega_p_mhz**2) / (2 * Omega_p_mhz**2 + gamma2_mhz**2))
Omega_LO_opt_analytic_mhz = (np.sqrt(3) / 3) * Gamma4_mhz

print("=" * 70)
print("CORE (v5) -- LO-DRESSED atomic response, REAL QuTiP")
print("=" * 70)
print(f"qutip version = {qt.__version__}")
print(f"Gamma (four-level, Eq.78) = {Gamma4_mhz:.4f} MHz")
print(f"Omega_LO,opt (analytic) = {Omega_LO_opt_analytic_mhz:.4f} MHz (paper states 4.23 MHz)")
print("=" * 70)


def basis4():
    return [qt.basis(4, i) for i in range(4)]


def hamiltonian(Omega_total, Delta_p=0.0, Delta_c=0.0, Delta_RF=0.0):
    """LO-dressed 4-level Hamiltonian: Omega_total = |Omega_LO + Omega_RF*
    exp(j*theta)| is the effective coupling on |3>-|4>, all detunings
    zero by default (matches fig6 panel (a) exactly)."""
    k1, k2, k3, k4 = basis4()
    H = qt.qzero(4)
    H += -Delta_p * k2 * k2.dag()
    H += -(Delta_p + Delta_c) * k3 * k3.dag()
    H += -(Delta_p + Delta_c + Delta_RF) * k4 * k4.dag()
    H += (Omega_p / 2.0) * (k1 * k2.dag() + k2 * k1.dag())
    H += (Omega_c / 2.0) * (k2 * k3.dag() + k3 * k2.dag())
    H += (Omega_total / 2.0) * (k3 * k4.dag() + k4 * k3.dag())
    return H


def collapse_ops():
    k1, k2, k3, k4 = basis4()
    return [
        np.sqrt(gamma2) * k1 * k2.dag(),
        np.sqrt(gamma3) * k2 * k3.dag(),
        np.sqrt(gamma4) * k3 * k4.dag(),
    ]


C_OPS = collapse_ops()


def pout_at(Omega_total, Delta_p=0.0, Delta_c=0.0, Delta_RF=0.0):
    """Real qutip.steadystate() solve -> Pout (Watts) via Beer-Lambert."""
    H = hamiltonian(Omega_total, Delta_p, Delta_c, Delta_RF)
    rho_ss = qt.steadystate(H, C_OPS, method="direct")
    rho21 = complex(rho_ss[1, 0])
    chi = C0 * rho21
    return Pin * np.exp(-kp * L_cell * np.imag(chi))


if __name__ == "__main__":
    # Same grid as fig6_v5_qutip.py panel (a) so this is a drop-in,
    # cross-checkable canonical source rather than a second, possibly-
    # diverging one.
    omega_total_mhz = np.linspace(0.01, 10.0, 201)
    omega_total_vals = 2.0 * np.pi * omega_total_mhz * 1e6

    Pout_vals = np.zeros(len(omega_total_mhz))
    t0 = time.time()
    for i, Ot in enumerate(omega_total_vals):
        Pout_vals[i] = pout_at(Ot)
    dt = time.time() - t0
    print(f"\nSweep completed in {dt:.1f}s ({len(omega_total_mhz)} REAL qutip solves, "
          f"{dt/len(omega_total_mhz)*1000:.2f} ms/solve avg)")

    dPout = np.gradient(Pout_vals, omega_total_mhz)
    edge_mask = omega_total_mhz > 0.3
    idx_opt_local = np.argmax(np.abs(dPout[edge_mask]))
    Omega_LO_opt_numeric_mhz = omega_total_mhz[edge_mask][idx_opt_local]
    print(f"Omega_LO,opt (NUMERICAL, real QuTiP) = {Omega_LO_opt_numeric_mhz:.4f} MHz (paper states 4.67 MHz)")

    # Sanity cross-check against the frozen Fig.6 data, if present.
    fig6_npz = OUTPUT_DIR.parent / "fig6" / "fig6_v5_qutip_ldr_response.npz"
    if fig6_npz.exists():
        ref = np.load(fig6_npz)
        max_diff = np.max(np.abs(ref["Pout_vals"] - Pout_vals))
        print(f"Cross-check vs fig6 data: max|diff| = {max_diff:.3e} W "
              f"({'OK' if max_diff < 1e-12 else 'MISMATCH -- investigate'})")

    np.savez(
        OUTPUT_DIR / "lo_dressed_atomic_response.npz",
        omega_total_mhz=omega_total_mhz, Pout_vals=Pout_vals, dPout=dPout,
        Gamma4_mhz=Gamma4_mhz, Omega_LO_opt_analytic_mhz=Omega_LO_opt_analytic_mhz,
        Omega_LO_opt_numeric_mhz=Omega_LO_opt_numeric_mhz,
        Pin=Pin, C0=C0, kp=kp, L=L_cell, N0=N0, sweep_seconds=dt,
    )

    with open(OUTPUT_DIR / "lo_dressed_atomic_response.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["omega_total_MHz", "Pout_microW"])
        for i, ot in enumerate(omega_total_mhz):
            writer.writerow([f"{ot:.4f}", f"{Pout_vals[i]*1e6:.6f}"])

    print("\nSaved: lo_dressed_atomic_response.npz, lo_dressed_atomic_response.csv")
    print("DONE.")
