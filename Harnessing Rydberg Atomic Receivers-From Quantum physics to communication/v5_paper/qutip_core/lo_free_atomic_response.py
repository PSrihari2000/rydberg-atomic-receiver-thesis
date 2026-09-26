# ============================================================
# CORE (v5 paper) -- REUSABLE LO-FREE atomic response, REAL QuTiP.
#
# Same physics as the frozen fig3/fig3_v5_qutip.py (4-level Hamiltonian,
# Delta_p=Delta_RF=0, Omega_RF is the direct RF coupling on the |3>-|4>
# transition), extracted here as a standalone, well-documented source so
# later figures (Fig.7/8/9/10) can load Pout(Delta_c, Omega_RF) directly
# instead of re-running a QuTiP sweep or reaching into fig3's own folder.
# Fig.3 itself is FROZEN and untouched -- this is a parallel, not a
# replacement, and uses the identical grid so the two are numerically
# interchangeable (cross-checked below at Delta_c=0).
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

print("=" * 70)
print("CORE (v5) -- LO-FREE atomic response, REAL QuTiP")
print("=" * 70)
print(f"qutip version = {qt.__version__}")
print(f"N0    = {N0:.4e} m^-3")
print(f"Pin   = {Pin*1e6:.4f} microW (paper-stated, direct)")
print("=" * 70)


def basis4():
    return [qt.basis(4, i) for i in range(4)]


def hamiltonian(Omega_RF, Delta_p=0.0, Delta_c=0.0, Delta_RF=0.0):
    """LO-free 4-level Hamiltonian: Omega_RF drives |3>-|4> directly
    (no LO tone superposed), identical structure to fig3/fig6."""
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
    """Real qutip.steadystate() solve -> Pout (Watts) via Beer-Lambert."""
    H = hamiltonian(Omega_RF, Delta_p, Delta_c, Delta_RF)
    rho_ss = qt.steadystate(H, C_OPS, method="direct")
    rho21 = complex(rho_ss[1, 0])
    chi = C0 * rho21
    return Pin * np.exp(-kp * L_cell * np.imag(chi))


if __name__ == "__main__":
    # Same grid as fig3_v5_qutip.py so this is a drop-in, cross-checkable
    # canonical source rather than a second, possibly-diverging one.
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

    # Sanity cross-check against the frozen Fig.3 data, if present: this
    # script's grid is identical, so the two surfaces should agree to
    # machine precision (same equations, same steady-state solves).
    fig3_npz = OUTPUT_DIR.parent / "fig3" / "fig3_v5_qutip_response.npz"
    if fig3_npz.exists():
        ref = np.load(fig3_npz)
        max_diff = np.max(np.abs(ref["Pout_surface"] - Pout_surface))
        print(f"Cross-check vs frozen fig3 data: max|diff| = {max_diff:.3e} W "
              f"({'OK' if max_diff < 1e-12 else 'MISMATCH -- investigate'})")

    np.savez(
        OUTPUT_DIR / "lo_free_atomic_response.npz",
        delta_c_mhz=delta_c_mhz, omega_rf_mhz=omega_rf_mhz,
        Pout_surface=Pout_surface, Pin=Pin, C0=C0, kp=kp, L=L_cell, N0=N0,
        sweep_seconds=dt,
    )

    with open(OUTPUT_DIR / "lo_free_atomic_response.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["delta_c_MHz", "omega_RF_MHz", "Pout_microW"])
        for i, orf in enumerate(omega_rf_mhz):
            for j, dc in enumerate(delta_c_mhz):
                writer.writerow([f"{dc:.4f}", f"{orf:.4f}", f"{Pout_surface[i, j]*1e6:.6f}"])

    print("\nSaved: lo_free_atomic_response.npz, lo_free_atomic_response.csv")
    print("DONE.")
