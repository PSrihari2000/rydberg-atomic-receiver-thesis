# ============================================================
# FIG. 3 -- d=0.36mm INDEPENDENT REBUILD
# Probe laser transmission Pout(Delta_c, Omega_RF)
#
# Identical physics/parameters/grid to fig3_fresh_build/fig3_hamiltonian_qutip.py
# (Sec.IV, Eqs.2-7), ONLY d_probe changed from 0.76mm to 0.36mm. Every
# grid point is a fresh QuTiP steady-state solve -- no cached/reused data,
# no fig3_fresh_build files touched or imported.
# ============================================================

import time
from pathlib import Path
import csv

import numpy as np
import qutip as qt
import matplotlib.pyplot as plt

OUTPUT_DIR = Path(__file__).resolve().parent

e_charge = 1.6e-19
a0 = 5.2e-11
hbar = 1.054571817e-34
eps0 = 8.854e-12
eta0 = 377.0

L = 1.0e-2
N0 = 1.0e15

gamma1 = 0.0
gamma2 = 2.0*np.pi*5.2e6
gamma3 = 2.0*np.pi*3.9e3
gamma4 = 2.0*np.pi*1.7e3

Omega_p = 2.0*np.pi*8.0e6
Omega_c = 2.0*np.pi*1.0e6

wp_RF = -1443.459*e_charge*a0
wp_12 = (2.5*e_charge*a0)**2

lambda_p = 852e-9
kp = 2.0*np.pi/lambda_p

d_probe = 0.36e-3   # ONLY parameter changed from fig3_fresh_build

Pin = (np.pi/(2.0*eta0)) * (d_probe*Omega_p*hbar/(2.0*np.sqrt(wp_12)))**2
C0 = -2.0*N0*wp_12/(eps0*hbar*Omega_p)

print("=" * 70)
print("FIG. 3 -- d=0.36mm INDEPENDENT REBUILD")
print("=" * 70)
print(f"d_probe = {d_probe*1e3:.4f} mm")
print(f"Pin  = {Pin:.6e} W = {Pin/1e-6:.4f} microW")
print(f"C0   = {C0:.6e}")

def basis4(): return [qt.basis(4,i) for i in range(4)]

def hamiltonian(Omega_RF, Delta_p=0.0, Delta_c=0.0, Delta_RF=0.0):
    k1,k2,k3,k4 = basis4()
    H = qt.qzero(4)
    H += -Delta_p*k2*k2.dag()
    H += -(Delta_p+Delta_c)*k3*k3.dag()
    H += -(Delta_p+Delta_c+Delta_RF)*k4*k4.dag()
    H += (Omega_p/2.0)*(k1*k2.dag()+k2*k1.dag())
    H += (Omega_c/2.0)*(k2*k3.dag()+k3*k2.dag())
    H += (Omega_RF/2.0)*(k3*k4.dag()+k4*k3.dag())
    return H

def collapse_ops():
    k1,k2,k3,k4 = basis4()
    return [np.sqrt(gamma2)*k1*k2.dag(), np.sqrt(gamma3)*k2*k3.dag(), np.sqrt(gamma4)*k3*k4.dag()]

def pout_at(Omega_RF, Delta_c, Delta_p=0.0, Delta_RF=0.0):
    H = hamiltonian(Omega_RF, Delta_p, Delta_c, Delta_RF)
    rho_ss = qt.steadystate(H, collapse_ops(), method="direct")
    rho21 = complex(rho_ss[1,0])
    chi = C0*rho21
    return Pin*np.exp(-kp*L*np.imag(chi))

H_test = hamiltonian(2.0*np.pi*6e6, Delta_c=0.0)
rho_test = qt.steadystate(H_test, collapse_ops(), method="direct")
print(f"\nSanity check @ Omega_RF/2pi=6MHz, Delta_c=0: Tr(rho)={rho_test.tr()}, "
      f"hermiticity_err={(rho_test-rho_test.dag()).norm():.3e}")
Pout_test = pout_at(2.0*np.pi*6e6, 0.0)
print(f"  Pout = {Pout_test/1e-6:.6f} microW")

delta_c_mhz = np.linspace(-20.0, 20.0, 201)
omega_rf_mhz = np.linspace(0.0, 20.0, 161)
delta_c_vals = 2.0*np.pi*delta_c_mhz*1e6
omega_rf_vals = 2.0*np.pi*omega_rf_mhz*1e6
n_total = len(delta_c_mhz)*len(omega_rf_mhz)
print(f"\nGrid: {len(omega_rf_mhz)} x {len(delta_c_mhz)} = {n_total} fresh QuTiP steady-state solves")

Pout_surface = np.zeros((len(omega_rf_mhz), len(delta_c_mhz)))
t0 = time.time()
for i, Omega_RF in enumerate(omega_rf_vals):
    for j, Delta_c in enumerate(delta_c_vals):
        Pout_surface[i, j] = pout_at(Omega_RF, Delta_c)
    if (i+1) % 20 == 0 or i == len(omega_rf_vals)-1:
        elapsed = time.time()-t0
        rate = (i+1)/elapsed
        eta = (len(omega_rf_vals)-i-1)/rate if rate>0 else float("nan")
        print(f"Row {i+1:4d}/{len(omega_rf_vals)}  elapsed={elapsed:6.1f}s  ETA={eta:6.1f}s")
dt = time.time()-t0
print(f"\nSweep completed in {dt:.1f}s ({n_total} solves, {dt/n_total*1000:.2f} ms/solve avg)")

peak_val = Pout_surface.max()
peak_idx = np.unravel_index(np.argmax(Pout_surface), Pout_surface.shape)
print(f"Peak Pout = {peak_val/1e-6:.4f} microW at Omega_RF/2pi={omega_rf_mhz[peak_idx[0]]:.3f}MHz, "
      f"Delta_c/2pi={delta_c_mhz[peak_idx[1]]:.3f}MHz")

np.savez(OUTPUT_DIR / "freshbuild_fig3_d036mm_raw.npz",
          delta_c_mhz=delta_c_mhz, omega_rf_mhz=omega_rf_mhz, Pout_surface=Pout_surface,
          Pin=Pin, C0=C0, kp=kp, L=L, d_probe=d_probe)
print("Saved: freshbuild_fig3_d036mm_raw.npz (raw grid, for Fig.4-d036mm to reuse)")

csv_path = OUTPUT_DIR / "freshbuild_fig3_d036mm.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["# d_probe_mm", f"{d_probe*1e3:.4f}"])
    writer.writerow(["# Pin_uW", f"{Pin/1e-6:.6f}"])
    writer.writerow(["# Peak_Pout_uW", f"{peak_val/1e-6:.6f}"])
    writer.writerow(["# Peak_at_OmegaRF_MHz", f"{omega_rf_mhz[peak_idx[0]]:.4f}"])
    writer.writerow(["# Peak_at_DeltaC_MHz", f"{delta_c_mhz[peak_idx[1]]:.4f}"])
    writer.writerow([])
    writer.writerow(["Delta_c_MHz"] + [f"{w:.4f}" for w in omega_rf_mhz])
    for j in range(len(delta_c_mhz)):
        writer.writerow([f"{delta_c_mhz[j]:.4f}"] + [f"{Pout_surface[i,j]/1e-6:.6f}" for i in range(len(omega_rf_mhz))])
print(f"Saved: {csv_path.name}")

X, Y = np.meshgrid(delta_c_mhz, omega_rf_mhz)
Z = Pout_surface/1e-6
fig = plt.figure(figsize=(9,7))
ax = fig.add_subplot(111, projection="3d")
surf = ax.plot_surface(X, Y, Z, cmap="turbo", linewidth=0, antialiased=True, alpha=0.93)
dc0 = np.argmin(np.abs(delta_c_mhz))
ax.plot(np.full_like(omega_rf_mhz, delta_c_mhz[dc0]), omega_rf_mhz, Z[:, dc0],
        color="red", linestyle="--", linewidth=2, marker="x", markersize=5,
        label=r"$P_{out}$ vs $\Omega_{RF}$ ($\Delta_c=0$)")
ax.set_xlabel(r"Coupling detuning, $\Delta_c/2\pi$ (MHz)")
ax.set_ylabel(r"RF Rabi frequency, $\Omega_{RF}/2\pi$ (MHz)")
ax.set_zlabel(r"$P_{out}$ ($\times10^{-6}$ W)")
ax.set_title(r"Fig. 3. $P_{out}$ versus $\Delta_c$ and $\Omega_{RF}$" "\n(d = 0.36 mm, independent rebuild)")
ax.view_init(elev=28, azim=100)
fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1, label=r"$P_{out}$ ($\mu$W)")
ax.legend(fontsize=9, loc="upper left")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "freshbuild_fig3_d036mm.png", dpi=220, bbox_inches="tight")
plt.close(fig)
print("Saved: freshbuild_fig3_d036mm.png")

report = f"""# Fig. 3 math report -- d=0.36mm independent rebuild

Identical physics/parameters/grid to `fig3_fresh_build/fig3_hamiltonian_qutip.py`
(paper Sec.IV, Eqs.2-7) -- the ONLY change is d_probe = 0.36mm instead of the
paper-literal 0.76mm. Every grid point is a fresh QuTiP steady-state solve.
Does not touch fig3_fresh_build or the older qutip_d036mm files.

## Hamiltonian (rotating frame, rad/s)

    H = -Delta_p|2><2| - (Delta_p+Delta_c)|3><3| - (Delta_p+Delta_c+Delta_RF)|4><4|
        + (Omega_p/2)(|1><2|+|2><1|) + (Omega_c/2)(|2><3|+|3><2|) + (Omega_RF/2)(|3><4|+|4><3|)

Steady state via QuTiP `steadystate` (direct method), collapse operators
sqrt(gamma2)|1><2|, sqrt(gamma3)|2><3|, sqrt(gamma4)|3><4|. Delta_p=Delta_RF=0.

Pout = Pin * exp(-kp*L*Im(C0*rho21))   (Eq.2/4)

## Parameters (paper Sec.IV, only d_probe changed)

    L=1cm, N0=1e15 m^-3, d_probe=0.36mm (vs paper-literal 0.76mm)
    Omega_p/2pi=8.0MHz, Omega_c/2pi=1.0MHz
    gamma2/2pi=5.2MHz, gamma3/2pi=3.9kHz, gamma4/2pi=1.7kHz
    wp_RF=-1443.459 e*a0, wp_12=(2.5 e*a0)^2, lambda_p=852nm

## Grid

    Delta_c/2pi: [-20,20]MHz, 201 points
    Omega_RF/2pi: [0,20]MHz, 161 points
    Total: {n_total} fresh QuTiP steady-state solves, {dt:.1f}s ({dt/n_total*1000:.2f}ms/solve)

## Results (this run)

    Pin = {Pin:.6e} W = {Pin/1e-6:.4f} microW
    C0 = {C0:.6e}
    Sanity check (Omega_RF/2pi=6MHz, Delta_c=0): Pout={Pout_test/1e-6:.6f}microW,
        Tr(rho)={rho_test.tr()}, hermiticity_error={(rho_test-rho_test.dag()).norm():.3e}
    Peak Pout = {peak_val/1e-6:.4f} microW at Omega_RF/2pi={omega_rf_mhz[peak_idx[0]]:.3f}MHz,
        Delta_c/2pi={delta_c_mhz[peak_idx[1]]:.3f}MHz

## Comparison to the 0.76mm fresh_build

Pin scales as d_probe^2 (pure prefactor on Pout, proven algebraically and confirmed
numerically throughout this project -- see project memory). Expected ratio:
(0.36/0.76)^2 = {(0.36/0.76)**2:.4f}. Actual Pin ratio this run: to be compared against
fig3_fresh_build's own Pin once both are available side by side.
"""
with open(OUTPUT_DIR / "freshbuild_fig3_d036mm_math_report.md", "w") as f:
    f.write(report)
print("Saved: freshbuild_fig3_d036mm_math_report.md")
print("\nDONE.")
