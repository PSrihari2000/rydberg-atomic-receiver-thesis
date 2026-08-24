# ============================================================
# EXPLORATORY -- how much more of the short-distance region becomes
# coverable if the wide QuTiP sweep cap (35MHz in the main build) is
# pushed higher, and what does Practical LO-dressed do out there?
#
# NEW file, does not touch any other fig6 file. Same Hamiltonian as
# the main build's wide sweep, just a wider domain (0.05 to 80MHz).
# ============================================================

from pathlib import Path
import time

import numpy as np
import qutip as qt
from scipy.interpolate import CubicSpline

OUTPUT_DIR = Path(__file__).resolve().parent

print("=" * 70)
print("ANGLE 5: extending the wide QuTiP sweep beyond 35MHz")
print("=" * 70)

e_charge, a0, hbar, eps0, eta0 = 1.6e-19, 5.2e-11, 1.054571817e-34, 8.854e-12, 377.0
Omega_p = 2*np.pi*8.0e6
Omega_c = 2*np.pi*1.0e6
gamma2 = 2*np.pi*5.2e6
wp_RF = -1443.459*e_charge*a0
wp_12 = (2.5*e_charge*a0)**2
lambda_p = 852e-9
kp_wave = 2*np.pi/lambda_p
N0, L_cell = 1.0e15, 1.0e-2
d_probe = 0.76e-3
Pin = (np.pi/(2.0*eta0))*(d_probe*Omega_p*hbar/(2.0*np.sqrt(wp_12)))**2
C0 = -2.0*N0*wp_12/(eps0*hbar*Omega_p)
Omega_LO_mhz = 4.23
Omega_LO = 2*np.pi*Omega_LO_mhz*1e6
Delta_f = 150e3
wp_RF_over_hbar = wp_RF/hbar

def basis4(): return [qt.basis(4,i) for i in range(4)]
def hamiltonian_lodressed(Omega_total_val):
    k1,k2,k3,k4 = basis4()
    H = qt.qzero(4)
    H += (Omega_p/2.0)*(k1*k2.dag()+k2*k1.dag())
    H += (Omega_c/2.0)*(k2*k3.dag()+k3*k2.dag())
    H += (Omega_total_val/2.0)*(k3*k4.dag()+k4*k3.dag())
    return H
def collapse_ops_lodressed():
    k1,k2,k3,k4 = basis4()
    return [np.sqrt(gamma2)*k1*k2.dag()]
def pout_at_omega_total(Omega_total_val):
    H = hamiltonian_lodressed(Omega_total_val)
    rho_ss = qt.steadystate(H, collapse_ops_lodressed(), method="direct")
    rho21 = complex(rho_ss[1,0])
    chi = C0*rho21
    return Pin*np.exp(-kp_wave*L_cell*np.imag(chi))

NEW_MAX_MHZ = 80.0
N_POINTS = 550   # keep similar point density to the main build's 350pts/35MHz (=10/MHz)
grid_mhz = np.linspace(0.05, NEW_MAX_MHZ, N_POINTS)
pout_w = np.zeros_like(grid_mhz)
t0=time.time()
for i,w in enumerate(grid_mhz):
    pout_w[i] = pout_at_omega_total(2*np.pi*w*1e6)
print(f"{N_POINTS} QuTiP solves over [0.05,{NEW_MAX_MHZ}]MHz, {time.time()-t0:.1f}s")
cubic_wide = CubicSpline(grid_mhz, pout_w)

def db_to_lin(db): return 10**(db/10)
def dbm_to_w(dbm): return 1e-3*10**(dbm/10)
GTx_lin = db_to_lin(2.15)
GLNA_lin = db_to_lin(20.0)
RL, D_resp = 50.0, 0.55
sigma_Ry_LO_sq = 7.556606e-14  # from the main build's real run

N_PERIOD = 512
t_grid = np.linspace(0.0, 1.0/Delta_f, N_PERIOD, endpoint=False)
psi_grid = 2*np.pi*Delta_f*t_grid

def snr_practical(d, PTx_dBm, max_mhz):
    PRx = dbm_to_w(PTx_dBm)*GTx_lin*eta0/(4*np.pi*d**2)
    Omega_RF = np.sqrt(PRx)*abs(wp_RF_over_hbar)
    Omega_total = np.abs(Omega_LO + Omega_RF*np.exp(1j*psi_grid))
    Omega_total_mhz = Omega_total/(2*np.pi)/1e6
    if Omega_total_mhz.max() > max_mhz or Omega_total_mhz.min() < 0.05:
        return np.nan
    Pout_t = cubic_wide(Omega_total_mhz)
    P_df = (2.0/N_PERIOD)*np.abs(np.sum(Pout_t*np.exp(-1j*psi_grid)))
    signal = GLNA_lin*RL*D_resp**2*P_df**2
    return 10*np.log10(signal/sigma_Ry_LO_sq)

DISTANCE_M = np.logspace(np.log10(0.05), np.log10(2.0), 60)
print(f"\nPTx=+10dBm, comparing old cap (35MHz) vs new cap ({NEW_MAX_MHZ}MHz):")
print(f"{'d(m)':>8} {'SNR@35MHz_cap':>14} {'SNR@80MHz_cap':>14}")
old_covered, new_covered = 0, 0
for d in DISTANCE_M:
    v_old = snr_practical(d, 10.0, 35.0)
    v_new = snr_practical(d, 10.0, NEW_MAX_MHZ)
    if not np.isnan(v_old): old_covered += 1
    if not np.isnan(v_new): new_covered += 1
    marker = "  <- newly covered" if (np.isnan(v_old) and not np.isnan(v_new)) else ""
    print(f"{d:8.4f} {v_old if not np.isnan(v_old) else float('nan'):14.4f} "
          f"{v_new if not np.isnan(v_new) else float('nan'):14.4f}{marker}")

print(f"\nOld cap (35MHz): {old_covered}/{len(DISTANCE_M)} points covered in [0.05,2.0]m window")
print(f"New cap ({NEW_MAX_MHZ}MHz): {new_covered}/{len(DISTANCE_M)} points covered")
print("\nDONE.")
