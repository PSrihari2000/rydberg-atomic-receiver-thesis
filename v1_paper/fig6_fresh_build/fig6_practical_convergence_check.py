# ============================================================
# EXPLORATORY -- convergence check for the Practical LO-dressed
# "hump" (rising above LO-free's ceiling around d~0.2-3m).
#
# NEW file, does not touch fig6_snr_vs_distance.py or any other
# existing fig6 file.
#
# Tests whether the hump is genuine physics or a numerical artifact by
# re-running the SAME real pipeline at HIGHER resolution in two
# independent places:
#   1. The wide QuTiP Omega_total sweep: 350 pts (as used) vs 700 pts
#      (identical Hamiltonian/parameters, just finer spacing) -- checks
#      whether the cubic-spline interpolation is ringing/overshooting
#      between under-sampled points.
#   2. The Fourier extraction: 512 samples/period (as used) vs 2048 --
#      checks whether the beat-frequency amplitude is well-converged.
# If the SNR values at the hump barely move under both refinements,
# the hump is real; if they shift substantially, it was an artifact.
# ============================================================

from pathlib import Path
import time

import numpy as np
import qutip as qt
from scipy.interpolate import CubicSpline

OUTPUT_DIR = Path(__file__).resolve().parent

print("=" * 70)
print("CONVERGENCE CHECK: Practical LO-dressed hump, d=0.5 to 3m, PTx=-10dBm")
print("=" * 70)

# ------------------------------------------------------------
# Identical parameters/physics to fig6_snr_vs_distance.py
# ------------------------------------------------------------

e_charge = 1.6e-19
a0 = 5.2e-11
hbar = 1.054571817e-34
eps0 = 8.854e-12
eta0 = 377.0
GTx_dBi = 2.15
GLNA_dB = 20.0
RL = 50.0
D_resp = 0.55
sigma_BGN_dBm = -90.0

Omega_p = 2.0*np.pi*8.0e6
Omega_c = 2.0*np.pi*1.0e6
gamma2 = 2.0*np.pi*5.2e6
wp_RF = -1443.459*e_charge*a0
wp_12 = (2.5*e_charge*a0)**2
lambda_p = 852e-9
kp_wave = 2.0*np.pi/lambda_p
N0 = 1.0e15
L_cell = 1.0e-2
Pin = (np.pi/(2.0*eta0))*(0.76e-3*Omega_p*hbar/(2.0*np.sqrt(wp_12)))**2
C0 = -2.0*N0*wp_12/(eps0*hbar*Omega_p)
Omega_LO_mhz = 4.23
Omega_LO = 2.0*np.pi*Omega_LO_mhz*1e6
Delta_f = 150e3
wp_RF_over_hbar = wp_RF/hbar

def db_to_lin(db): return 10.0**(db/10.0)
def dbm_to_w(dbm): return 1e-3*10.0**(dbm/10.0)

GTx_lin = db_to_lin(GTx_dBi)
GLNA_lin = db_to_lin(GLNA_dB)
sigma_BGN_sq = dbm_to_w(sigma_BGN_dBm)

def basis4(): return [qt.basis(4, i) for i in range(4)]

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

def run_wide_sweep(n_points, max_mhz=35.0):
    grid_mhz = np.linspace(0.05, max_mhz, n_points)
    pout_w = np.zeros_like(grid_mhz)
    t0=time.time()
    for i,w in enumerate(grid_mhz):
        pout_w[i] = pout_at_omega_total(2*np.pi*w*1e6)
    print(f"  {n_points}-pt sweep: {time.time()-t0:.1f}s")
    return grid_mhz, pout_w

print("\nBuilding sweep A (350 pts, as used in fig6_snr_vs_distance.py)...")
grid_350, pout_350 = run_wide_sweep(350)
cubic_350 = CubicSpline(grid_350, pout_350)

print("Building sweep B (700 pts, 2x density, SAME physics)...")
grid_700, pout_700 = run_wide_sweep(700)
cubic_700 = CubicSpline(grid_700, pout_700)

# ------------------------------------------------------------
# Recompute Practical LO-dressed SNR at hump-region distances,
# for both sweep resolutions AND both Fourier sample densities
# ------------------------------------------------------------

sigma_Ry_LO_sq = 7.556606e-14  # from the main script's real run (fixed noise floor, unaffected by this check)

def snr_practical(d, PTx_dBm, cubic, n_period):
    PTx_W = dbm_to_w(PTx_dBm)
    PRx = PTx_W*GTx_lin*eta0/(4*np.pi*d**2)
    ERF = np.sqrt(PRx)
    Omega_RF = ERF*abs(wp_RF_over_hbar)
    t_grid = np.linspace(0.0, 1.0/Delta_f, n_period, endpoint=False)
    psi = 2*np.pi*Delta_f*t_grid
    Omega_total = np.abs(Omega_LO + Omega_RF*np.exp(1j*psi))
    Omega_total_mhz = Omega_total/(2*np.pi)/1e6
    if Omega_total_mhz.max() > 35.0 or Omega_total_mhz.min() < 0.05:
        return np.nan
    Pout_t = cubic(Omega_total_mhz)
    P_df = (2.0/n_period)*np.abs(np.sum(Pout_t*np.exp(-1j*psi)))
    signal = GLNA_lin*RL*D_resp**2*P_df**2
    return 10*np.log10(signal/sigma_Ry_LO_sq)

TEST_DISTANCES = [0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
print(f"\n{'d(m)':>6} {'350pt/512samp':>14} {'700pt/512samp':>14} {'350pt/2048samp':>15} {'700pt/2048samp':>15}")
for d in TEST_DISTANCES:
    v1 = snr_practical(d, -10.0, cubic_350, 512)
    v2 = snr_practical(d, -10.0, cubic_700, 512)
    v3 = snr_practical(d, -10.0, cubic_350, 2048)
    v4 = snr_practical(d, -10.0, cubic_700, 2048)
    print(f"{d:6.2f} {v1:14.4f} {v2:14.4f} {v3:15.4f} {v4:15.4f}")

print("\nDONE.")
