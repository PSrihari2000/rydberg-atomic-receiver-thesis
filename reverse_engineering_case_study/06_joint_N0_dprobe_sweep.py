# ============================================================
# *** CASE STUDY -- NOT a validated result ***
#
# TEST: joint 2D sweep over N0 and d_probe together (both real,
# continuous, independently-defensible knobs -- unlike a single knob,
# maybe a COMBINATION reaches both targets where neither did alone).
# fRF fixed at the transition-consistent 6.9458GHz (the one genuinely
# verified correction found this project). Uses the SAME exact
# algebraic rescaling tricks proven earlier (both N0 and d_probe act as
# provable closed-form rescalings of the already-computed real data --
# no new QuTiP solves needed for the sweep itself).
#
# Does not touch any frozen fresh_build file or n0_density_sensitivity/.
# ============================================================

from pathlib import Path
import csv

import numpy as np
from scipy.optimize import brentq

OUTPUT_DIR = Path(__file__).resolve().parent
FIG5_CSV = OUTPUT_DIR.parent / "fig5_fresh_build" / "fig5_data.csv"

print("=" * 70)
print("*** CASE STUDY 06 -- joint N0 x d_probe sweep ***")
print("=" * 70)

e_charge, a0, hbar, eps0, eta0 = 1.6e-19, 5.2e-11, 1.054571817e-34, 8.854e-12, 377.0
c_light = 2.998e8
Omega_p = 2.0*np.pi*8.0e6
wp_RF = -1443.459*e_charge*a0
wp_12 = (2.5*e_charge*a0)**2
lambda_p = 852e-9
wp_RF_over_hbar = wp_RF/hbar
fp = c_light/lambda_p
GLNA_lin, RL, D_resp, T_temp, eta_eff, B_bw = 100.0, 50.0, 0.55, 290.0, 0.8, 1.0e6
kB = 1.380649e-23
GTx_lin = 10**(2.15/10)
GRx_lin = GTx_lin
sigma_BGN_sq = 1e-3*10**(-90/10)
sigma_TN_sq = 4*kB*T_temp*B_bw
FRF_FIX = 6.9458e9
lambda_RF = c_light/FRF_FIX

D_PROBE_BASELINE = 0.76e-3
N0_BASELINE = 1.0e15

with open(FIG5_CSV, newline="") as f:
    rows = list(csv.reader(f))
meta = {}
i = 0
while rows[i]:
    if rows[i][0] != "# METADATA":
        meta[rows[i][0]] = float(rows[i][1])
    i += 1
i += 3
fig5a_omega_mhz, fig5a_pout_uW = [], []
while i < len(rows) and rows[i]:
    fig5a_omega_mhz.append(float(rows[i][0]))
    fig5a_pout_uW.append(float(rows[i][1]))
    i += 1
fig5a_omega_mhz = np.array(fig5a_omega_mhz)
fig5a_pout_W_baseline = np.array(fig5a_pout_uW)*1e-6
Omega_LO_mhz = meta["Omega_LO_operating_MHz"]

def linear_fit_score(x, y):
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope*x+intercept
    ss_res = np.sum((y-y_pred)**2); ss_tot = np.sum((y-np.mean(y))**2)
    return slope, intercept, (1.0-ss_res/ss_tot if ss_tot>0 else 1.0)

def find_linear_range(x_vals, y_vals, x_anchor, r2_threshold=0.998):
    center = int(np.argmin(np.abs(x_vals-x_anchor)))
    best=None
    max_radius = min(center, len(x_vals)-center-1)
    for radius in range(3, max_radius+1):
        left,right = center-radius, center+radius+1
        slope,intercept,r2 = linear_fit_score(x_vals[left:right], y_vals[left:right])
        if r2>=r2_threshold:
            best=dict(slope=slope, x_low=x_vals[left], x_high=x_vals[right-1], r2=r2)
        else:
            break
    return best

def gap_and_coverage(N0_new, d_probe_new):
    # d_probe rescaling: Pin_new/Pin_old = (d_probe_new/d_probe_old)^2, a PURE prefactor
    # (proven throughout this project) -- so Pout scales by this same factor BEFORE the
    # N0 exponential rescaling is applied (Pin itself changes, C0's Pin-independence unaffected).
    pin_ratio = (d_probe_new/D_PROBE_BASELINE)**2
    pout_after_diameter = fig5a_pout_W_baseline * pin_ratio
    Pin_new = pin_ratio * (np.pi/(2.0*eta0))*(D_PROBE_BASELINE*Omega_p*hbar/(2.0*np.sqrt(wp_12)))**2
    # N0 rescaling (exact identity, re-derived relative to the diameter-adjusted Pin)
    pout_final = Pin_new * (pout_after_diameter/Pin_new) ** (N0_new/N0_BASELINE)

    fit = find_linear_range(fig5a_omega_mhz, pout_final, Omega_LO_mhz)
    if fit is None:
        return None
    kappa_true = fit["slope"]/(2*np.pi*1e6)
    P0_bar = float(np.interp(Omega_LO_mhz, fig5a_omega_mhz, pout_final))
    Ibar = eta_eff*P0_bar*e_charge/(hbar*fp)
    sigma_PSN_sq = 2*e_charge*B_bw*Ibar
    A_LO = GLNA_lin*RL*D_resp**2*kappa_true**2*wp_RF_over_hbar**2
    sigma_Ry_LO_sq = A_LO*sigma_BGN_sq + GLNA_lin*RL*D_resp**2*sigma_PSN_sq + sigma_TN_sq

    def snr_conv_dB(d, PTx_W=1e-4):
        Pc = (PTx_W*GTx_lin/(4*np.pi*d**2)) * (lambda_RF**2/(4*np.pi))
        return 10*np.log10(Pc/(GRx_lin*GLNA_lin*sigma_BGN_sq + sigma_TN_sq))
    def snr_theo_dB(d, PTx_W=1e-4):
        PRx = PTx_W*GTx_lin*eta0/(4*np.pi*d**2)
        return 10*np.log10(A_LO*PRx/sigma_Ry_LO_sq)

    gap_100m = snr_theo_dB(100.0) - snr_conv_dB(100.0)
    try:
        conv_cross = brentq(lambda d: snr_conv_dB(d), 1e-3, 1e15)
        theo_cross = brentq(lambda d: snr_theo_dB(d), 1e-3, 1e15)
        coverage = theo_cross - conv_cross
    except ValueError:
        coverage = None
    return dict(gap_100m=gap_100m, coverage=coverage, kappa=kappa_true)

# ------------------------------------------------------------
# Verify d_probe rescaling piece against a fresh QuTiP spot-check
# ------------------------------------------------------------
import qutip as qt
def basis4(): return [qt.basis(4,i) for i in range(4)]
def hamiltonian_lodressed(Omega_total_val, Omega_p_, Omega_c_):
    k1,k2,k3,k4 = basis4()
    H = qt.qzero(4)
    H += (Omega_p_/2.0)*(k1*k2.dag()+k2*k1.dag())
    H += (Omega_c_/2.0)*(k2*k3.dag()+k3*k2.dag())
    H += (Omega_total_val/2.0)*(k3*k4.dag()+k4*k3.dag())
    return H
gamma2 = 2.0*np.pi*5.2e6
def collapse_ops_lodressed():
    k1,k2,k3,k4 = basis4()
    return [np.sqrt(gamma2)*k1*k2.dag()]
N0_fixed_check = 1e15
C0_check = -2.0*N0_fixed_check*wp_12/(eps0*hbar*Omega_p)
kp_wave = 2.0*np.pi/lambda_p
L_cell = 1.0e-2
def pout_direct(Omega_total_val, d_probe_val):
    Pin_val = (np.pi/(2.0*eta0))*(d_probe_val*Omega_p*hbar/(2.0*np.sqrt(wp_12)))**2
    H = hamiltonian_lodressed(Omega_total_val, Omega_p, 2.0*np.pi*1.0e6)
    rho_ss = qt.steadystate(H, collapse_ops_lodressed(), method="direct")
    rho21 = complex(rho_ss[1,0])
    chi = C0_check*rho21
    return Pin_val*np.exp(-kp_wave*L_cell*np.imag(chi))

print("\nVerifying joint diameter+N0 rescaling against a fresh QuTiP spot-check (d_probe=0.5mm)...")
Pout_direct_check = pout_direct(2*np.pi*4.0e6, 0.5e-3)
idx_check = np.argmin(np.abs(fig5a_omega_mhz-4.0))
pin_ratio_check = (0.5e-3/D_PROBE_BASELINE)**2
Pin_new_check = pin_ratio_check * (np.pi/(2.0*eta0))*(D_PROBE_BASELINE*Omega_p*hbar/(2.0*np.sqrt(wp_12)))**2
pout_after_d_check = fig5a_pout_W_baseline[idx_check] * pin_ratio_check
pout_predicted_check = Pin_new_check * (pout_after_d_check/Pin_new_check) ** (1.0)  # N0 same, exponent=1
print(f"  Direct fresh QuTiP: {Pout_direct_check/1e-6:.6f}uW, predicted (rescaled from real data): "
      f"{pout_predicted_check/1e-6:.6f}uW, reldiff={abs(Pout_direct_check-pout_predicted_check)/Pout_direct_check:.2e}")

# ------------------------------------------------------------
# 2D SWEEP
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("2D SWEEP: N0 x d_probe, target gap=44dB, target coverage=1500m")
print("=" * 70)

N0_grid = np.logspace(15.0, 16.3, 14)
d_probe_grid = np.array([0.36, 0.5, 0.76, 1.0, 1.5, 2.0, 3.0])*1e-3

best_combo = None
best_dist = np.inf
print(f"\n{'N0':>12} {'d_probe(mm)':>12} {'gap(dB)':>10} {'coverage(m)':>14} {'note':>10}")
for N0 in N0_grid:
    for dp in d_probe_grid:
        r = gap_and_coverage(N0, dp)
        if r is None or r["coverage"] is None:
            print(f"{N0:12.3e} {dp*1e3:12.2f} {'--':>10} {'--':>14} no linear range")
            continue
        dist_to_target = np.sqrt(((r["gap_100m"]-44.0)/44.0)**2 + ((r["coverage"]-1500.0)/1500.0)**2)
        note = ""
        if dist_to_target < best_dist:
            best_dist = dist_to_target
            best_combo = (N0, dp, r)
            note = "<- best so far"
        print(f"{N0:12.3e} {dp*1e3:12.2f} {r['gap_100m']:10.4f} {r['coverage']:14.4f} {note}")

print(f"\nBEST COMBINATION FOUND (minimizing joint relative distance to gap=44dB AND coverage=1500m):")
if best_combo:
    N0_b, dp_b, r_b = best_combo
    print(f"  N0={N0_b:.4e} ({N0_b/N0_BASELINE:.3f}x baseline), d_probe={dp_b*1e3:.3f}mm "
          f"({dp_b/D_PROBE_BASELINE:.3f}x baseline)")
    print(f"  gap={r_b['gap_100m']:.4f}dB (target 44dB), coverage={r_b['coverage']:.4f}m (target 1500m)")
    print(f"  Still {'FAR from' if best_dist > 0.3 else 'reasonably close to'} both targets simultaneously "
          f"(joint relative distance = {best_dist:.4f})")
else:
    print("  No valid combination found across this grid.")

# ------------------------------------------------------------
# STEP 2: finer targeted grid near the promising region
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 2: finer grid search near the promising region")
print("=" * 70)

N0_fine = np.linspace(1.0e15, 3.0e15, 11)
dp_fine = np.linspace(1.2e-3, 3.0e-3, 10)
results = []
for N0 in N0_fine:
    for dp in dp_fine:
        r = gap_and_coverage(N0, dp)
        if r is None or r["coverage"] is None:
            continue
        results.append((N0, dp, r["gap_100m"], r["coverage"]))

results.sort(key=lambda x: abs(x[3]-1500))
print("\nClosest to coverage=1500m:")
for N0, dp, gap, cov in results[:6]:
    print(f"  N0={N0:.3e}, d_probe={dp*1e3:.3f}mm: gap={gap:.4f}dB, coverage={cov:.4f}m")

results.sort(key=lambda x: abs(x[2]-44))
print("\nClosest to gap=44dB:")
for N0, dp, gap, cov in results[:6]:
    print(f"  N0={N0:.3e}, d_probe={dp*1e3:.3f}mm: gap={gap:.4f}dB, coverage={cov:.4f}m")

results.sort(key=lambda x: np.sqrt(((x[2]-44)/44)**2+((x[3]-1500)/1500)**2))
print("\nBest JOINT match (minimizing combined relative distance to both targets):")
for N0, dp, gap, cov in results[:6]:
    print(f"  N0={N0:.3e}, d_probe={dp*1e3:.3f}mm: gap={gap:.4f}dB, coverage={cov:.4f}m")

print("\nDONE.")
