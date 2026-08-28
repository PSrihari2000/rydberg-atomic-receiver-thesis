# ============================================================
# *** CASE STUDY -- NOT a validated result ***
#
# TEST: Omega_LO sensitivity. Paper states 4.23MHz (Sec.IV, the value
# used to generate all curves) but Appendix B's own closed-form Eq.58
# gives an "optimal" 2.9576MHz -- a known ~30% paper-internal
# discrepancy, never resolved. What does Fig.6's gap/coverage look like
# across the whole range, and specifically AT the Eq.58 value?
#
# Uses fig5_fresh_build's REAL already-computed Omega_total sweep
# (0.05 to ~11.23MHz) -- re-anchoring the SAME real curve at a
# different Omega_LO is just re-running the linear-range search at a
# different x_anchor on real data already in hand. NO new QuTiP solves
# needed, since the anchor sweep stays within the already-covered
# domain for every value tested here.
#
# Does not touch any frozen fresh_build file.
# ============================================================

from pathlib import Path
import csv

import numpy as np

OUTPUT_DIR = Path(__file__).resolve().parent
FIG5_CSV = OUTPUT_DIR.parent / "fig5_fresh_build" / "fig5_data.csv"

print("=" * 70)
print("*** CASE STUDY 02 -- Omega_LO anchor sweep ***")
print("=" * 70)

e_charge, a0, hbar, eps0, eta0 = 1.6e-19, 5.2e-11, 1.054571817e-34, 8.854e-12, 377.0
c_light = 2.998e8
wp_RF = -1443.459*e_charge*a0
lambda_p = 852e-9
wp_RF_over_hbar = wp_RF/hbar
fp = c_light/lambda_p
GLNA_lin, RL, D_resp, T_temp, eta_eff, B_bw = 100.0, 50.0, 0.55, 290.0, 0.8, 1.0e6
kB = 1.380649e-23
GTx_lin = 10**(2.15/10)
GRx_lin = GTx_lin
sigma_BGN_sq = 1e-3*10**(-90/10)
sigma_TN_sq = 4*kB*T_temp*B_bw
fRF = 3.5e9
lambda_RF = c_light/fRF

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
fig5a_pout_W = np.array(fig5a_pout_uW)*1e-6
print(f"Loaded real Fig.5(a) curve: domain [{fig5a_omega_mhz.min():.3f},{fig5a_omega_mhz.max():.3f}]MHz")

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

def gap_at_omega_LO(Omega_LO_mhz):
    fit = find_linear_range(fig5a_omega_mhz, fig5a_pout_W, Omega_LO_mhz)
    if fit is None:
        return None, None, None
    kappa_true = fit["slope"]/(2*np.pi*1e6)
    P0_bar = float(np.interp(Omega_LO_mhz, fig5a_omega_mhz, fig5a_pout_W))
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
    return gap_100m, kappa_true, fit

print("\nSweeping Omega_LO from 1.0 to 9.0 MHz (paper's stated value 4.23MHz, Eq.58's 'optimal' 2.9576MHz)")
print(f"{'Omega_LO(MHz)':>14} {'gap(dB)':>10} {'kappa':>14} {'lin_range':>22} {'note':>10}")
omega_LO_grid = np.arange(1.0, 9.01, 0.25)
best_gap = None
for w in omega_LO_grid:
    gap, kappa, fit = gap_at_omega_LO(w)
    note = ""
    if abs(w-4.23) < 0.13: note = "<- paper's stated"
    if abs(w-2.9576) < 0.13: note = "<- Eq.58 'optimal'"
    if gap is not None:
        print(f"{w:14.4f} {gap:10.4f} {kappa:14.4e} [{fit['x_low']:.2f},{fit['x_high']:.2f}]MHz{'':8} {note}")
        if best_gap is None or gap > best_gap[0]:
            best_gap = (gap, w)
    else:
        print(f"{w:14.4f} {'NO FIT':>10} {'--':>14} {'--':>22} {note}")

if best_gap:
    print(f"\nLargest gap found across this Omega_LO sweep: {best_gap[0]:.4f}dB at Omega_LO={best_gap[1]:.4f}MHz "
          f"(paper claims ~44dB)")

print("\nDONE.")
