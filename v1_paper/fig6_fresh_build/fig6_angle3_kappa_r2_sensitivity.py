# ============================================================
# EXPLORATORY -- how sensitive is kappa (Fig.5's linear-fit slope) to
# the R^2 threshold chosen for the expanding-window linear fit?
#
# NEW file, does not touch any other fig6 file. Reuses Fig.5's REAL
# saved Omega_total_MHz/Pout_uW curve (fig5_data.csv) -- no new QuTiP
# solves -- and re-runs the SAME fitting algorithm fig5_quutip.py used,
# at several different R^2 thresholds instead of just the one (0.998)
# that was used there.
# ============================================================

import csv
from pathlib import Path

import numpy as np

OUTPUT_DIR = Path(__file__).resolve().parent
FIG5_CSV = OUTPUT_DIR.parent / "fig5_fresh_build" / "fig5_data.csv"

print("=" * 70)
print("ANGLE 3: kappa sensitivity to Fig.5's R^2 fit threshold")
print("=" * 70)

with open(FIG5_CSV, newline="") as f:
    rows = list(csv.reader(f))
meta = {}
i = 0
while rows[i]:
    if rows[i][0] != "# METADATA":
        meta[rows[i][0]] = float(rows[i][1])
    i += 1
i += 3  # blank + "# FIG.5(a)..." comment + header
omega_mhz, pout_uW = [], []
while i < len(rows) and rows[i]:
    omega_mhz.append(float(rows[i][0]))
    pout_uW.append(float(rows[i][1]))
    i += 1
omega_mhz = np.array(omega_mhz)
pout_W = np.array(pout_uW) * 1e-6
Omega_LO_mhz = meta["Omega_LO_operating_MHz"]
print(f"Loaded {len(omega_mhz)} real Fig.5(a) points, Omega_LO={Omega_LO_mhz}MHz")
print(f"Original run (R^2>=0.998): kappa={meta['linear_fit_slope_W_per_MHz']:.6e} W/MHz, "
      f"range=[{meta['linear_range_lower_MHz']:.4f},{meta['linear_range_upper_MHz']:.4f}]MHz")

def linear_fit_score(x, y):
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope*x + intercept
    ss_res = np.sum((y-y_pred)**2)
    ss_tot = np.sum((y-np.mean(y))**2)
    r2 = 1.0 - ss_res/ss_tot if ss_tot > 0 else 1.0
    return slope, intercept, r2

def find_linear_range(x_vals, y_vals, x_anchor, r2_threshold):
    center = int(np.argmin(np.abs(x_vals - x_anchor)))
    best = None
    max_radius = min(center, len(x_vals)-center-1)
    for radius in range(3, max_radius+1):
        left, right = center-radius, center+radius+1
        slope, intercept, r2 = linear_fit_score(x_vals[left:right], y_vals[left:right])
        if r2 >= r2_threshold:
            best = dict(slope=slope, intercept=intercept, r2=r2,
                        x_low=x_vals[left], x_high=x_vals[right-1], n_pts=right-left)
        else:
            break
    return best

R2_VALUES = [0.990, 0.995, 0.998, 0.999, 0.9995]
print(f"\n{'R2_threshold':>12} {'kappa(W/MHz)':>16} {'range_low':>10} {'range_high':>10} {'n_pts':>6} {'kappa_vs_paper':>14}")
kappa_paper = meta['linear_fit_slope_W_per_MHz']
for r2t in R2_VALUES:
    fit = find_linear_range(omega_mhz, pout_W, Omega_LO_mhz, r2t)
    if fit is None:
        print(f"{r2t:12.4f}  NO FIT FOUND (even the smallest window fails this threshold)")
        continue
    pct = 100.0*(fit['slope']-kappa_paper)/abs(kappa_paper)
    print(f"{r2t:12.4f} {fit['slope']:16.6e} {fit['x_low']:10.4f} {fit['x_high']:10.4f} "
          f"{fit['n_pts']:6d} {pct:13.2f}%")

# ------------------------------------------------------------
# Propagate to the actual SNR checkpoint (Theoretical LO-dressed,
# d=100m, PTx=-10dBm) to see how much this actually matters downstream
# ------------------------------------------------------------
GLNA_lin, RL, D_resp = 100.0, 50.0, 0.55
e_charge, a0, hbar = 1.6e-19, 5.2e-11, 1.054571817e-34
wp_RF = -1443.459*e_charge*a0
wp_RF_over_hbar = wp_RF/hbar
sigma_BGN_sq = 1e-12
sigma_TN_sq = 1.601553e-14
eta0 = 377.0
GTx_lin = 10**(2.15/10)

def dbm_to_w(dbm): return 1e-3*10**(dbm/10)

# Recompute sigma_Ry_LO^2 and SNR for each kappa candidate (Ibar/P0_bar unaffected by kappa)
P0_bar = 35.6693e-6
eta_eff = 0.8
fp = 2.998e8/852e-9

print(f"\n{'R2_threshold':>12} {'kappa(W/rad/s)':>16} {'A_LO':>12} {'SNR_theo_dB@100m,-10dBm':>26}")
for r2t in R2_VALUES:
    fit = find_linear_range(omega_mhz, pout_W, Omega_LO_mhz, r2t)
    if fit is None:
        continue
    kappa_true = fit['slope'] / (2*np.pi*1e6)
    Ibar = eta_eff*P0_bar*e_charge/(hbar*fp)
    sigma_PSN_sq = 2*e_charge*1e6*Ibar
    A_LO = GLNA_lin*RL*D_resp**2*kappa_true**2*wp_RF_over_hbar**2
    sigma_Ry_LO_sq = A_LO*sigma_BGN_sq + GLNA_lin*RL*D_resp**2*sigma_PSN_sq + sigma_TN_sq
    PRx = dbm_to_w(-10.0)*GTx_lin*eta0/(4*np.pi*100.0**2)
    snr_theo = 10*np.log10(A_LO*PRx/sigma_Ry_LO_sq)
    print(f"{r2t:12.4f} {kappa_true:16.6e} {A_LO:12.4e} {snr_theo:26.4f}")

print("\nDONE.")
