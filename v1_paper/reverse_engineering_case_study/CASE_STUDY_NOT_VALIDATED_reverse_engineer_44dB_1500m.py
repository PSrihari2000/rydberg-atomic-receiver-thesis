# ============================================================
# *** CASE STUDY -- EXPLICITLY NOT A VALIDATED RESULT ***
# *** NOT adopted into any frozen baseline, NOT for submission ***
#
# Reverse-engineering exercise: what parameter value(s), in our existing
# real pipeline, WOULD be needed to hit the paper's claimed ~44dB gap and
# ~1500m extended coverage? This is deliberate trial-and-error, done to
# see whether the "required" value lands anywhere near something
# independently plausible (a typo, a literature value, a round number)
# -- a lead-generation technique, not a result. If the required value
# turns out arbitrary/implausible, that itself is the honest finding:
# no clean single-parameter explanation exists.
#
# Does NOT touch fig3/4/5/6_fresh_build or n0_density_sensitivity/.
# Reuses only their already-computed real Pout data (same exact
# provenance as n0_sensitivity_check.py), via the exact N0-rescaling
# identity proven there (verified to ~1e-15 against fresh QuTiP solves).
# ============================================================

from pathlib import Path
import csv

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import find_peaks
from scipy.optimize import brentq

OUTPUT_DIR = Path(__file__).resolve().parent
FIG3_NPZ = OUTPUT_DIR.parent / "fig3_fresh_build" / "fig3_quantum_response.npz"
FIG5_CSV = OUTPUT_DIR.parent / "fig5_fresh_build" / "fig5_data.csv"

print("=" * 70)
print("*** CASE STUDY -- reverse-engineering, NOT a validated result ***")
print("=" * 70)

e_charge, a0, hbar, eps0, eta0 = 1.6e-19, 5.2e-11, 1.054571817e-34, 8.854e-12, 377.0
c_light = 2.998e8
L_cell = 1.0e-2
Omega_p = 2.0*np.pi*8.0e6
Omega_c = 2.0*np.pi*1.0e6
wp_RF = -1443.459*e_charge*a0
wp_12 = (2.5*e_charge*a0)**2
lambda_p = 852e-9
kp_wave = 2.0*np.pi/lambda_p
d_probe = 0.76e-3
Pin = (np.pi/(2.0*eta0))*(d_probe*Omega_p*hbar/(2.0*np.sqrt(wp_12)))**2
wp_RF_over_hbar = wp_RF/hbar
fp = c_light/lambda_p

GLNA_lin, RL, D_resp, T_temp, eta_eff, B_bw = 100.0, 50.0, 0.55, 290.0, 0.8, 1.0e6
kB = 1.380649e-23
GTx_lin = 10**(2.15/10)
GRx_lin = GTx_lin
sigma_BGN_sq = 1e-3*10**(-90/10)
sigma_TN_sq = 4*kB*T_temp*B_bw

N0_BASELINE = 1.0e15

FRF_PAPER = 3.5e9
FRF_TRANSITION_CONSISTENT = 6.9458e9   # Jing et al. 2020's real value for this exact transition

# ------------------------------------------------------------
# Load real baseline data (same provenance as n0_sensitivity_check.py)
# ------------------------------------------------------------

data = np.load(FIG3_NPZ)
delta_c_mhz = data["delta_c_mhz"]
omega_rf_mhz = data["omega_rf_mhz"]
Pout_surface_baseline = data["Pout_surface"]

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

def rescale_pout(Pout_old, N0_new, N0_old=N0_BASELINE):
    return Pin * (Pout_old/Pin) ** (N0_new/N0_old)

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

def gap_and_coverage(N0_new, fRF):
    """Full Fig.4->Fig.5->Fig.6 pipeline at a given N0 and fRF, real
    equations throughout, only N0/fRF varied."""
    pout5_rescaled = rescale_pout(fig5a_pout_W_baseline, N0_new)
    fit = find_linear_range(fig5a_omega_mhz, pout5_rescaled, Omega_LO_mhz)
    if fit is None:
        return None
    kappa_true = fit["slope"]/(2*np.pi*1e6)
    P0_bar = float(np.interp(Omega_LO_mhz, fig5a_omega_mhz, pout5_rescaled))
    Ibar = eta_eff*P0_bar*e_charge/(hbar*fp)
    sigma_PSN_sq = 2*e_charge*B_bw*Ibar
    A_LO = GLNA_lin*RL*D_resp**2*kappa_true**2*wp_RF_over_hbar**2
    sigma_Ry_LO_sq = A_LO*sigma_BGN_sq + GLNA_lin*RL*D_resp**2*sigma_PSN_sq + sigma_TN_sq

    lambda_RF = c_light/fRF
    def snr_conv_dB(d, PTx_W=1e-4):
        Pc = (PTx_W*GTx_lin/(4*np.pi*d**2)) * (lambda_RF**2/(4*np.pi))
        return 10*np.log10(Pc/(GRx_lin*GLNA_lin*sigma_BGN_sq + sigma_TN_sq))
    def snr_theo_dB(d, PTx_W=1e-4):
        PRx = PTx_W*GTx_lin*eta0/(4*np.pi*d**2)
        return 10*np.log10(A_LO*PRx/sigma_Ry_LO_sq)

    gap_100m = snr_theo_dB(100.0) - snr_conv_dB(100.0)
    conv_cross = brentq(lambda d: snr_conv_dB(d), 1e-3, 1e12)
    theo_cross = brentq(lambda d: snr_theo_dB(d), 1e-3, 1e12)
    coverage = theo_cross - conv_cross
    return dict(gap_100m=gap_100m, conv_cross=conv_cross, theo_cross=theo_cross,
                coverage=coverage, kappa=kappa_true, P0_bar=P0_bar,
                lin_range=(fit["x_low"], fit["x_high"]))

# ------------------------------------------------------------
# STEP 1: baseline and fRF-fix-only reference points (already known,
# recomputed here for direct side-by-side comparison)
# ------------------------------------------------------------

print("\nSTEP 1: reference points")
r_base = gap_and_coverage(N0_BASELINE, FRF_PAPER)
r_frf = gap_and_coverage(N0_BASELINE, FRF_TRANSITION_CONSISTENT)
print(f"  Baseline (N0=1e15, fRF=3.5GHz):        gap={r_base['gap_100m']:.4f}dB, "
      f"coverage={r_base['coverage']:.4f}m, conv_cross={r_base['conv_cross']:.4f}m")
print(f"  +fRF fix only (N0=1e15, fRF=6.9458GHz): gap={r_frf['gap_100m']:.4f}dB, "
      f"coverage={r_frf['coverage']:.4f}m, conv_cross={r_frf['conv_cross']:.4f}m")
print(f"  Paper's claims: gap~44dB, coverage~1500m")

# ------------------------------------------------------------
# STEP 2: solve for the N0 that hits EXACTLY 44dB gap, with the fRF fix
# already applied (the most defensible starting point available)
# ------------------------------------------------------------

print("\nSTEP 2: solving for N0 that gives EXACTLY 44dB gap (fRF fix applied)")

def gap_minus_target(log10_N0, target=44.0):
    N0 = 10**log10_N0
    r = gap_and_coverage(N0, FRF_TRANSITION_CONSISTENT)
    if r is None:
        return np.nan
    return r["gap_100m"] - target

# bracket search
log10_N0_grid = np.linspace(15.0, 16.0, 21)
vals = [gap_minus_target(x) for x in log10_N0_grid]
for x, v in zip(log10_N0_grid, vals):
    print(f"  log10(N0)={x:.2f} (N0={10**x:.3e}): gap-44 = {v if not np.isnan(v) else 'NO LINEAR RANGE'}")

bracket = None
for k in range(len(log10_N0_grid)-1):
    if not np.isnan(vals[k]) and not np.isnan(vals[k+1]) and vals[k]*vals[k+1] < 0:
        bracket = (log10_N0_grid[k], log10_N0_grid[k+1])
        break

if bracket:
    log10_N0_solution = brentq(gap_minus_target, bracket[0], bracket[1], xtol=1e-6)
    N0_solution = 10**log10_N0_solution
    r_solution = gap_and_coverage(N0_solution, FRF_TRANSITION_CONSISTENT)
    print(f"\n  SOLUTION: N0 = {N0_solution:.4e} m^-3 gives EXACTLY 44dB gap")
    print(f"  At this N0: coverage={r_solution['coverage']:.4f}m (paper claims ~1500m), "
          f"linear range={r_solution['lin_range']}, kappa={r_solution['kappa']:.4e}")
    print(f"  Ratio to baseline N0 (1e15): {N0_solution/N0_BASELINE:.4f}x")
    print(f"  Ratio to Jing et al.'s real value (3.5e16): {N0_solution/3.5e16:.4f}x")
else:
    print("\n  No sign change found in the tested log10(N0) range [15,16] -- target may be outside this bracket.")
    N0_solution = None
    r_solution = None

# ------------------------------------------------------------
# STEP 3: solve for the N0 that hits EXACTLY 1500m coverage instead,
# see what gap that implies (the reverse direction, for completeness)
# ------------------------------------------------------------

print("\nSTEP 3: solving for N0 that gives EXACTLY 1500m coverage (fRF fix applied)")

def coverage_minus_target(log10_N0, target=1500.0):
    N0 = 10**log10_N0
    r = gap_and_coverage(N0, FRF_TRANSITION_CONSISTENT)
    if r is None:
        return np.nan
    return r["coverage"] - target

log10_N0_grid2 = np.linspace(15.0, 17.0, 41)
vals2 = [coverage_minus_target(x) for x in log10_N0_grid2]
bracket2 = None
for k in range(len(log10_N0_grid2)-1):
    if not np.isnan(vals2[k]) and not np.isnan(vals2[k+1]) and vals2[k]*vals2[k+1] < 0:
        bracket2 = (log10_N0_grid2[k], log10_N0_grid2[k+1])
        break

if bracket2:
    log10_N0_sol2 = brentq(coverage_minus_target, bracket2[0], bracket2[1], xtol=1e-6)
    N0_sol2 = 10**log10_N0_sol2
    r2 = gap_and_coverage(N0_sol2, FRF_TRANSITION_CONSISTENT)
    print(f"  SOLUTION: N0 = {N0_sol2:.4e} m^-3 gives EXACTLY 1500m coverage")
    print(f"  At this N0: gap={r2['gap_100m']:.4f}dB (paper claims ~44dB), "
          f"linear range={r2['lin_range']}")
    print(f"  Ratio to baseline N0 (1e15): {N0_sol2/N0_BASELINE:.4f}x")
else:
    print("  No sign change found in log10(N0) in [15,17] -- 1500m coverage may need an even larger N0,")
    print("  or may not be reachable at all via N0 alone while a linear region still exists.")
    for x, v in zip(log10_N0_grid2, vals2):
        print(f"    log10(N0)={x:.2f} (N0={10**x:.3e}): coverage-1500 = "
              f"{v if not np.isnan(v) else 'NO LINEAR RANGE'}")

print("\nDONE.")
