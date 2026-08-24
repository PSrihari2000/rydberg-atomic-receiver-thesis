# ============================================================
# *** CASE STUDY -- NOT a validated result ***
#
# TEST 1: which probe diameter (0.76mm literal, or 0.38mm = its radius)
# actually matches the PAPER'S OWN PUBLISHED Fig.3 peak value? Already
# known (fig3_fresh_build's own math report, Flag 1): d=0.76mm gives
# Pin=39.08uW, peak Pout=38.29uW -- a 3.95x mismatch against the
# paper's stated ~9.6-9.8uW peak. Since Pin~d^2, d=0.38mm would give
# Pin~9.77uW, matching almost exactly. Quantifies this precisely and
# checks the KNOCK-ON effect on Fig.6's gap/coverage if 0.38mm (the
# paper-plot-matching value) were used instead of 0.76mm (the
# paper-TEXT-stated value) -- combined with the verified fRF fix.
#
# TEST 2: incorporates the web-search-confirmed finding that Gong et al.
# 2412.05554 (same author group) uses L=1500m as a FIXED INPUT distance
# for an unrelated Doppler-broadening-free RAQR analysis (received
# amplitude -71.8dBV/m at pathloss exponent 2.0) -- independently
# re-confirming this project's earlier-documented suspicion that our
# paper's "~1500m" Fig.6 claim may be borrowed from there rather than
# derived from Fig.6's own stated parameters.
#
# Does not touch any frozen fresh_build file.
# ============================================================

from pathlib import Path
import csv

import numpy as np
from scipy.optimize import brentq

OUTPUT_DIR = Path(__file__).resolve().parent
FIG5_CSV = OUTPUT_DIR.parent / "fig5_fresh_build" / "fig5_data.csv"

print("=" * 70)
print("*** CASE STUDY 09 -- diameter-vs-paper-plot + web-search findings ***")
print("=" * 70)

print("\nTEST 1: which diameter matches the paper's OWN published Fig.3 peak?")
print(f"  d=0.76mm (paper TEXT-stated, used throughout this project): Pin=39.0761uW, "
      f"real computed peak Pout=38.2921uW")
print(f"  Paper's own PUBLISHED Fig.3 peak (read off the figure): ~9.6-9.8uW")
print(f"  Ratio: {38.2921/9.7:.4f}x too high at d=0.76mm")
d_needed_ratio = np.sqrt(9.7/38.2921)
d_needed = 0.76 * d_needed_ratio
print(f"  Diameter that WOULD match (Pin~d^2, solving for the ratio): d={d_needed:.4f}mm")
print(f"  This is almost exactly 0.76mm's RADIUS (0.38mm) -- {d_needed/0.38:.4f}x -- matching")
print(f"  the paper's own plot far better than its own stated 'diameter' value does.")

# ------------------------------------------------------------
# Knock-on effect on Fig.6 if using d=0.38mm (Fig.3-plot-matching) instead
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("Knock-on effect on Fig.6's gap/coverage: d=0.38mm (matches paper's Fig.3 plot)")
print("vs d=0.76mm (matches paper's Fig.3 TEXT), both combined with the verified fRF fix")
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

def gap_and_coverage_for_diameter(d_probe_new):
    pin_ratio = (d_probe_new/D_PROBE_BASELINE)**2
    pout_scaled = fig5a_pout_W_baseline * pin_ratio
    fit = find_linear_range(fig5a_omega_mhz, pout_scaled, Omega_LO_mhz)
    if fit is None:
        return None
    kappa_true = fit["slope"]/(2*np.pi*1e6)
    P0_bar = float(np.interp(Omega_LO_mhz, fig5a_omega_mhz, pout_scaled))
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
    conv_cross = brentq(lambda d: snr_conv_dB(d), 1e-3, 1e12)
    theo_cross = brentq(lambda d: snr_theo_dB(d), 1e-3, 1e12)
    return dict(gap_100m=gap_100m, coverage=theo_cross-conv_cross, conv_cross=conv_cross)

for label, d_probe in [("d=0.76mm (paper TEXT-stated)", 0.76e-3),
                        ("d=0.38mm (matches paper's Fig.3 PLOT)", 0.38e-3)]:
    r = gap_and_coverage_for_diameter(d_probe)
    print(f"  {label}: gap={r['gap_100m']:.4f}dB, coverage={r['coverage']:.4f}m "
          f"(both with the verified fRF fix applied)")

print("\nTEST 2: web-search-confirmed evidence on the '~1500m' claim's origin")
print("  Found (2026-08-24, WebSearch): Gong et al. 2412.05554 (same author group as our main")
print("  paper) states 'At a distance of 1500m with a pathloss exponent of 2.0, the amplitude")
print("  of the received RF signal is -71.8dBV/m' -- used as a FIXED INPUT for an unrelated")
print("  Doppler-broadening-free RAQR analysis, NOT a derived 0dB-SNR crossing point.")
print("  This independently RE-CONFIRMS (via a fresh web search, not just prior project memory)")
print("  that 1500m is a real, specific number this exact author group uses elsewhere as a")
print("  representative/illustrative distance -- consistent with (not proof of, but supporting)")
print("  the hypothesis that our main paper's Fig.6 '~1500m' is reused from there, rather than")
print("  independently derived under Fig.6's own stated Sec.IV/V-A parameters.")
print("  Combined with THIS case study's own finding (investigation 06) that 1500m coverage is")
print("  mathematically UNREACHABLE via any LO-dressed-side parameter under our real pipeline,")
print("  this is now TWO independent lines of evidence pointing the same direction.")

print("\nDONE.")
