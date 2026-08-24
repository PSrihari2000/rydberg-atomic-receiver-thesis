# ============================================================
# *** CASE STUDY -- NOT a validated result ***
#
# Follow-up combining two earlier partial leads: the verified fRF fix
# (Conventional side, real, independently grounded) + the Eq.35 literal
# |P(Delta_f)|^2 reading (LO-dressed side, physically circular but
# numerically interesting from test 04). Does combining them get closer
# to 44dB, and what does #06's conv_cross/coverage relationship then say
# about coverage?
#
# Does not touch any frozen fresh_build file.
# ============================================================

import numpy as np

print("=" * 70)
print("*** CASE STUDY 08 -- combined fRF fix + Eq.35 literal reading ***")
print("=" * 70)

e_charge, hbar = 1.6e-19, 1.054571817e-34
c_light = 2.998e8
lambda_p = 852e-9
fp = c_light/lambda_p
eta_eff, B_bw = 0.8, 1.0e6
kB = 1.380649e-23
T_temp = 290.0
GLNA_lin, RL, D_resp = 100.0, 50.0, 0.55
GTx_lin = 10**(2.15/10)
GRx_lin = GTx_lin
eta0 = 377.0
sigma_TN_sq = 4*kB*T_temp*B_bw
sigma_BGN_sq = 1e-3*10**(-90/10)
wp_RF = -1443.459*1.6e-19*5.2e-11
wp_RF_over_hbar = wp_RF/hbar
kappa_true = -1.677464e-13
A_LO = GLNA_lin*RL*D_resp**2*kappa_true**2*wp_RF_over_hbar**2

FRF_FIX = 6.9458e9
lambda_RF = c_light/FRF_FIX

def snr_conv_dB(d, PTx_W=1e-4):
    Pc = (PTx_W*GTx_lin/(4*np.pi*d**2)) * (lambda_RF**2/(4*np.pi))
    return 10*np.log10(Pc/(GRx_lin*GLNA_lin*sigma_BGN_sq + sigma_TN_sq))

def snr_theo_dB_eq35_literal(d, PTx_W=1e-4):
    PRx = PTx_W*GTx_lin*eta0/(4*np.pi*d**2)
    Omega_RF = np.sqrt(PRx)*abs(wp_RF_over_hbar)
    P_delta_f = abs(kappa_true)*Omega_RF
    Ibar_literal = eta_eff*(P_delta_f**2)/(hbar*fp)*e_charge
    sigma_PSN_sq_literal = 2*e_charge*B_bw*Ibar_literal
    sigma_Ry_LO_sq_literal = A_LO*sigma_BGN_sq + GLNA_lin*RL*D_resp**2*sigma_PSN_sq_literal + sigma_TN_sq
    signal = A_LO*PRx
    return 10*np.log10(signal/sigma_Ry_LO_sq_literal)

gap_100m = snr_theo_dB_eq35_literal(100.0) - snr_conv_dB(100.0)
print(f"\nCombined (fRF fix + Eq.35 literal): gap @ d=100m,PTx=-10dBm = {gap_100m:.4f}dB "
      f"(paper claims ~44dB)")

from scipy.optimize import brentq
conv_cross = brentq(lambda d: snr_conv_dB(d), 1e-3, 1e9)
theo_cross = brentq(lambda d: snr_theo_dB_eq35_literal(d), 1e-3, 1e9)
coverage = theo_cross - conv_cross
print(f"conv_cross={conv_cross:.4f}m, theo_cross={theo_cross:.4f}m, coverage={coverage:.4f}m "
      f"(paper claims ~1500m)")

print(f"\nCross-check against #06's exact relationship (coverage = conv_cross*(10^(gap/20)-1)):")
predicted_coverage = conv_cross * (10**(gap_100m/20) - 1)
print(f"  predicted={predicted_coverage:.4f}m vs actual={coverage:.4f}m "
      f"(should match closely -- Eq.35's literal reading doesn't break the parallel-lines "
      f"property since Conventional is unaffected by it)")

print(f"\nVerdict: gap={gap_100m:.4f}dB is closer to 44dB than the real baseline (28.89dB) or even")
print(f"the fRF-fix-alone case (34.84dB), but coverage={coverage:.4f}m is STILL far short of 1500m")
print(f"-- confirms #06's finding again: conv_cross is the bottleneck, not the LO-dressed-side")
print(f"noise model, however it's read.")

print("\nDONE.")
