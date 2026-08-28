# ============================================================
# *** CASE STUDY -- NOT a validated result ***
#
# TEST: Eq.35's photocurrent formula, Ibar = eta_eff*|P(Delta_f)|^2/(hbar*fp)*e,
# uses the tiny AC signal |P(Delta_f)| (distance-dependent), not the DC
# operating point P0_bar, if taken completely literally. This project's
# established resolution uses P0_bar instead (physically motivated: shot
# noise should track the actual DC photon flux, not the small AC signal
# being measured) -- this test checks what the LITERAL reading would do,
# and whether it's even usable (it would make sigma_Ry_LO^2, hence SNR
# itself, depend on the very quantity being measured -- circular).
#
# Does not touch any frozen fresh_build file.
# ============================================================

import numpy as np

print("=" * 70)
print("*** CASE STUDY 04 -- Eq.35 Ibar: literal |P(df)| vs P0_bar reading ***")
print("=" * 70)

e_charge, hbar = 1.6e-19, 1.054571817e-34
c_light = 2.998e8
lambda_p = 852e-9
fp = c_light/lambda_p
eta_eff, B_bw = 0.8, 1.0e6
kB = 1.380649e-23
T_temp = 290.0
GLNA_lin, RL, D_resp = 100.0, 50.0, 0.55
sigma_TN_sq = 4*kB*T_temp*B_bw
sigma_BGN_sq = 1e-3*10**(-90/10)
wp_RF = -1443.459*1.6e-19*5.2e-11
wp_RF_over_hbar = wp_RF/hbar
kappa_true = -1.677464e-13   # real baseline value, from fig6_snr_vs_distance.py's run
P0_bar = 35.6693e-6           # real baseline value

GTx_lin = 10**(2.15/10)
eta0 = 377.0

def PRx_at(d, PTx_W=1e-4):
    return PTx_W*GTx_lin*eta0/(4*np.pi*d**2)

print("\nP0_bar-based (USED in the real baseline): Ibar and sigma_Ry_LO^2 are CONSTANT across distance")
Ibar_P0 = eta_eff*P0_bar*e_charge/(hbar*fp)
sigma_PSN_sq_P0 = 2*e_charge*B_bw*Ibar_P0
A_LO = GLNA_lin*RL*D_resp**2*kappa_true**2*wp_RF_over_hbar**2
sigma_Ry_LO_sq_P0 = A_LO*sigma_BGN_sq + GLNA_lin*RL*D_resp**2*sigma_PSN_sq_P0 + sigma_TN_sq
print(f"  Ibar={Ibar_P0:.4e}A, sigma_Ry_LO^2={sigma_Ry_LO_sq_P0:.4e}")

print("\nLiteral |P(Delta_f)|-based: |P(Delta_f)| = kappa*Omega_RF, itself a function of distance")
print("(this makes the noise floor depend on the very signal being measured -- checking what happens)")

def snr_theo_dB_literal(d, PTx_W=1e-4):
    PRx = PRx_at(d, PTx_W)
    Omega_RF = np.sqrt(PRx)*abs(wp_RF_over_hbar)
    P_delta_f = abs(kappa_true)*Omega_RF   # |P(df)| = |kappa|*Omega_RF, Eq.29 rearranged
    Ibar_literal = eta_eff*(P_delta_f**2)/(hbar*fp)*e_charge   # Eq.35 EXACTLY as literally printed (squared)
    sigma_PSN_sq_literal = 2*e_charge*B_bw*Ibar_literal
    sigma_Ry_LO_sq_literal = A_LO*sigma_BGN_sq + GLNA_lin*RL*D_resp**2*sigma_PSN_sq_literal + sigma_TN_sq
    signal = A_LO*PRx
    return 10*np.log10(signal/sigma_Ry_LO_sq_literal), sigma_Ry_LO_sq_literal

def snr_conv_dB(d, PTx_W=1e-4):
    fRF = 3.5e9
    lambda_RF = c_light/fRF
    GRx_lin = GTx_lin
    Pc = (PTx_W*GTx_lin/(4*np.pi*d**2)) * (lambda_RF**2/(4*np.pi))
    return 10*np.log10(Pc/(GRx_lin*GLNA_lin*sigma_BGN_sq + sigma_TN_sq))

print(f"\n{'d(m)':>8} {'SNR_theo_literal(dB)':>22} {'sigma_Ry_LO^2_literal':>22} {'SNR_conv(dB)':>14} {'gap(dB)':>10}")
for d in [1, 10, 100, 1000, 10000]:
    snr_t, sRy = snr_theo_dB_literal(d)
    snr_c = snr_conv_dB(d)
    print(f"{d:8.0f} {snr_t:22.4f} {sRy:22.4e} {snr_c:14.4f} {snr_t-snr_c:10.4f}")

print("\nNote: under the literal reading, the noise floor ITSELF changes with distance (unlike the")
print("P0_bar-based reading, where it's fixed) -- physically dubious (a receiver's own photon shot")
print("noise depending on the strength of the signal it hasn't detected yet is circular), but")
print("checking the numbers anyway per the open mandate.")
print("\nDONE.")
