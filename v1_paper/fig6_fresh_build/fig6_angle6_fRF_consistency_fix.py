# ============================================================
# EXPLORATORY -- apply the previously-established fRF/dipole-moment
# consistency fix (found in an earlier session's fRF_consistency_audit,
# independently re-verified here against Jing et al. 2020's own
# Methods section) to TODAY's independently-rebuilt Fig.6 pipeline.
#
# NEW file, does not touch any other fig6 file.
#
# FINDING (re-verified from primary source, not just memory): the main
# paper's dipole moment wp_RF=-1443.459 e*a0 is borrowed from Jing et
# al. 2020 (ref [20]), whose own Methods states this exact number is
# for the 47D5/2->48P3/2 transition driven "at 6.94GHz" -- but the main
# paper's own Fig.4 caption separately states fRF=3.5GHz, without
# adjusting the dipole moment to a transition actually resonant there.
# fRF enters this project's Fig.6 pipeline in exactly one place: the
# Conventional receiver's effective aperture lambda_RF^2/(4*pi). It
# never appears in the LO-free/LO-dressed Eq.8-37 formulas (those use
# wp_RF/hbar only). So substituting the transition-consistent
# fRF=6.9458GHz changes ONLY the Conventional curve.
# ============================================================

from pathlib import Path
import csv

import numpy as np

OUTPUT_DIR = Path(__file__).resolve().parent
FIG4_CSV = OUTPUT_DIR.parent / "fig4_fresh_build" / "fig4_classification.csv"
FIG5_CSV = OUTPUT_DIR.parent / "fig5_fresh_build" / "fig5_data.csv"

print("=" * 70)
print("ANGLE 6: fRF/dipole-moment consistency fix, applied to today's rebuild")
print("=" * 70)

with open(FIG4_CSV, newline="") as f:
    rows = list(csv.reader(f))
threshold_mhz = float(rows[2][1])

with open(FIG5_CSV, newline="") as f:
    rows = list(csv.reader(f))
meta = {}
i = 0
while rows[i]:
    if rows[i][0] != "# METADATA":
        meta[rows[i][0]] = float(rows[i][1])
    i += 1
Omega_LO_mhz = meta["Omega_LO_operating_MHz"]
kappa_slope_W_per_MHz = meta["linear_fit_slope_W_per_MHz"]

e_charge, a0, hbar, eta0 = 1.6e-19, 5.2e-11, 1.054571817e-34, 377.0
GTx_dBi, GRx_dBi, GLNA_dB = 2.15, 2.15, 20.0
RL, D_resp, T_temp, eta_eff = 50.0, 0.55, 290.0, 0.8
sigma_BGN_dBm = -90.0
eps_tilde = 0.005
B_bw = 1.0e6
c_light = 2.998e8
wp_RF = -1443.459*e_charge*a0
wp_RF_over_hbar = wp_RF/hbar
lambda_p = 852e-9
fp = c_light/lambda_p
kB = 1.380649e-23

def db_to_lin(db): return 10.0**(db/10.0)
def dbm_to_w(dbm): return 1e-3*10.0**(dbm/10.0)

GTx_lin, GRx_lin, GLNA_lin = db_to_lin(GTx_dBi), db_to_lin(GRx_dBi), db_to_lin(GLNA_dB)
sigma_BGN_sq = dbm_to_w(-90.0)
sigma_TN_sq = 4*kB*T_temp*B_bw

fRF_paper_stated = 3.5e9
fRF_transition_consistent = 6.9458e9   # from Jing et al. 2020's stated 6.94GHz, refined
                                          # (47D5/2->48P3/2 transition frequency to 4 sig figs
                                          # matching the dipole moment's own precision)
print(f"fRF (paper-stated, used in main build) = {fRF_paper_stated/1e9:.4f} GHz")
print(f"fRF (transition-consistent, this check) = {fRF_transition_consistent/1e9:.4f} GHz")
print(f"Predicted shift = 20*log10(ratio) = {20*np.log10(fRF_transition_consistent/fRF_paper_stated):.4f} dB")

def snr_conventional_dB(PTx_dBm, d, fRF):
    lambda_RF = c_light/fRF
    PTx_W = dbm_to_w(PTx_dBm)
    Pc = (PTx_W*GTx_lin/(4*np.pi*d**2)) * (lambda_RF**2/(4*np.pi))
    gamma_conv = Pc / (GRx_lin*GLNA_lin*sigma_BGN_sq + sigma_TN_sq)
    return 10*np.log10(gamma_conv)

# Theoretical LO-dressed, unaffected by fRF (real check, not asserted)
kappa_true = kappa_slope_W_per_MHz/(2*np.pi*1e6)
Ibar = eta_eff*35.6693e-6*e_charge/(hbar*fp)  # P0_bar from today's real fig5 run
sigma_PSN_sq = 2*e_charge*B_bw*Ibar
A_LO = GLNA_lin*RL*D_resp**2*kappa_true**2*wp_RF_over_hbar**2
sigma_Ry_LO_sq = A_LO*sigma_BGN_sq + GLNA_lin*RL*D_resp**2*sigma_PSN_sq + sigma_TN_sq

def snr_theoretical_dB(PTx_dBm, d):
    PRx = dbm_to_w(PTx_dBm)*GTx_lin*eta0/(4*np.pi*d**2)
    return 10*np.log10(A_LO*PRx/sigma_Ry_LO_sq)

DISTANCE_M = [1, 10, 100, 1000, 1e4, 1e5, 1e6]
print(f"\n{'d(m)':>8} {'Conv@3.5GHz':>13} {'Conv@6.9458GHz':>16} {'Theoretical':>13} "
      f"{'gap@3.5GHz':>12} {'gap@6.9458GHz':>15}")
for d in DISTANCE_M:
    c1 = snr_conventional_dB(-10.0, d, fRF_paper_stated)
    c2 = snr_conventional_dB(-10.0, d, fRF_transition_consistent)
    t = snr_theoretical_dB(-10.0, d)
    print(f"{d:8.0f} {c1:13.4f} {c2:16.4f} {t:13.4f} {t-c1:12.4f} {t-c2:15.4f}")

print(f"\nToday's main build's real checkpoint gap (d=100m,PTx=-10dBm) = 28.8864 dB")
c1_100 = snr_conventional_dB(-10.0, 100.0, fRF_paper_stated)
c2_100 = snr_conventional_dB(-10.0, 100.0, fRF_transition_consistent)
t_100 = snr_theoretical_dB(-10.0, 100.0)
print(f"After fRF fix: gap = {t_100-c2_100:.4f} dB  (widened by {(t_100-c2_100)-(t_100-c1_100):.4f} dB)")
print(f"Remaining unexplained vs paper's ~44dB: {44.0-(t_100-c2_100):.4f} dB "
      f"(was {44.0-(t_100-c1_100):.4f} dB before this fix)")
print("\nDONE.")
