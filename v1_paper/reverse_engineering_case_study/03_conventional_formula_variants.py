# ============================================================
# *** CASE STUDY -- NOT a validated result ***
#
# TEST: all defensible readings of the Conventional receiver's SNR
# formula (Fig.2's diagram + Sec.V-A's GRx/effective-area text), side
# by side, plus their effect on the checkpoint gap.
#
# Reading A: GRx*GLNA scales BOTH signal and noise (symmetric receiver
#   chain gain). Already tried, ruled out (makes Conventional beat
#   LO-dressed -- contradicts the paper's premise).
# Reading B: GRx*GLNA scales noise only, matching Fig.2's literal
#   diagram y=sqrt(PRx)*h*x+n_Conv. USED in the real fig6_fresh_build
#   baseline.
# Reading C (new here): full Friis with GRx baked directly into the
#   effective aperture (A_eff = GRx*lambda^2/(4*pi), not just
#   lambda^2/(4*pi)), no separate downstream GRx on the noise term either
#   -- i.e. GRx enters exactly once, as an aperture-capture factor, the
#   textbook-standard link-budget form with no additional gain-on-noise
#   term beyond GLNA.
#
# All real formulas, no fabrication -- just testing which combination,
# if any, gets closer to the paper's ~44dB.
#
# Does not touch any frozen fresh_build file.
# ============================================================

import numpy as np

print("=" * 70)
print("*** CASE STUDY 03 -- Conventional formula variants ***")
print("=" * 70)

kB = 1.380649e-23
c_light = 2.998e8
GTx_dBi, GRx_dBi, GLNA_dB = 2.15, 2.15, 20.0
T_temp, B_bw = 290.0, 1.0e6
sigma_BGN_dBm = -90.0
fRF = 3.5e9
lambda_RF = c_light/fRF

def db_to_lin(db): return 10**(db/10)
def dbm_to_w(dbm): return 1e-3*10**(dbm/10)
GTx_lin = db_to_lin(GTx_dBi)
GRx_lin = db_to_lin(GRx_dBi)
GLNA_lin = db_to_lin(GLNA_dB)
sigma_BGN_sq = dbm_to_w(sigma_BGN_dBm)
sigma_TN_sq = 4*kB*T_temp*B_bw

PTx_W = dbm_to_w(-10.0)
d = 100.0

def flux_S(d): return PTx_W*GTx_lin/(4*np.pi*d**2)

# Reading A: symmetric gain
Pc_A = flux_S(d) * (lambda_RF**2/(4*np.pi))
snr_A = 10*np.log10(Pc_A*GRx_lin*GLNA_lin / (GRx_lin*GLNA_lin*sigma_BGN_sq + sigma_TN_sq))

# Reading B: literal Fig.2 diagram (used in the real baseline)
Pc_B = flux_S(d) * (lambda_RF**2/(4*np.pi))
snr_B = 10*np.log10(Pc_B / (GRx_lin*GLNA_lin*sigma_BGN_sq + sigma_TN_sq))

# Reading C: GRx baked into the aperture, GLNA only on noise (no GRx-on-noise separately)
Pc_C = flux_S(d) * (GRx_lin*lambda_RF**2/(4*np.pi))
snr_C = 10*np.log10(Pc_C*GLNA_lin / (GLNA_lin*sigma_BGN_sq + sigma_TN_sq))

# Reading D: GRx baked into aperture AND still separately multiplies noise (double-counted, for completeness)
Pc_D = flux_S(d) * (GRx_lin*lambda_RF**2/(4*np.pi))
snr_D = 10*np.log10(Pc_D*GLNA_lin / (GRx_lin*GLNA_lin*sigma_BGN_sq + sigma_TN_sq))

print(f"\nCheckpoint @ d=100m, PTx=-10dBm, Conventional SNR under each reading:")
print(f"  Reading A (symmetric GRx*GLNA on signal+noise):        {snr_A:.4f}dB")
print(f"  Reading B (literal Fig.2, GRx*GLNA on noise only, USED): {snr_B:.4f}dB")
print(f"  Reading C (GRx baked into aperture, GLNA-only on noise): {snr_C:.4f}dB")
print(f"  Reading D (GRx in aperture AND on noise, double-count): {snr_D:.4f}dB")

# Theoretical LO-dressed reference (baseline real value)
theo_ref = 5.5571  # from the real fig6_snr_vs_distance.py run at N0=1e15, fRF=3.5GHz baseline
print(f"\nTheoretical LO-dressed SNR (real baseline) = {theo_ref}dB")
for name, snr in [("A", snr_A), ("B (used)", snr_B), ("C", snr_C), ("D", snr_D)]:
    print(f"  Gap under Reading {name}: {theo_ref-snr:.4f}dB")

print("\nDONE.")
