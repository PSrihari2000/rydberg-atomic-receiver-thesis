# ============================================================
# *** CASE STUDY -- NOT a validated result ***
#
# Follow-up to 06's key finding: coverage is locked to gap via
#   coverage = conv_cross * (10^(gap/20) - 1)
# and conv_cross (Conventional's OWN 0dB crossing) needs to be
# 9.5245m to jointly satisfy 44dB+1500m -- but every real Conventional
# calculation gives 3.43-6.82m, too small. This asks: what SPECIFIC
# Conventional-side parameter change would close THAT gap (not the
# LO-dressed side at all)? Solves for the exact value of each candidate
# parameter, one at a time, holding everything else at its real value.
#
# Does not touch any frozen fresh_build file.
# ============================================================

import numpy as np

print("=" * 70)
print("*** CASE STUDY 07 -- what Conventional-side parameter would close it? ***")
print("=" * 70)

kB = 1.380649e-23
c_light = 2.998e8
GTx_dBi, GRx_dBi, GLNA_dB = 2.15, 2.15, 20.0
T_temp, B_bw = 290.0, 1.0e6
sigma_BGN_dBm = -90.0
fRF_baseline = 3.5e9

def db_to_lin(db): return 10**(db/10)
def dbm_to_w(dbm): return 1e-3*10**(dbm/10)
GTx_lin = db_to_lin(GTx_dBi)
GRx_lin = db_to_lin(GRx_dBi)
GLNA_lin = db_to_lin(GLNA_dB)
sigma_BGN_sq = dbm_to_w(sigma_BGN_dBm)
sigma_TN_sq = 4*kB*T_temp*B_bw
PTx_W = dbm_to_w(-10.0)
noise_floor = GRx_lin*GLNA_lin*sigma_BGN_sq + sigma_TN_sq

CONV_CROSS_REQUIRED = 9.5245   # from 06's exact analysis

def conv_cross(fRF=fRF_baseline, GTx_lin_=GTx_lin, GRx_lin_=GRx_lin, GLNA_lin_=GLNA_lin,
                sigma_BGN_sq_=sigma_BGN_sq, sigma_TN_sq_=sigma_TN_sq, extra_aperture_factor=1.0):
    lambda_RF = c_light/fRF
    noise = GRx_lin_*GLNA_lin_*sigma_BGN_sq_ + sigma_TN_sq_
    # d_cross: Pc(d) = noise -> PTx*GTx*lambda^2/((4pi)^2 d^2) * extra_aperture_factor = noise
    return np.sqrt(PTx_W*GTx_lin_*lambda_RF**2*extra_aperture_factor / ((4*np.pi)**2 * noise))

print(f"\nCurrent real values: conv_cross(fRF=3.5GHz)={conv_cross():.4f}m, "
      f"conv_cross(fRF=6.9458GHz)={conv_cross(fRF=6.9458e9):.4f}m")
print(f"Required: {CONV_CROSS_REQUIRED}m\n")

print("Candidate 1: what fRF alone would give conv_cross=9.5245m?")
fRF_needed = c_light * np.sqrt(PTx_W*GTx_lin/((4*np.pi)**2*noise_floor)) / CONV_CROSS_REQUIRED
print(f"  fRF_needed = {fRF_needed/1e9:.4f} GHz")
print(f"  (paper states fRF=3.5GHz; Jing et al.'s real transition is 6.9458GHz; neither is")
print(f"  close to {fRF_needed/1e9:.2f}GHz -- no recognizable transition/reference value found nearby)")

print("\nCandidate 2: what GTx (or GRx, symmetric roles) alone would give conv_cross=9.5245m?")
gain_factor_needed = (CONV_CROSS_REQUIRED/conv_cross())**2
GTx_dBi_needed = GTx_dBi + 10*np.log10(gain_factor_needed)
print(f"  Needs GTx*{gain_factor_needed:.4f}x -> GTx_dBi_needed = {GTx_dBi_needed:.4f}dBi "
      f"(paper states 2.15dBi -- a half-wave dipole, a standard specific antenna type, "
      f"not an arbitrary number)")

print("\nCandidate 3: what sigma_BGN alone would give conv_cross=9.5245m?")
# conv_cross ~ 1/sqrt(noise), noise ~ GRx*GLNA*sigma_BGN_sq (dominant term)
noise_factor_needed = (conv_cross()/CONV_CROSS_REQUIRED)**2
sigma_BGN_dBm_needed = sigma_BGN_dBm + 10*np.log10(noise_factor_needed)
print(f"  Needs noise*{noise_factor_needed:.4f}x -> sigma_BGN_needed = {sigma_BGN_dBm_needed:.4f}dBm "
      f"(paper states -90dBm)")

print("\nCandidate 4: what if the effective aperture formula is missing a factor "
      "(e.g. lambda^2 instead of lambda^2/(4pi), a full 4pi=12.57x=+11.0dB)?")
cc_no4pi = conv_cross(extra_aperture_factor=4*np.pi)
print(f"  conv_cross with NO /(4pi) in the aperture = {cc_no4pi:.4f}m (required: {CONV_CROSS_REQUIRED}m)")
print(f"  This is closer than any single-parameter fix above, but A_eff=G*lambda^2/(4*pi) is a")
print(f"  rigorous, unambiguous antenna-theory identity (not something the paper could realistically")
print(f"  have 'wrong' the way a transition frequency or density can be typo'd) -- flagged as an")
print(f"  interesting NUMBER, not a plausible physical explanation.")

print("\nCandidate 5: combined fRF fix (6.9458GHz, real) + some additional Conventional-side factor?")
remaining_factor_needed = (CONV_CROSS_REQUIRED/conv_cross(fRF=6.9458e9))**2
remaining_dB_needed = 10*np.log10(remaining_factor_needed)
print(f"  After the real fRF fix, still need an additional {remaining_dB_needed:.4f}dB of SOMETHING")
print(f"  on the Conventional side (higher Tx/Rx gain, lower noise floor, etc.) with no specific")
print(f"  candidate identified that matches this exactly.")

print("\nDONE.")
