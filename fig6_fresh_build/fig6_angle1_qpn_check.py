# ============================================================
# EXPLORATORY -- is sigma^2_QPN (quantum projection noise, Eq.12)
# actually negligible for OUR system, or just assumed so?
#
# NEW file, does not touch any other fig6 file.
#
# Paper: "sigma^2_QPN ~= 1/(4*Ceff*Natoms) ... has a low magnitude and
# can be reasonably ignored, when applying an extremely large value of
# Natoms." No Ceff value is stated -- Ceff<=1 is a defined upper bound
# (Ceff=1 = ideal readout), so using Ceff=1 gives the SMALLEST/most
# generous possible sigma^2_QPN, a best-case check.
#
# Natoms is computed for real from paper-stated parameters: total
# atom density N0=1e15 m^-3 (Sec.IV) times the real interaction volume
# (vapor cell length L=1cm x cross-sectional area of the narrower,
# probe, beam, 1/e^2 diameter 0.76mm -- the probe beam is what's
# actually being read out, so its cross-section sets the relevant
# interaction volume).
#
# NOTE (flagged, not resolved): Eq.12 as literally written is
# dimensionless (Natoms is a pure count), while sigma^2_UN and
# sigma^2_BGN are power-like (W). Direct addition in Eq.15 is
# therefore not strictly dimensionally consistent as printed -- this
# is a known looseness in how Eq.12 is transcribed from its cited
# source [5, Eq.(38)] (Degen/Reinhard/Cappellaro, quantum sensing
# review), which typically expresses this as an uncertainty on a
# measured PROBABILITY, not a power. Not resolved here -- the paper's
# own instruction is to ignore this term regardless, so this check
# just verifies that instruction is well-justified in scale.
# ============================================================

import numpy as np

print("=" * 70)
print("ANGLE 1: sigma^2_QPN negligibility check, real Natoms")
print("=" * 70)

N0 = 1.0e15          # total atom density, m^-3, paper Sec.IV
L_cell = 1.0e-2       # vapor cell length, m
d_probe = 0.76e-3     # probe beam 1/e^2 diameter, m, paper Sec.IV
d_coupling = 1.95e-3  # coupling beam 1/e^2 diameter, m, paper Sec.IV

r_probe = d_probe / 2.0
V_interaction = np.pi * r_probe**2 * L_cell   # narrower (probe) beam sets the interaction volume
Natoms = N0 * V_interaction

Ceff = 1.0  # ideal readout upper bound (paper defines Ceff<=1, no explicit value given)
sigma_QPN_sq = 1.0 / (4.0 * Ceff * Natoms)

print(f"Interaction volume (probe-beam cylinder, d={d_probe*1e3:.2f}mm, L={L_cell*100:.1f}cm) "
      f"= {V_interaction:.4e} m^3")
print(f"Natoms = N0 x V = {N0:.2e} x {V_interaction:.4e} = {Natoms:.4e} atoms")
print(f"sigma^2_QPN (Eq.12, Ceff=1, best-case/smallest estimate) = 1/(4x{Natoms:.3e}) = {sigma_QPN_sq:.4e}")

print("\nFor comparison, the OTHER noise terms actually used in this project's Fig.6 (real units, W):")
sigma_BGN_sq = 1e-3 * 10**(-90/10)  # -90dBm in W, used directly as sigma^2_BGN
print(f"  sigma^2_BGN (-90dBm)            = {sigma_BGN_sq:.4e} W")
print(f"  sigma^2_UN at d=100m,PTx=-10dBm  ~ 1e-7 to 1e-5 W range (see fig6_data.csv Omega_RF columns)")
print(f"  sigma^2_Ry,LO (Eq.36, fixed)     = 7.556606e-14  (from fig6_snr_vs_distance.py's real run)")

ratio_to_bgn = sigma_QPN_sq / sigma_BGN_sq
print(f"\nsigma^2_QPN / sigma^2_BGN = {sigma_QPN_sq:.3e} / {sigma_BGN_sq:.3e} = {ratio_to_bgn:.3e}")
print("CORRECTED CONCLUSION: sigma^2_QPN (5.51e-08, DIMENSIONLESS as Eq.12 literally gives it)")
print("is actually numerically LARGER than sigma^2_BGN (1e-12 W) -- the opposite of 'negligible'")
print("if compared as raw numbers. But this comparison is NOT actually valid: Eq.12 is a pure")
print("(dimensionless) number -- Natoms is a count, Ceff is dimensionless -- while sigma^2_BGN")
print("and sigma^2_UN are power-like (W). They are not the same kind of quantity, so putting them")
print("side by side and calling one 'bigger' is comparing apples to oranges, not a real physical")
print("comparison. The paper's Eq.15 (sigma^2_Ry = sigma^2_UN + sigma^2_QPN + sigma^2_BGN) silently")
print("adds a dimensionless number to two power-valued ones -- as literally written, this is not")
print("dimensionally consistent, and no conversion factor is given in the paper text to fix it.")
print("This is a genuine, unresolved gap in the paper's own equation, not something this check")
print("can responsibly patch by guessing a conversion factor. The paper's qualitative claim")
print("('can be reasonably ignored... for extremely large Natoms') is not independently verifiable")
print("from what's actually stated -- it has to be taken on the paper's word, not confirmed here.")
print("\nDONE.")
