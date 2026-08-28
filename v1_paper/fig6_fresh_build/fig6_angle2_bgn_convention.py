# ============================================================
# EXPLORATORY -- does the sigma^2_BGN convention (direct -90dBm-in-W
# value, vs squaring that value again) materially change Fig.6's
# Conventional/gap numbers?
#
# NEW file, does not touch any other fig6 file.
#
# The main build uses Convention A: sigma^2_BGN = dbm_to_w(-90) directly
# (matching the paper's own text calling -90dBm "the noise power").
# Convention B: treat -90dBm as an amplitude-like sigma_BGN, so
# sigma^2_BGN = (dbm_to_w(-90))^2.
# ============================================================

import numpy as np

print("=" * 70)
print("ANGLE 2: sigma^2_BGN convention -- direct vs squared")
print("=" * 70)

def dbm_to_w(dbm): return 1e-3 * 10.0**(dbm/10.0)
def db_to_lin(db): return 10.0**(db/10.0)

sigma_BGN_dBm = -90.0
sigma_BGN_direct = dbm_to_w(sigma_BGN_dBm)          # Convention A (used in the main build)
sigma_BGN_squared = sigma_BGN_direct ** 2            # Convention B (alternative reading)

print(f"-90dBm in Watts               = {sigma_BGN_direct:.4e} W")
print(f"Convention A (direct, used):  sigma^2_BGN = {sigma_BGN_direct:.4e}")
print(f"Convention B (squared again): sigma^2_BGN = {sigma_BGN_squared:.4e}")
print(f"Ratio A/B = {sigma_BGN_direct/sigma_BGN_squared:.4e}  (Convention B is essentially zero by comparison)")

GTx_dBi = 2.15
GRx_dBi = 2.15
GLNA_dB = 20.0
kB = 1.380649e-23
T = 290.0
B_bw = 1.0e6
c_light = 2.998e8
fRF = 3.5e9
lambda_RF = c_light / fRF

GTx_lin = db_to_lin(GTx_dBi)
GRx_lin = db_to_lin(GRx_dBi)
GLNA_lin = db_to_lin(GLNA_dB)
sigma_TN_sq = 4.0 * kB * T * B_bw

def snr_conventional_dB(PTx_dBm, d, sigma_BGN_sq):
    PTx_W = dbm_to_w(PTx_dBm)
    Pc = (PTx_W * GTx_lin / (4.0*np.pi*d**2)) * (lambda_RF**2 / (4.0*np.pi))
    gamma_conv = Pc / (GRx_lin * GLNA_lin * sigma_BGN_sq + sigma_TN_sq)
    return 10.0*np.log10(gamma_conv)

print(f"\nsigma^2_TN (thermal, fixed) = {sigma_TN_sq:.4e} W")

print(f"\n{'d(m)':>8} {'Conv_A(dB)':>12} {'Conv_B(dB)':>12} {'Difference':>12}")
for d in [1, 10, 100, 1000, 1e4, 1e5, 1e6]:
    a = snr_conventional_dB(-10.0, d, sigma_BGN_direct)
    b = snr_conventional_dB(-10.0, d, sigma_BGN_squared)
    print(f"{d:8.0f} {a:12.4f} {b:12.4f} {b-a:12.4f}")

print("\nWhich noise term dominates the Conventional denominator under each convention?")
print(f"  GRx*GLNA*sigma^2_BGN (Convention A) = {GRx_lin*GLNA_lin*sigma_BGN_direct:.4e}")
print(f"  GRx*GLNA*sigma^2_BGN (Convention B) = {GRx_lin*GLNA_lin*sigma_BGN_squared:.4e}")
print(f"  sigma^2_TN (thermal, either convention) = {sigma_TN_sq:.4e}")

# Recompute the reference checkpoint (d=100m, PTx=-10dBm) gap against Theoretical LO-dressed
A_LO = 5.519570e-07   # from the main script's real run
sigma_Ry_LO_sq = 7.556606e-14
eta0 = 377.0
def PRx_flux(PTx_dBm, d):
    return dbm_to_w(PTx_dBm) * GTx_lin * eta0 / (4.0*np.pi*d**2)
def snr_theoretical_dB(PTx_dBm, d):
    return 10.0*np.log10(A_LO * PRx_flux(PTx_dBm, d) / sigma_Ry_LO_sq)

d_ref, ptx_ref = 100.0, -10.0
conv_a = snr_conventional_dB(ptx_ref, d_ref, sigma_BGN_direct)
conv_b = snr_conventional_dB(ptx_ref, d_ref, sigma_BGN_squared)
theo = snr_theoretical_dB(ptx_ref, d_ref)
print(f"\nCheckpoint @ d=100m, PTx=-10dBm:")
print(f"  Theoretical LO-dressed = {theo:.4f}dB")
print(f"  Convention A: Conventional={conv_a:.4f}dB, gap={theo-conv_a:.4f}dB  (main build's real result)")
print(f"  Convention B: Conventional={conv_b:.4f}dB, gap={theo-conv_b:.4f}dB")
print("\nDONE.")
