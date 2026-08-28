# ============================================================
# FIG. 9(c) LO-FREE -- QUANTUM-DETECTOR RECONSTRUCTION
# (reconciliation against Documentation/02_Quantum_Distortion_
#  QuTiP_Implementation.pdf's blueprint, which was discovered
#  AFTER fig9_ser_vs_snr.py was already frozen -- this is a NEW
#  file, per project convention, not an edit of the frozen one)
#
# The frozen fig9_ser_vs_snr.py modeled the "distortion region"
# as a hard gate (SNR_real=0 below the AT-splitting threshold,
# saturating the Proakis PAM formula at its own ceiling). Doc.02
# instead specifies a REAL Monte-Carlo quantum detector: even
# below threshold, the receiver still attempts detection using
# its best (noisy, QuTiP-derived) AT-splitting reading, rather
# than being assumed to fail completely. This file builds that
# more faithful reconstruction and reports whether it changes
# the finding.
#
# REUSES (real, already-computed, no new QuTiP):
#   - fig3_fresh_build/fig3_quantum_response.npz: real Pout(Delta_c,
#     Omega_RF) grid -- used to build a REAL AT-splitting lookup
#     table via peak detection (doc02 Sec.12), not a new solve.
# ============================================================

from pathlib import Path

import numpy as np
from scipy.signal import find_peaks
from scipy.interpolate import interp1d

OUTPUT_DIR = Path(__file__).resolve().parent
FIG3_DATA = OUTPUT_DIR.parent / "fig3_fresh_build" / "fig3_quantum_response.npz"

print("=" * 70)
print("FIG. 9(c) LO-FREE -- QUANTUM-DETECTOR RECONSTRUCTION (doc02 reconciliation)")
print("=" * 70)

# ------------------------------------------------------------
# 1. Build the REAL AT-splitting lookup table from fig3's grid
#    (doc02 Sec.12.2-12.5). Peaks are in TRANSMISSION (Pout),
#    not absorption -- verified directly on the real data before
#    trusting find_peaks (doc02 Sec.12.3's own required check):
#    at small Omega_RF, Pout has ONE central peak at Delta_c=0
#    (EIT window); at large Omega_RF it splits into two peaks
#    flanking a central dip -- confirmed by direct inspection.
# ------------------------------------------------------------

fig3 = np.load(FIG3_DATA)
delta_c_mhz = fig3["delta_c_mhz"]
omega_rf_mhz = fig3["omega_rf_mhz"]
Pout_surface = fig3["Pout_surface"]

print(f"Loaded real grid: {Pout_surface.shape} (Omega_RF x Delta_c), "
      f"Omega_RF range=[{omega_rf_mhz.min():.2f},{omega_rf_mhz.max():.2f}]MHz")


def find_at_splitting(delta_c, pout_row):
    prominence = 0.01 * (pout_row.max() - pout_row.min())
    peaks, _ = find_peaks(pout_row, prominence=prominence)
    if len(peaks) < 2:
        return np.nan
    strongest = peaks[np.argsort(pout_row[peaks])[-2:]]
    strongest = np.sort(strongest)
    return abs(delta_c[strongest[1]] - delta_c[strongest[0]])


measured_splitting = np.array([find_at_splitting(delta_c_mhz, Pout_surface[i, :])
                                for i in range(len(omega_rf_mhz))])
valid = np.isfinite(measured_splitting)
print(f"Peaks resolved for {valid.sum()}/{len(omega_rf_mhz)} Omega_RF grid points "
      f"(lowest resolvable: {omega_rf_mhz[valid].min():.3f}MHz)")

# Relative measurement error vs the doc's own AT-resolution criterion
rel_err = np.full_like(measured_splitting, np.nan)
rel_err[valid] = np.abs(measured_splitting[valid] - omega_rf_mhz[valid]) / omega_rf_mhz[valid]
for tol in [0.10, 0.05, 0.02]:
    ok = valid & (rel_err <= tol)
    th = omega_rf_mhz[ok].min() if ok.any() else float("nan")
    print(f"  AT-resolution tolerance {tol*100:.0f}%: threshold Omega_RF = {th:.3f}MHz "
          f"(fig4_fresh_build's FWHM-based threshold: 5.500MHz)")

f_at_lookup = interp1d(omega_rf_mhz[valid], measured_splitting[valid], kind="linear",
                        bounds_error=False, fill_value=np.nan)  # no extrapolation (doc02 rule)

# ------------------------------------------------------------
# 2. Physical constants + REAL noise-derived measurement-noise
#    std, following the paper's own Eq.(14): sigma_UN=(eps*E_RF)^2
#    is defined ON the field amplitude, and Omega_RF is LINEAR in
#    E_RF via the dipole moment -- so the same sigma_Ry (Eq.15,
#    QPN~0) maps directly to an Omega_RF-domain noise std. This is
#    the "noise obtained from the LO-free noise model" doc02
#    requires (Sec.13.3), not an arbitrary value.
# ------------------------------------------------------------

e_charge = 1.6e-19
a0 = 5.2e-11
hbar = 1.054571817e-34
wp_RF = -1443.459 * e_charge * a0
wp_over_hbar = wp_RF / hbar
sigma_BGN_sq_W = 10 ** ((-90.0 - 30.0) / 10.0)
epsilon_tilde = 0.005


def prx_from_snr_lo_free(snr_lin):
    denom = 1.0 - snr_lin * epsilon_tilde ** 2
    prx = np.where(denom > 0, snr_lin * sigma_BGN_sq_W / np.where(denom > 0, denom, np.nan), np.nan)
    return prx


def omega_rf_mhz_from_prx(prx):
    return np.abs(wp_over_hbar) * np.sqrt(np.clip(prx, 0, None)) / (2.0 * np.pi * 1e6)


def splitting_noise_std_mhz(prx, snr_lin):
    """Omega_RF-domain std of the AT-splitting reading, from the SAME
    sigma_Ry = PRx/SNR noise budget used throughout this project (Eq.16
    inverted for sigma_Ry^2), converted via the same dipole-moment slope."""
    sigma_Ry_sq = prx / snr_lin
    return np.abs(wp_over_hbar) * np.sqrt(np.clip(sigma_Ry_sq, 0, None)) / (2.0 * np.pi * 1e6)


# ------------------------------------------------------------
# 3. Monte-Carlo quantum detector (doc02 Sec.13.1/13.3), faithful
#    to the paper's own "LO-free measures amplitude only" model:
#    PAM symbols are represented by their ABSOLUTE amplitude
#    (doc02's own code uses pam_levels_abs / np.abs(estimated_omega)
#    -- an amplitude-only receiver cannot distinguish sign, a real
#    physical limitation of this architecture, not a simplification
#    introduced here).
# ------------------------------------------------------------

def pam_levels_abs(M):
    levels = np.arange(-(M - 1), M, 2, dtype=float)
    levels = levels / np.sqrt(np.mean(levels ** 2))
    return np.abs(levels)


def lo_free_quantum_mc_ser(snr_db, M, n_symbols=200000, seed=0):
    rng = np.random.default_rng(seed)
    snr_lin = 10.0 ** (snr_db / 10.0)
    prx = prx_from_snr_lo_free(snr_lin)
    if not np.isfinite(prx):
        return np.nan, np.nan
    base_omega_rf_mhz = omega_rf_mhz_from_prx(prx)
    noise_std_mhz = splitting_noise_std_mhz(prx, snr_lin)

    levels_abs = pam_levels_abs(M)
    tx_idx = rng.integers(0, M, size=n_symbols)
    tx_amp = levels_abs[tx_idx]                      # normalized amplitude in [~0,1]
    true_omega_rf_symbols = base_omega_rf_mhz * tx_amp / levels_abs.max()

    measured = f_at_lookup(true_omega_rf_symbols)
    measured = measured + noise_std_mhz * rng.standard_normal(n_symbols)
    estimated_amp = np.abs(measured)
    estimated_amp = np.nan_to_num(estimated_amp, nan=0.0)

    rx_idx = np.argmin(np.abs(estimated_amp[:, None] - levels_abs[None, :] * base_omega_rf_mhz / levels_abs.max()),
                        axis=1)
    ser = np.mean(rx_idx != tx_idx)
    return ser, base_omega_rf_mhz


# ------------------------------------------------------------
# 4. Sweep and compare against the frozen (hard-gated) result
# ------------------------------------------------------------

SNR_dB = np.linspace(-40.0, 46.0, 87)   # coarser than the closed-form sweep -- MC is the expensive step
results = {log2M: np.full_like(SNR_dB, np.nan) for log2M in [2, 4, 6, 8]}
omega_used = np.full_like(SNR_dB, np.nan)

print("\nRunning Monte-Carlo quantum detector (200k symbols/point)...")
for i, snr_db in enumerate(SNR_dB):
    for log2M in [2, 4, 6, 8]:
        M = 2 ** log2M
        ser, orf = lo_free_quantum_mc_ser(snr_db, M, n_symbols=200000, seed=i * 10 + log2M)
        results[log2M][i] = ser
        if log2M == 2:
            omega_used[i] = orf

for snr_check in [-40.0, -20.0, 0.0, 20.0, 40.0, 45.0, 46.0]:
    i = int(np.argmin(np.abs(SNR_dB - snr_check)))
    print(f"SNR={SNR_dB[i]:6.2f}dB  Omega_RF(base)={omega_used[i]:.4e}MHz  "
          f"SER: log2M=2:{results[2][i]:.4f} 4:{results[4][i]:.4f} 6:{results[6][i]:.4f} 8:{results[8][i]:.4f}")

np.savez(OUTPUT_DIR / "fig9_lofree_quantum_detector.npz", SNR_dB=SNR_dB, omega_used=omega_used,
         **{f"ser_log2M{k}": v for k, v in results.items()},
         omega_rf_mhz=omega_rf_mhz, measured_splitting=measured_splitting)
print("\nSaved: fig9_lofree_quantum_detector.npz")
print("DONE.")
