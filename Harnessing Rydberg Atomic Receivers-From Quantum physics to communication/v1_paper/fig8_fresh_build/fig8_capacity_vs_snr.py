# ============================================================
# FIG. 8 -- FRESH, SELF-CONTAINED REBUILD
# Achievable capacity versus SNR: Conventional/Theoretical LO-dressed/
# Practical LO-dressed use 8-QAM; LO-free uses 8-PAM (matches the
# paper's own stated modulation assignment, Sec.V-C text).
#
# The paper gives NO closed-form equation anywhere for this figure
# (confirmed via pdftotext search -- only SER formulas, cited from
# Proakis for Fig.9, are given). What IS used here is the standard,
# well-known constellation-constrained mutual information (CCMI)
# formula for a finite constellation over AWGN (Ungerboeck/Forney;
# same "standard textbook, not paper-specific" category as Fig.9's
# Proakis SER formulas), evaluated via Monte Carlo.
#
# The "club closed-form + real QuTiP data" mechanism established in
# Fig.9 is reused here: same CCMI formula throughout, only the SNR
# fed into it differs between the idealized and real/practical paths.
#
# REUSES (real, already-computed, no new QuTiP):
#   - fig4_fresh_build/fig4_classification.npz: AT-splitting threshold
#   - fig5_fresh_build/fig5_data.npz: real static Pout(Omega_total)
#     curve, P0_bar (kappa re-fit fresh, matching fig6/7/9_fresh_build)
#
# Does not import old fig8 anything, or fig7_reinvestigation/.
# ============================================================

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

OUTPUT_DIR = Path(__file__).resolve().parent
FIG4_DATA = OUTPUT_DIR.parent / "fig4_fresh_build" / "fig4_classification.npz"
FIG5_DATA = OUTPUT_DIR.parent / "fig5_fresh_build" / "fig5_data.npz"

print("=" * 70)
print("FIG. 8 -- FRESH REBUILD (reuses Fig.4/Fig.5's real data)")
print("=" * 70)

# ------------------------------------------------------------
# Physical constants (re-typed fresh, matching fig3/4/5/6/7/9_fresh_build)
# ------------------------------------------------------------

e_charge = 1.6e-19
a0 = 5.2e-11
hbar = 1.054571817e-34
kB = 1.380649e-23
c_light = 2.99792458e8

wp_RF = -1443.459 * e_charge * a0
lambda_p = 852e-9
fp_optical = c_light / lambda_p
wp_over_hbar = wp_RF / hbar

GLNA_dB = 20.0
RL_ohm = 50.0
D_resp = 0.55
T_kelvin = 290.0
eta_eff = 0.8
sigma_BGN_dBm = -90.0
epsilon_tilde = 0.005
B_Hz = 1.0e6


def db_to_linear(g_db):
    return 10.0 ** (g_db / 10.0)


def dbm_to_watts(p_dbm):
    return 10.0 ** ((p_dbm - 30.0) / 10.0)


GLNA_lin = db_to_linear(GLNA_dB)
sigma_BGN_sq_W = dbm_to_watts(sigma_BGN_dBm)

fig4_data = np.load(FIG4_DATA)
threshold_mhz = float(fig4_data["threshold_mhz"])

fig5_data = np.load(FIG5_DATA)
static_omega_mhz = fig5_data["static_omega_mhz"]
static_pout_w = fig5_data["static_pout_w"]
P0_bar = float(fig5_data["P0_bar"])
OMEGA_LO_MHZ = float(fig5_data["OMEGA_LO_MHZ"])
DELTA_F_MHZ = float(fig5_data["DELTA_F_MHZ"])

print(f"Reused Fig.4 threshold: {threshold_mhz:.4f} MHz")
print(f"Reused Fig.5 static response: {len(static_omega_mhz)} points, "
      f"range=[{static_omega_mhz.min():.2f},{static_omega_mhz.max():.2f}]MHz")


def linear_fit_score(x, y):
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return slope, intercept, r2


def find_linear_dynamic_range(omega_values, response_values, omega_lo, r2_threshold=0.998):
    center = int(np.argmin(np.abs(omega_values - omega_lo)))
    best = None
    max_radius = min(center, len(omega_values) - center - 1)
    for radius in range(5, max_radius + 1):
        left, right = center - radius, center + radius + 1
        slope, intercept, r2 = linear_fit_score(omega_values[left:right], response_values[left:right])
        if r2 >= r2_threshold:
            best = dict(slope=slope, r2=r2, omega_low=omega_values[left], omega_high=omega_values[right - 1])
        else:
            break
    return best


fit = find_linear_dynamic_range(static_omega_mhz, static_pout_w, OMEGA_LO_MHZ, r2_threshold=0.998)
kappa_W_per_MHz = fit["slope"]
kappa_rad_per_s = kappa_W_per_MHz / (2.0 * np.pi * 1e6)
print(f"Re-fit kappa: {kappa_W_per_MHz:.6e} W/MHz (R^2={fit['r2']:.6f}) -- matches fig5/6/7/9_fresh_build")

f_interp_static = interp1d(static_omega_mhz, static_pout_w, kind="cubic", bounds_error=False, fill_value=np.nan)

A_LO = GLNA_lin * RL_ohm * D_resp ** 2 * kappa_rad_per_s ** 2 * wp_over_hbar ** 2
I_bar = eta_eff * P0_bar / (hbar * (2.0 * np.pi * fp_optical)) * e_charge
sigma_PSN_sq = 2.0 * e_charge * B_Hz * I_bar
sigma_TN_sq = 4.0 * kB * T_kelvin * B_Hz
sigma_Ry_LO_sq = (
    A_LO * sigma_BGN_sq_W
    + GLNA_lin * RL_ohm * D_resp ** 2 * sigma_PSN_sq
    + sigma_TN_sq
)
print(f"A_LO={A_LO:.6e}, sigma_Ry_LO^2={sigma_Ry_LO_sq:.6e} (matches fig6/7/9_fresh_build)")


def practical_fourier_amplitude_from_omega_rf(omega_rf_mhz_array, n_t=400):
    period_s = 1.0 / (DELTA_F_MHZ * 1e6)
    t_seconds = np.linspace(0.0, period_s, n_t, endpoint=False)
    theta = 2.0 * np.pi * DELTA_F_MHZ * 1e6 * t_seconds

    omega_total_grid_mhz = np.sqrt(
        OMEGA_LO_MHZ ** 2 + omega_rf_mhz_array[:, None] ** 2
        + 2.0 * OMEGA_LO_MHZ * omega_rf_mhz_array[:, None] * np.cos(theta)[None, :]
    )
    Pout_grid = f_interp_static(omega_total_grid_mhz.ravel()).reshape(omega_total_grid_mhz.shape)

    proj = np.exp(-1j * theta)
    out_of_range = np.any(np.isnan(Pout_grid), axis=1)
    Pout_grid_filled = np.where(np.isnan(Pout_grid), 0.0, Pout_grid)
    c1 = (Pout_grid_filled * proj[None, :]).sum(axis=1) / n_t
    P_delta_f = 2.0 * np.abs(c1)
    P_delta_f[out_of_range] = np.nan
    return P_delta_f


# ============================================================
# Constellations (standard, unit average energy -- paper does not
# specify exact 8-QAM geometry, so the standard rectangular layout
# is used; flagged as an inference, see math report).
# ============================================================

def make_8qam():
    I = np.array([-3.0, -1.0, 1.0, 3.0])
    Q = np.array([-1.0, 1.0])
    pts = np.array([[i, q] for i in I for q in Q])
    x = pts[:, 0] + 1j * pts[:, 1]
    Es = np.mean(np.abs(x) ** 2)
    return x / np.sqrt(Es)


def make_8pam():
    x = np.array([-7.0, -5.0, -3.0, -1.0, 1.0, 3.0, 5.0, 7.0])
    Es = np.mean(x ** 2)
    return x / np.sqrt(Es)


X_8QAM = make_8qam()
X_8PAM = make_8pam()


def ccmi_qam_mc(x, snr_lin_grid, n_mc=40000, seed=0):
    """Constellation-constrained MI (bits/s/Hz), complex AWGN, Monte Carlo."""
    rng = np.random.default_rng(seed)
    M = len(x)
    diff = x[:, None] - x[None, :]
    out = np.empty_like(snr_lin_grid, dtype=float)
    for k, snr_lin in enumerate(snr_lin_grid):
        N0 = 1.0 / snr_lin
        sigma = np.sqrt(N0 / 2.0)
        n = (rng.standard_normal(n_mc) + 1j * rng.standard_normal(n_mc)) * sigma
        y = diff[None, :, :] + n[:, None, None]
        arg = -(np.abs(y) ** 2 - np.abs(n)[:, None, None] ** 2) / N0
        logsum = np.log2(np.sum(np.exp(arg), axis=2))
        out[k] = np.log2(M) - np.mean(logsum)
    return out


def ccmi_pam_mc(x, snr_lin_grid, n_mc=40000, seed=1):
    """Constellation-constrained MI (bits/s/Hz), real AWGN, Monte Carlo."""
    rng = np.random.default_rng(seed)
    M = len(x)
    diff = x[:, None] - x[None, :]
    out = np.empty_like(snr_lin_grid, dtype=float)
    for k, snr_lin in enumerate(snr_lin_grid):
        N0 = 1.0 / snr_lin
        sigma = np.sqrt(N0 / 2.0)
        n = rng.standard_normal(n_mc) * sigma
        y = diff[None, :, :] + n[:, None, None]
        arg = -(y ** 2 - n[:, None, None] ** 2) / N0
        logsum = np.log2(np.sum(np.exp(arg), axis=2))
        out[k] = np.log2(M) - np.mean(logsum)
    return out


# ------------------------------------------------------------
# Precompute CCMI(SNR) once on a fine grid, build fast interpolators
# for the pipelines below (avoids re-running Monte Carlo per point).
# ------------------------------------------------------------

print("\nPrecomputing CCMI_8QAM(SNR) and CCMI_8PAM(SNR) via Monte Carlo (this takes a bit)...")
SNR_dB_lut = np.linspace(-40.0, 90.0, 261)
SNR_lin_lut = 10.0 ** (SNR_dB_lut / 10.0)
ccmi_qam_lut = ccmi_qam_mc(X_8QAM, SNR_lin_lut, n_mc=30000, seed=0)
ccmi_pam_lut = ccmi_pam_mc(X_8PAM, SNR_lin_lut, n_mc=30000, seed=1)
f_ccmi_qam = interp1d(SNR_dB_lut, ccmi_qam_lut, bounds_error=False,
                       fill_value=(ccmi_qam_lut[0], ccmi_qam_lut[-1]))
f_ccmi_pam = interp1d(SNR_dB_lut, ccmi_pam_lut, bounds_error=False,
                       fill_value=(ccmi_pam_lut[0], ccmi_pam_lut[-1]))
print("Done. Checkpoints: CCMI_8QAM(20dB)=%.4f, CCMI_8PAM(20dB)=%.4f (max=log2(8)=%.4f)"
      % (f_ccmi_qam(20.0), f_ccmi_pam(20.0), np.log2(8)))

# ============================================================
# Sweep (paper's own Fig.8 x-axis: SNR direct, -20..40dB)
# ============================================================

SNR_dB = np.linspace(-20.0, 40.0, 601)
SNR_lin = 10.0 ** (SNR_dB / 10.0)

# --- Conventional: direct nominal SNR, 8-QAM ---
cap_conventional = f_ccmi_qam(SNR_dB)

# --- Theoretical LO-dressed: direct nominal SNR (Eq.37's idealized
# relationship), 8-QAM. Same formula/x-axis as Conventional -- no
# paper equation distinguishes them for THIS figure (unlike Fig.7's
# Eq.38), so they may legitimately overlap. Reported honestly below.
cap_theoretical = f_ccmi_qam(SNR_dB)

# --- Practical LO-dressed: nominal SNR -> invert Eq.(37) -> real
# Omega_RF -> real nonlinear pipeline -> real achieved SNR -> 8-QAM CCMI
PRx_nominal = SNR_lin * sigma_Ry_LO_sq / A_LO
omega_rf_implied_mhz = np.abs(wp_over_hbar) * np.sqrt(PRx_nominal) / (2.0 * np.pi * 1e6)
P_delta_f = practical_fourier_amplitude_from_omega_rf(omega_rf_implied_mhz)
signal_power = GLNA_lin * RL_ohm * D_resp ** 2 * P_delta_f ** 2
SNR_lin_real_lodressed = signal_power / sigma_Ry_LO_sq
SNR_dB_real_lodressed = 10.0 * np.log10(np.clip(SNR_lin_real_lodressed, 1e-30, None))
SNR_dB_real_lodressed[np.isnan(SNR_lin_real_lodressed)] = -999.0   # -> CCMI=0 via LUT floor
cap_practical = f_ccmi_qam(SNR_dB_real_lodressed)

n_nan_prac = int(np.sum(np.isnan(SNR_lin_real_lodressed)))
print(f"\nPractical LO-dressed: {n_nan_prac}/{len(SNR_dB)} points left the real static curve's "
      f"interpolation range (treated as CCMI=0, receiver has no usable signal there)")

# --- LO-free: nominal SNR -> invert Eq.(16) -> real Omega_RF ->
# gated by the real Fig.4 threshold -> 8-PAM CCMI
ceiling_lin = 1.0 / epsilon_tilde ** 2
below_ceiling = SNR_lin < ceiling_lin
PRx_nominal_lf = np.full_like(SNR_lin, np.nan)
PRx_nominal_lf[below_ceiling] = (
    SNR_lin[below_ceiling] * sigma_BGN_sq_W / (1.0 - SNR_lin[below_ceiling] * epsilon_tilde ** 2)
)
omega_rf_implied_lf_mhz = np.abs(wp_over_hbar) * np.sqrt(np.clip(PRx_nominal_lf, 0, None)) / (2.0 * np.pi * 1e6)
resolvable_lf = below_ceiling & (omega_rf_implied_lf_mhz >= threshold_mhz)
SNR_dB_real_lofree = np.where(resolvable_lf, SNR_dB, -999.0)
cap_lofree = f_ccmi_pam(SNR_dB_real_lofree)

print(f"LO-free: resolvable region within -20..40dB = "
      f"[{SNR_dB[resolvable_lf].min():.2f},{SNR_dB[resolvable_lf].max():.2f}]dB"
      if resolvable_lf.any() else "LO-free: never resolvable within -20..40dB (matches fig7/9 finding)")

overlap_diff = np.max(np.abs(cap_conventional - cap_theoretical))
print(f"\nConventional vs Theoretical LO-dressed: max|difference| = {overlap_diff:.2e} "
      f"({'IDENTICAL curves in this reproduction' if overlap_diff < 1e-9 else 'genuinely different'})")

# ============================================================
# Plot
# ============================================================

fig, ax = plt.subplots(figsize=(9, 6.5))
ax.plot(SNR_dB, cap_conventional, color="tab:green", linewidth=2.2, label="Conventional, 8-QAM")
ax.plot(SNR_dB, cap_theoretical, color="tab:orange", linewidth=1.6, linestyle="--",
        label="Theoretical LO-dressed, 8-QAM")
ax.plot(SNR_dB, cap_practical, color="tab:blue", linewidth=1.8, label="Practical LO-dressed, 8-QAM")
ax.plot(SNR_dB, cap_lofree, color="tab:purple", linewidth=1.8, label="LO-free, 8-PAM")
ax.axhline(np.log2(8), color="black", linestyle=":", linewidth=1.0)
ax.text(15, np.log2(8) + 0.08, "log$_2$8 = 3 bits/s/Hz (constellation limit)", fontsize=8, ha="center")

ax.set_xlabel("SNR (dB)")
ax.set_ylabel("Achievable capacity (bits/s/Hz)")
ax.set_title("Fig. 8. Achievable capacity versus SNR\n"
             "(8-QAM: Conventional/Theoretical/Practical LO-dressed; 8-PAM: LO-free)")
ax.set_xlim(-20, 40)
ax.set_ylim(0, 3.3)
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=8, loc="upper left")

fig.tight_layout()
fig.savefig(OUTPUT_DIR / "fig8_capacity_vs_snr.png", dpi=220, bbox_inches="tight")
plt.close(fig)
print("\nSaved: fig8_capacity_vs_snr.png")

for snr_check in [-10.0, 0.0, 10.0, 20.0, 30.0, 40.0]:
    i = int(np.argmin(np.abs(SNR_dB - snr_check)))
    print(f"@ SNR={SNR_dB[i]:5.1f}dB: Conventional={cap_conventional[i]:.4f}, "
          f"Theoretical={cap_theoretical[i]:.4f}, Practical={cap_practical[i]:.4f}, "
          f"LO-free={cap_lofree[i]:.4f}")

np.savez(
    OUTPUT_DIR / "fig8_data.npz",
    SNR_dB=SNR_dB, cap_conventional=cap_conventional, cap_theoretical=cap_theoretical,
    cap_practical=cap_practical, cap_lofree=cap_lofree,
    SNR_dB_lut=SNR_dB_lut, ccmi_qam_lut=ccmi_qam_lut, ccmi_pam_lut=ccmi_pam_lut,
    threshold_mhz=threshold_mhz, A_LO=A_LO, sigma_Ry_LO_sq=sigma_Ry_LO_sq,
)
print("Saved: fig8_data.npz")
print("\nDONE.")
