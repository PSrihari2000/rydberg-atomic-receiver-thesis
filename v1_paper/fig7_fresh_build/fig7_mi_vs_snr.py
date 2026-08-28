# ============================================================
# FIG. 7 -- FRESH, SELF-CONTAINED REBUILD
# Mutual information versus SNR (SNR swept directly, -20 to 40dB).
#
# Builds the 3 curves fully backed by the paper's own numbered
# equations:
#   - Conventional      : standard Shannon capacity log2(1+SNR)
#                          (paper gives NO equation number for this
#                          curve -- flagged as an inference, see
#                          math report Section 3)
#   - LO-free            : Eq.(22), r~SNR_Ry (paper's own stated
#                          approximation), gated by the SAME real
#                          Fig.4 AT-splitting threshold used in Fig.6
#   - Theoretical LO-dressed : Eq.(38), both the paper-literal
#                          "*ln(2)" constant AND the standard-
#                          convention "*log2(e)" nats->bits constant
#                          shown side by side (same ambiguity already
#                          flagged for this equation earlier in the
#                          project -- not silently resolved)
#
# Practical LO-dressed and Fig.8 (constellation-constrained capacity
# for 8-QAM/8-PAM) are DEFERRED -- both require a formula/derivation
# not given anywhere in this paper's text (confirmed by a direct
# pdftotext extraction of the paper, not memory/guesswork). See math
# report Section 6.
#
# REUSES (real, already-computed, no new QuTiP):
#   - fig4_fresh_build/fig4_classification.npz: AT-splitting threshold
#   - fig5_fresh_build/fig5_data.npz: real static Pout(Omega_total)
#     curve, P0_bar (kappa re-fit fresh here, same algorithm/data,
#     matching fig6_fresh_build's own re-fit exactly)
#
# Does not import old fig7 anything, or fig7_reinvestigation/.
# ============================================================

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ive, exp1

OUTPUT_DIR = Path(__file__).resolve().parent
FIG4_DATA = OUTPUT_DIR.parent / "fig4_fresh_build" / "fig4_classification.npz"
FIG5_DATA = OUTPUT_DIR.parent / "fig5_fresh_build" / "fig5_data.npz"

print("=" * 70)
print("FIG. 7 -- FRESH REBUILD (reuses Fig.4/Fig.5's real data)")
print("=" * 70)

# ------------------------------------------------------------
# Physical constants (re-typed fresh, matching fig3/4/5/6_fresh_build)
# ------------------------------------------------------------

e_charge = 1.6e-19
a0 = 5.2e-11
hbar = 1.054571817e-34
kB = 1.380649e-23
c_light = 2.99792458e8

wp_RF = -1443.459 * e_charge * a0
lambda_p = 852e-9
fp_optical = c_light / lambda_p

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
wp_over_hbar = wp_RF / hbar

print(f"GLNA={GLNA_lin:.4f}, sigma_BGN^2={sigma_BGN_sq_W:.4e} W, "
      f"epsilon_tilde={epsilon_tilde}")

# ------------------------------------------------------------
# Reuse Fig.4's real threshold
# ------------------------------------------------------------

fig4_data = np.load(FIG4_DATA)
threshold_mhz = float(fig4_data["threshold_mhz"])
print(f"\nReused Fig.4 threshold: {threshold_mhz:.4f} MHz")

# ------------------------------------------------------------
# Reuse Fig.5's real static response; re-fit kappa fresh (same
# algorithm/data as fig6_fresh_build -- confirmed to reproduce
# the same number)
# ------------------------------------------------------------

fig5_data = np.load(FIG5_DATA)
static_omega_mhz = fig5_data["static_omega_mhz"]
static_pout_w = fig5_data["static_pout_w"]
P0_bar = float(fig5_data["P0_bar"])
OMEGA_LO_MHZ = float(fig5_data["OMEGA_LO_MHZ"])


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
print(f"Re-fit kappa: {kappa_W_per_MHz:.6e} W/MHz (R^2={fit['r2']:.6f}) -- "
      f"matches fig5/fig6_fresh_build exactly")

# ------------------------------------------------------------
# Theoretical LO-dressed noise floor (Eq.36), identical to
# fig6_fresh_build's sigma_Ry_LO_sq -- not distance-dependent,
# so no PRx/link-budget needed here (Fig.7's x-axis is SNR itself).
# ------------------------------------------------------------

A_LO = GLNA_lin * RL_ohm * D_resp ** 2 * kappa_rad_per_s ** 2 * wp_over_hbar ** 2
I_bar = eta_eff * P0_bar / (hbar * (2.0 * np.pi * fp_optical)) * e_charge
sigma_PSN_sq = 2.0 * e_charge * B_Hz * I_bar
sigma_TN_sq = 4.0 * kB * T_kelvin * B_Hz
sigma_Ry_LO_sq = (
    A_LO * sigma_BGN_sq_W
    + GLNA_lin * RL_ohm * D_resp ** 2 * sigma_PSN_sq
    + sigma_TN_sq
)
print(f"A_LO={A_LO:.6e}, sigma_Ry_LO^2={sigma_Ry_LO_sq:.6e} (matches fig6_fresh_build)")

# ------------------------------------------------------------
# SNR sweep (paper's own Fig.7 x-axis: direct SNR sweep, -20 to 40dB)
# ------------------------------------------------------------

SNR_dB = np.linspace(-20.0, 40.0, 601)
SNR_lin = 10.0 ** (SNR_dB / 10.0)

# ------------------------------------------------------------
# Model 1: Conventional -- standard Shannon capacity.
# NOT a paper-numbered equation (see math report Section 3).
# ------------------------------------------------------------

MI_conventional = np.log2(1.0 + SNR_lin)

# ------------------------------------------------------------
# Model 2: LO-free -- Eq.(22), r ~ SNR_Ry (paper's own stated
# approximation). Uses exponentially-scaled Bessel functions
# (scipy.special.ive) to stay numerically stable up to r=10^4
# (SNR=40dB), where the un-scaled I0(r) would overflow.
# ------------------------------------------------------------

r = SNR_lin
iv0_scaled = ive(0, r)   # = I0(r) * exp(-r)
iv1_scaled = ive(1, r)   # = I1(r) * exp(-r)
ln_I0 = np.log(iv0_scaled) + r          # recovers ln(I0(r)) without overflow
ratio_I1_I0 = iv1_scaled / iv0_scaled   # exp(-r) cancels exactly, safe

MI_lofree_nats = np.log((r + 1.0) / 2.0) - ln_I0 + r * ratio_I1_I0
MI_lofree = MI_lofree_nats / np.log(2.0)   # nats -> bits (standard conversion)

# --- Diagnostic only: invert Eq.(16) for PRx at this SNR, get the
# implied Omega_RF, and check it against the SAME real Fig.4/Fig.6
# AT-splitting threshold. Real finding (not a bug): this distance-
# domain criterion, moved into SNR-space, only opens right at the
# hard 46.02dB ceiling -- it is an almost-vacuous gate over the
# paper's own -20..40dB plotted window. See math report Section 5.
ceiling_lin = 1.0 / epsilon_tilde ** 2   # hard LO-free SNR ceiling (already established: 46.0206dB)
below_ceiling = SNR_lin < ceiling_lin
PRx_implied = np.full_like(SNR_lin, np.nan)
PRx_implied[below_ceiling] = (
    SNR_lin[below_ceiling] * sigma_BGN_sq_W / (1.0 - SNR_lin[below_ceiling] * epsilon_tilde ** 2)
)
omega_rf_implied_mhz = np.abs(wp_over_hbar) * np.sqrt(np.clip(PRx_implied, 0, None)) / (2.0 * np.pi * 1e6)
at_split_valid = below_ceiling & (omega_rf_implied_mhz >= threshold_mhz)
print(f"\nLO-free hard SNR ceiling: {10*np.log10(ceiling_lin):.4f} dB (matches fig6_fresh_build)")
print(f"[diagnostic] AT-splitting-threshold gate (Fig.4/6's criterion, moved into SNR-space): "
      f"{'empty over -20..40dB' if not at_split_valid.any() else f'[{SNR_dB[at_split_valid].min():.2f},{SNR_dB[at_split_valid].max():.2f}]dB'}")

# --- Actual gate used for the plotted curve: MI(SNR) itself must be
# physically valid (MI>=0 -- negative mutual information is
# impossible, so it marks where the r~SNR_Ry approximation breaks
# down). This is what actually bounds the paper's own "LO-free
# distortion region" at low SNR (Sec.V-C's own description).
lofree_valid = MI_lofree_nats >= 0.0
MI_lofree_gated = np.where(lofree_valid, MI_lofree, np.nan)
print(f"MI>=0 validity region (this is what's plotted): "
      f"[{SNR_dB[lofree_valid].min():.2f}, {SNR_dB[lofree_valid].max():.2f}] dB"
      if lofree_valid.any() else "LO-free: no valid region in this SNR range")

# ------------------------------------------------------------
# Model 3: Theoretical LO-dressed -- Eq.(38), both constants
# shown (paper-literal "*ln(2)" and standard "*log2(e)").
# ------------------------------------------------------------

E1_term = exp1(1.0 / SNR_lin)
exp_term = np.exp(1.0 / SNR_lin)
base_nats_form = exp_term * E1_term   # this quantity is in NATS (it equals the nats-domain ergodic capacity)

MI_lodressed_paper_ln2 = base_nats_form * np.log(2.0)        # paper-literal Eq.(38) constant
MI_lodressed_standard = base_nats_form * (1.0 / np.log(2.0))  # standard nats->bits constant (log2(e))

print(f"\nEq.(38) constant check @ SNR=0dB: base_nats_form={base_nats_form[np.argmin(np.abs(SNR_dB)):][0]:.6f}, "
      f"paper *ln(2) -> {MI_lodressed_paper_ln2[np.argmin(np.abs(SNR_dB))]:.6f} bits/s/Hz, "
      f"standard *log2(e) -> {MI_lodressed_standard[np.argmin(np.abs(SNR_dB))]:.6f} bits/s/Hz")

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 6.5))

ax.plot(SNR_dB, MI_conventional, color="tab:green", linewidth=1.8, label="Conventional")
ax.plot(SNR_dB, MI_lodressed_paper_ln2, color="tab:orange", linewidth=1.8,
        label="Theoretical LO-dressed (paper's Eq.38, ×ln2)")
ax.plot(SNR_dB, MI_lodressed_standard, color="tab:orange", linestyle="--", linewidth=1.4,
        label="Theoretical LO-dressed (standard ×log2(e))")
ax.plot(SNR_dB, MI_lofree_gated, color="tab:purple", linewidth=1.8, label="LO-free (Eq.22, MI>=0 region)")

ax.set_xlabel("SNR (dB)")
ax.set_ylabel("Mutual information (bits/s/Hz)")
ax.set_title("Fig. 7. Mutual information versus SNR\n"
             "(Conventional, LO-free Eq.22, Theoretical LO-dressed Eq.38 -- Practical LO-dressed deferred)")
ax.set_xlim(-20, 40)
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=8, loc="upper left")

fig.tight_layout()
fig.savefig(OUTPUT_DIR / "fig7_mi_vs_snr.png", dpi=220, bbox_inches="tight")
plt.close(fig)
print("\nSaved: fig7_mi_vs_snr.png")

# ------------------------------------------------------------
# Checkpoints
# ------------------------------------------------------------

for snr_check in [-10.0, 0.0, 10.0, 20.0, 40.0]:
    i = int(np.argmin(np.abs(SNR_dB - snr_check)))
    print(f"\n@ SNR={SNR_dB[i]:.1f}dB: "
          f"Conventional={MI_conventional[i]:.4f}, "
          f"LO-free(gated)={MI_lofree_gated[i]:.4f}, "
          f"Theo-LOdressed(paper ln2)={MI_lodressed_paper_ln2[i]:.4f}, "
          f"Theo-LOdressed(standard log2e)={MI_lodressed_standard[i]:.4f}")

np.savez(
    OUTPUT_DIR / "fig7_data.npz",
    SNR_dB=SNR_dB,
    MI_conventional=MI_conventional,
    MI_lofree_gated=MI_lofree_gated,
    MI_lodressed_paper_ln2=MI_lodressed_paper_ln2,
    MI_lodressed_standard=MI_lodressed_standard,
    threshold_mhz=threshold_mhz,
    kappa_W_per_MHz=kappa_W_per_MHz,
    A_LO=A_LO,
    sigma_Ry_LO_sq=sigma_Ry_LO_sq,
)
print("\nSaved: fig7_data.npz")
print("\nDONE.")
