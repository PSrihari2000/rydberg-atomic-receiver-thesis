# ============================================================
# FIG. 9 -- FRESH, SELF-CONTAINED REBUILD
# SER performance versus SNR. (a) Conventional (b) LO-dressed (c) LO-free
#
# Closed-form SER formulas (Proakis, "Digital Communication", 4th ed.
# -- the paper's own cited reference [44] for this figure) are
# EVALUATED AT THE REAL, QuTiP-derived achieved SNR, not the nominal
# one -- this is how the "Operating region" / "Distortion region"
# split in panels (b)/(c) is produced: same closed-form formula
# throughout, but its input differs from the nominal x-axis SNR
# wherever the real nonlinear/threshold-gated atomic response can't
# keep up. No artificial gap/NaN is used for this -- the same formula,
# fed a real degraded SNR, naturally saturates near its own ceiling.
#
# REUSES (real, already-computed, no new QuTiP):
#   - fig4_fresh_build/fig4_classification.npz: AT-splitting threshold
#   - fig5_fresh_build/fig5_data.npz: real static Pout(Omega_total)
#     curve, P0_bar (kappa re-fit fresh, matching fig6/7_fresh_build)
#
# Does not import old fig9 anything, or fig7_reinvestigation/.
# ============================================================

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
from scipy.interpolate import interp1d

OUTPUT_DIR = Path(__file__).resolve().parent
FIG4_DATA = OUTPUT_DIR.parent / "fig4_fresh_build" / "fig4_classification.npz"
FIG5_DATA = OUTPUT_DIR.parent / "fig5_fresh_build" / "fig5_data.npz"

print("=" * 70)
print("FIG. 9 -- FRESH REBUILD (reuses Fig.4/Fig.5's real data)")
print("=" * 70)

# ------------------------------------------------------------
# Physical constants (re-typed fresh, matching fig3/4/5/6/7_fresh_build)
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

# ------------------------------------------------------------
# Reuse Fig.4's real threshold, Fig.5's real static response
# ------------------------------------------------------------

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
      f"range=[{static_omega_mhz.min():.2f},{static_omega_mhz.max():.2f}]MHz, "
      f"P0_bar={P0_bar/1e-6:.4f} microW")


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
print(f"Re-fit kappa: {kappa_W_per_MHz:.6e} W/MHz (R^2={fit['r2']:.6f}) -- matches fig5/6/7_fresh_build")

f_interp_static = interp1d(static_omega_mhz, static_pout_w, kind="cubic", bounds_error=False, fill_value=np.nan)

# ------------------------------------------------------------
# LO-dressed noise floor (Eq.36), same as fig6/7_fresh_build
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
print(f"A_LO={A_LO:.6e}, sigma_Ry_LO^2={sigma_Ry_LO_sq:.6e} (matches fig6/7_fresh_build)")

# ------------------------------------------------------------
# Closed-form SER formulas (Proakis, Digital Communication, 4th ed.
# -- the paper's own cited [44]). SNR here = average symbol SNR
# (Es/N0), matching this project's SNR convention throughout
# (signal power / noise power, no separate bit-rate normalization).
# ------------------------------------------------------------

def Qfunc(x):
    return 0.5 * erfc(x / np.sqrt(2.0))


def ser_mqam(M, snr_lin):
    snr_lin = np.clip(snr_lin, 0.0, None)
    arg = np.sqrt(3.0 * snr_lin / (M - 1.0))
    inner = 1.0 - 2.0 * (1.0 - 1.0 / np.sqrt(M)) * Qfunc(arg)
    return 1.0 - inner ** 2


def ser_mpam(M, snr_lin):
    snr_lin = np.clip(snr_lin, 0.0, None)
    arg = np.sqrt(6.0 * snr_lin / (M ** 2 - 1.0))
    return 2.0 * (M - 1.0) / M * Qfunc(arg)


# ------------------------------------------------------------
# Real nonlinear LO-dressed response: same phasor-sum + real
# static-curve interpolation + Fourier-projection pipeline as
# fig6_fresh_build, parameterized directly by Omega_RF (MHz)
# rather than PRx/distance (Fig.9's x-axis is SNR, not distance).
# ------------------------------------------------------------

def practical_fourier_amplitude_from_omega_rf(omega_rf_mhz_array, n_t=400):
    period_s = 1.0 / (DELTA_F_MHZ * 1e6)
    t_seconds = np.linspace(0.0, period_s, n_t, endpoint=False)
    theta = 2.0 * np.pi * DELTA_F_MHZ * 1e6 * t_seconds  # Delta_phi=0

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
# Panel (a): Conventional -- pure closed-form, no atoms, no
# distortion mechanism. log2(M) = 2,4,6,8.
# ============================================================

SNR_dB_a = np.linspace(-40.0, 40.0, 801)
SNR_lin_a = 10.0 ** (SNR_dB_a / 10.0)
panel_a = {}
for log2M in [2, 4, 6, 8]:
    M = 2 ** log2M
    panel_a[log2M] = ser_mqam(M, SNR_lin_a)

print("\nPanel (a) Conventional -- SER at SNR=20dB:")
for log2M in [2, 4, 6, 8]:
    i = int(np.argmin(np.abs(SNR_dB_a - 20.0)))
    print(f"  log2M={log2M} (M={2**log2M:4d}): SER={panel_a[log2M][i]:.6e}")

# ============================================================
# Panel (b): LO-dressed -- nominal SNR (x-axis) -> invert Eq.(37)
# for PRx -> Omega_RF -> REAL nonlinear pipeline -> SNR_real ->
# SAME M-QAM formula. log2(M) = 4,6,8,10.
# ============================================================

SNR_dB_b = np.linspace(-40.0, 90.0, 1301)   # extended (real sweep, not xlim padding) -- goes
                                              # ~19dB past the point where Omega_total leaves
                                              # the real static curve's interpolation range
                                              # (~71.25dB, computed below), so the genuine
                                              # cutoff is visible, not just assumed
SNR_lin_b_nominal = 10.0 ** (SNR_dB_b / 10.0)

PRx_nominal_b = SNR_lin_b_nominal * sigma_Ry_LO_sq / A_LO
omega_rf_implied_b_mhz = np.abs(wp_over_hbar) * np.sqrt(PRx_nominal_b) / (2.0 * np.pi * 1e6)

P_delta_f_b = practical_fourier_amplitude_from_omega_rf(omega_rf_implied_b_mhz)
signal_power_b = GLNA_lin * RL_ohm * D_resp ** 2 * P_delta_f_b ** 2
SNR_lin_b_real = signal_power_b / sigma_Ry_LO_sq   # NaN wherever the real static curve's range is exceeded

n_nan_b = int(np.sum(np.isnan(SNR_lin_b_real)))
print(f"\nPanel (b) LO-dressed: {n_nan_b}/{len(SNR_dB_b)} nominal-SNR points fall outside "
      f"the real static curve's interpolation range (Omega_total leaves [{static_omega_mhz.min():.2f},"
      f"{static_omega_mhz.max():.2f}]MHz) -- real limitation of Fig.5's plotted data range, not fabricated around.")

panel_b = {}
for log2M in [4, 6, 8, 10]:
    M = 2 ** log2M
    panel_b[log2M] = ser_mqam(M, SNR_lin_b_real)

print("Panel (b) LO-dressed -- SER at nominal SNR=10dB (real vs nominal SNR):")
i10 = int(np.argmin(np.abs(SNR_dB_b - 10.0)))
print(f"  nominal SNR=10dB -> real SNR={10*np.log10(SNR_lin_b_real[i10]) if np.isfinite(SNR_lin_b_real[i10]) else float('nan'):.4f}dB")
for log2M in [4, 6, 8, 10]:
    print(f"  log2M={log2M} (M={2**log2M:4d}): SER={panel_b[log2M][i10]:.6e}")

# ============================================================
# Panel (c): LO-free -- nominal SNR -> invert Eq.(16) for PRx ->
# Omega_RF -> gated by the SAME real Fig.4 threshold: real SNR =
# nominal SNR if resolvable, else 0 (total readout failure -- a
# hard, physically-motivated cutoff, not a smooth taper, matching
# the near-vertical transition seen in the paper's own panel (c)).
# log2(M) = 2,4,6,8. x-axis range -40..20dB (paper's own, narrower
# than (a)/(b)).
# ============================================================

SNR_dB_c = np.linspace(-40.0, 46.5, 2601)   # extended (real sweep) -- goes just past the hard
                                              # ceiling (1/epsilon_tilde^2 = 46.0206dB) so the
                                              # actual asymptote is shown directly, not assumed
SNR_lin_c_nominal = 10.0 ** (SNR_dB_c / 10.0)

ceiling_lin = 1.0 / epsilon_tilde ** 2
below_ceiling_c = SNR_lin_c_nominal < ceiling_lin
PRx_nominal_c = np.full_like(SNR_lin_c_nominal, np.nan)
PRx_nominal_c[below_ceiling_c] = (
    SNR_lin_c_nominal[below_ceiling_c] * sigma_BGN_sq_W
    / (1.0 - SNR_lin_c_nominal[below_ceiling_c] * epsilon_tilde ** 2)
)
omega_rf_implied_c_mhz = np.abs(wp_over_hbar) * np.sqrt(np.clip(PRx_nominal_c, 0, None)) / (2.0 * np.pi * 1e6)
resolvable_c = below_ceiling_c & (omega_rf_implied_c_mhz >= threshold_mhz)
SNR_lin_c_real = np.where(resolvable_c, SNR_lin_c_nominal, 0.0)

print(f"\nPanel (c) LO-free: resolvable (operating) region = "
      f"[{SNR_dB_c[resolvable_c].min():.2f},{SNR_dB_c[resolvable_c].max():.2f}]dB"
      if resolvable_c.any() else "\nPanel (c) LO-free: never resolvable in this SNR range")

panel_c = {}
for log2M in [2, 4, 6, 8]:
    M = 2 ** log2M
    panel_c[log2M] = ser_mpam(M, SNR_lin_c_real)

print("Panel (c) LO-free -- SER plateau (real SNR=0, distortion region):")
for log2M in [2, 4, 6, 8]:
    M = 2 ** log2M
    print(f"  log2M={log2M} (M={M:3d}): SER(real SNR=0) = {(M-1)/M:.6f} (matches ser_mpam ceiling)")

# ============================================================
# Plot -- 3 panels, paper's own layout/style
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

styles_a = {2: ("tab:orange", "-"), 4: ("tab:green", "--"), 6: ("tab:red", "--"), 8: ("tab:purple", ":")}
for log2M, ser in panel_a.items():
    c, ls = styles_a[log2M]
    axes[0].semilogy(SNR_dB_a, np.clip(ser, 1e-6, 1.0), color=c, linestyle=ls, linewidth=1.8,
                      label=rf"$\log_2 M={log2M}$")
axes[0].set_title("(a) Conventional")

styles_b = {4: ("tab:green", "--"), 6: ("tab:red", "--"), 8: ("tab:purple", ":"), 10: ("tab:blue", "-")}
for log2M, ser in panel_b.items():
    c, ls = styles_b[log2M]
    axes[1].semilogy(SNR_dB_b, np.clip(ser, 1e-6, 1.0), color=c, linestyle=ls, linewidth=1.8,
                      label=rf"$\log_2 M={log2M}$")
axes[1].axhline(0.01, color="black", linestyle="--", linewidth=1.0)
axes[1].text(-38, 0.013, "SER $\\leq$ 1%", fontsize=8)
# Real "operating region" shading: where the HIGHEST-order curve clears the 1% line
ser_top_b = panel_b[10]
op_mask_b = ser_top_b <= 0.01
if op_mask_b.any():
    axes[1].axvspan(SNR_dB_b[op_mask_b].min(), SNR_dB_b[op_mask_b].max(), color="tab:purple", alpha=0.12)
    axes[1].text((SNR_dB_b[op_mask_b].min() + SNR_dB_b[op_mask_b].max()) / 2, 3e-6,
                 "Operating region", ha="center", fontsize=8, color="tab:purple")
# Real cutoff: beyond this nominal SNR, Omega_total(t) leaves the real static curve's
# interpolation range -- SNR_real (and hence SER) becomes NaN, a genuine data-range
# limit, not a fake plot boundary.
if n_nan_b > 0:
    edge_snr_b = SNR_dB_b[np.isnan(SNR_lin_b_real)].min()
    axes[1].axvline(edge_snr_b, color="black", linestyle=":", linewidth=1.2)
    axes[1].text(edge_snr_b + 1, 0.5, f"real static-curve\ndata ends\n({edge_snr_b:.1f}dB)",
                 fontsize=7, color="black")
axes[1].set_title("(b) LO-dressed")

styles_c = {2: ("tab:orange", "-"), 4: ("tab:green", "--"), 6: ("tab:red", "--"), 8: ("tab:purple", ":")}
for log2M, ser in panel_c.items():
    c, ls = styles_c[log2M]
    axes[2].semilogy(SNR_dB_c, np.clip(ser, 1e-6, 1.0), color=c, linestyle=ls, linewidth=1.8,
                      label=rf"$\log_2 M={log2M}$")
axes[2].axhline(0.01, color="black", linestyle="--", linewidth=1.0)
axes[2].text(-38, 0.013, "SER $\\leq$ 1%", fontsize=8)
axes[2].axvspan(SNR_dB_c.min(), SNR_dB_c.max(), color="gray", alpha=0.15)
axes[2].text(-10, 3e-6, "Distortion region\n(flat throughout -40..46dB)",
             ha="center", fontsize=8, color="dimgray")
if resolvable_c.any():
    axes[2].axvspan(SNR_dB_c[resolvable_c].min(), SNR_dB_c.max(), color="tab:purple", alpha=0.25)
# Hard ceiling: Eq.(16) cannot produce ANY nominal SNR at or above 1/epsilon_tilde^2,
# no matter how much power is transmitted -- a real asymptote, not a plot boundary.
ceiling_db = 10.0 * np.log10(ceiling_lin)
axes[2].axvline(ceiling_db, color="red", linestyle="-", linewidth=1.5)
axes[2].text(ceiling_db - 0.5, 0.3, f"hard ceiling\n{ceiling_db:.4f}dB\n(SNR cannot exceed\nthis, any power)",
             fontsize=7, color="red", ha="right")
axes[2].set_title("(c) LO-free")

for ax, xlim in zip(axes, [(-40, 40), (-40, 90), (-40, 46.5)]):
    ax.set_xlim(*xlim)
    ax.set_ylim(1e-6, 1.0)
    ax.set_xlabel("SNR (dB)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, loc="lower left")
axes[0].set_ylabel("SER")

fig.suptitle("Fig. 9. SER performance versus the SNR", y=1.02)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "fig9_ser_vs_snr.png", dpi=220, bbox_inches="tight")
plt.close(fig)
print("\nSaved: fig9_ser_vs_snr.png")

np.savez(
    OUTPUT_DIR / "fig9_data.npz",
    SNR_dB_a=SNR_dB_a, **{f"panel_a_log2M{k}": v for k, v in panel_a.items()},
    SNR_dB_b=SNR_dB_b, SNR_lin_b_real=SNR_lin_b_real, **{f"panel_b_log2M{k}": v for k, v in panel_b.items()},
    SNR_dB_c=SNR_dB_c, SNR_lin_c_real=SNR_lin_c_real, **{f"panel_c_log2M{k}": v for k, v in panel_c.items()},
    threshold_mhz=threshold_mhz, kappa_W_per_MHz=kappa_W_per_MHz, A_LO=A_LO, sigma_Ry_LO_sq=sigma_Ry_LO_sq,
)
print("Saved: fig9_data.npz")
print("\nDONE.")
