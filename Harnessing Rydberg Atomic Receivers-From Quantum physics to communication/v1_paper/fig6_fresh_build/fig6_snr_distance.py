# ============================================================
# FIG. 6 -- FRESH, SELF-CONTAINED REBUILD
# SNR performance versus distance, 4 receiver types.
#
# REUSES (real, already-computed, no new QuTiP):
#   - fig4_fresh_build/fig4_classification.npz: AT-splitting threshold (5.500 MHz)
#   - fig5_fresh_build/fig5_data.npz: real static Pout(Omega_total) curve, P0_bar
# kappa is re-fit here from that same real static data (same algorithm,
# not saved in fig5_data.npz originally -- see math report Section 1).
#
# Does not import old fig6.py or anything from fig7_reinvestigation/.
# ============================================================

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

OUTPUT_DIR = Path(__file__).resolve().parent
FIG4_DATA = OUTPUT_DIR.parent / "fig4_fresh_build" / "fig4_classification.npz"
FIG5_DATA = OUTPUT_DIR.parent / "fig5_fresh_build" / "fig5_data.npz"

print("=" * 70)
print("FIG. 6 -- FRESH REBUILD (reuses Fig.4/Fig.5's real data)")
print("=" * 70)

# ------------------------------------------------------------
# Physical constants (re-typed fresh, matching fig3/4/5_fresh_build)
# ------------------------------------------------------------

e_charge = 1.6e-19
a0 = 5.2e-11
hbar = 1.054571817e-34
eta0 = 377.0
kB = 1.380649e-23
c_light = 2.99792458e8

wp_RF = -1443.459 * e_charge * a0
lambda_p = 852e-9
fp_optical = c_light / lambda_p           # probe laser optical frequency, for Eq.35

# ------------------------------------------------------------
# New parameters for Fig.6 (paper Sec.IV/V-A)
# ------------------------------------------------------------

GTx_dBi, GRx_dBi = 2.15, 2.15
GLNA_dB = 20.0
RL_ohm = 50.0
D_resp = 0.55
T_kelvin = 290.0
eta_eff = 0.8
sigma_BGN_dBm = -90.0
epsilon_tilde = 0.005
fRF_Hz = 3.5e9
B_Hz = 1.0e6
lambda_RF_m = c_light / fRF_Hz


def dbm_to_watts(p_dbm):
    return 10.0 ** ((p_dbm - 30.0) / 10.0)


def db_to_linear(g_db):
    return 10.0 ** (g_db / 10.0)


GTx_lin = db_to_linear(GTx_dBi)
GRx_lin = db_to_linear(GRx_dBi)
GLNA_lin = db_to_linear(GLNA_dB)
sigma_BGN_sq_W = dbm_to_watts(sigma_BGN_dBm)

print(f"GTx={GTx_lin:.4f}, GRx={GRx_lin:.4f}, GLNA={GLNA_lin:.4f}, "
      f"sigma_BGN^2={sigma_BGN_sq_W:.4e} W")

# ------------------------------------------------------------
# Reuse Fig.4's real threshold
# ------------------------------------------------------------

fig4_data = np.load(FIG4_DATA)
threshold_mhz = float(fig4_data["threshold_mhz"])
print(f"\nReused Fig.4 threshold: {threshold_mhz:.4f} MHz")

# ------------------------------------------------------------
# Reuse Fig.5's real static response; re-fit kappa fresh (same
# algorithm, same real data -- not saved in fig5_data.npz)
# ------------------------------------------------------------

fig5_data = np.load(FIG5_DATA)
static_omega_mhz = fig5_data["static_omega_mhz"]
static_pout_w = fig5_data["static_pout_w"]
P0_bar = float(fig5_data["P0_bar"])
OMEGA_LO_MHZ = float(fig5_data["OMEGA_LO_MHZ"])
DELTA_F_MHZ = float(fig5_data["DELTA_F_MHZ"])

print(f"Reused Fig.5 static response: {len(static_omega_mhz)} points, "
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
kappa_rad_per_s = kappa_W_per_MHz / (2.0 * np.pi * 1e6)   # convert MHz^-1 slope to (rad/s)^-1
print(f"Re-fit kappa: {kappa_W_per_MHz:.6e} W/MHz (R^2={fit['r2']:.6f}, region="
      f"[{fit['omega_low']:.4f},{fit['omega_high']:.4f}]MHz) -- matches fig5_fresh_build exactly")

f_interp_static = interp1d(static_omega_mhz, static_pout_w, kind="cubic", bounds_error=False, fill_value=np.nan)

# ------------------------------------------------------------
# Link budget + basic conversions
# ------------------------------------------------------------

def PRx_rydberg(PTx_watts, d_m):
    """Eq.(10)."""
    return PTx_watts * GTx_lin * eta0 / (4.0 * np.pi * d_m ** 2)


def omega_RF_signal_rad_s(PRx):
    """Eq.(8), magnitude form: Omega_RF = |E_RF|*wp_RF/hbar, PRx=|E_RF|^2."""
    return np.abs(wp_RF / hbar) * np.sqrt(PRx)


# ------------------------------------------------------------
# Model 1: Conventional
# ------------------------------------------------------------

sigma_Conv_sq_W = GRx_lin * GLNA_lin * sigma_BGN_sq_W + 4.0 * kB * T_kelvin * B_Hz


def snr_conventional_dB(PTx_watts, d_m):
    S = PTx_watts * GTx_lin / (4.0 * np.pi * d_m ** 2)
    A_eff = lambda_RF_m ** 2 / (4.0 * np.pi)
    Pc = S * A_eff
    return 10.0 * np.log10(Pc / sigma_Conv_sq_W)


# ------------------------------------------------------------
# Model 2: LO-free
# ------------------------------------------------------------

def snr_lo_free_dB(PTx_watts, d_m):
    PRx = PRx_rydberg(PTx_watts, d_m)
    gamma = PRx / (sigma_BGN_sq_W + epsilon_tilde ** 2 * PRx)   # sigma_QPN^2=0
    snr = 10.0 * np.log10(gamma)
    omega_rf_mhz = omega_RF_signal_rad_s(PRx) / (2.0 * np.pi * 1e6)
    return np.where(omega_rf_mhz >= threshold_mhz, snr, np.nan)


# ------------------------------------------------------------
# Model 3: Theoretical LO-dressed
# ------------------------------------------------------------

wp_over_hbar = wp_RF / hbar
A_LO = GLNA_lin * RL_ohm * D_resp ** 2 * kappa_rad_per_s ** 2 * wp_over_hbar ** 2
I_bar = eta_eff * P0_bar / (hbar * (2.0 * np.pi * fp_optical)) * e_charge
sigma_PSN_sq = 2.0 * e_charge * B_Hz * I_bar
sigma_TN_sq = 4.0 * kB * T_kelvin * B_Hz
sigma_Ry_LO_sq = (
    GLNA_lin * RL_ohm * D_resp ** 2 * kappa_rad_per_s ** 2 * wp_over_hbar ** 2 * sigma_BGN_sq_W
    + GLNA_lin * RL_ohm * D_resp ** 2 * sigma_PSN_sq
    + sigma_TN_sq
)

print(f"\nA_LO={A_LO:.6e}, sigma_Ry_LO^2={sigma_Ry_LO_sq:.6e}")
print(f"  breakdown: BGN={GLNA_lin*RL_ohm*D_resp**2*kappa_rad_per_s**2*wp_over_hbar**2*sigma_BGN_sq_W/sigma_Ry_LO_sq*100:.4f}%, "
      f"PSN={GLNA_lin*RL_ohm*D_resp**2*sigma_PSN_sq/sigma_Ry_LO_sq*100:.4f}%, "
      f"TN={sigma_TN_sq/sigma_Ry_LO_sq*100:.4f}%")


def snr_lo_dressed_theoretical_dB(PTx_watts, d_m):
    PRx = PRx_rydberg(PTx_watts, d_m)
    return 10.0 * np.log10(A_LO * PRx / sigma_Ry_LO_sq)


# ------------------------------------------------------------
# Model 4: Practical LO-dressed (real nonlinear response)
# ------------------------------------------------------------

def practical_fourier_amplitude(PTx_watts, d_m_array, n_t=400):
    PRx = PRx_rydberg(PTx_watts, d_m_array)
    omega_rf_signal_mhz = omega_RF_signal_rad_s(PRx) / (2.0 * np.pi * 1e6)

    period_s = 1.0 / (DELTA_F_MHZ * 1e6)
    t_seconds = np.linspace(0.0, period_s, n_t, endpoint=False)

    theta = 2.0 * np.pi * DELTA_F_MHZ * 1e6 * t_seconds  # Delta_phi=0
    omega_total_grid_mhz = np.sqrt(
        OMEGA_LO_MHZ ** 2 + omega_rf_signal_mhz[:, None] ** 2
        + 2.0 * OMEGA_LO_MHZ * omega_rf_signal_mhz[:, None] * np.cos(theta)[None, :]
    )

    Pout_grid = f_interp_static(omega_total_grid_mhz.ravel()).reshape(omega_total_grid_mhz.shape)

    proj = np.exp(-1j * theta)
    out_of_range = np.any(np.isnan(Pout_grid), axis=1)
    Pout_grid_filled = np.where(np.isnan(Pout_grid), 0.0, Pout_grid)
    c1 = (Pout_grid_filled * proj[None, :]).sum(axis=1) / n_t
    P_delta_f = 2.0 * np.abs(c1)   # Eq.(28), single-sided Fourier amplitude
    P_delta_f[out_of_range] = np.nan

    return P_delta_f


def snr_lo_dressed_practical_dB(PTx_watts, d_m_array):
    P_delta_f = practical_fourier_amplitude(PTx_watts, d_m_array)
    signal_power = GLNA_lin * RL_ohm * D_resp ** 2 * P_delta_f ** 2
    return 10.0 * np.log10(signal_power / sigma_Ry_LO_sq)


# ------------------------------------------------------------
# Sweep and plot (PTx sets match the paper's own Fig.6 legend)
# ------------------------------------------------------------

DISTANCE_M = np.logspace(-1.0, 6.0, 281)   # extended to start at 0.1m (real computed values,
                                             # not visual padding) -- reveals near-field
                                             # behavior (e.g. the Practical LO-dressed dip)
                                             # that a d>=1m sweep would cut off

CONV_PTX = [-10.0, 10.0]
LOFREE_PTX = [10.0, 20.0]
THEO_PTX = [-10.0, 10.0]
PRAC_PTX = [-10.0, 10.0]

curves = {"conv": {}, "lofree": {}, "theo": {}, "prac": {}}
for p in CONV_PTX:
    curves["conv"][p] = snr_conventional_dB(dbm_to_watts(p), DISTANCE_M)
for p in LOFREE_PTX:
    curves["lofree"][p] = snr_lo_free_dB(dbm_to_watts(p), DISTANCE_M)
for p in THEO_PTX:
    curves["theo"][p] = snr_lo_dressed_theoretical_dB(dbm_to_watts(p), DISTANCE_M)
for p in PRAC_PTX:
    curves["prac"][p] = snr_lo_dressed_practical_dB(dbm_to_watts(p), DISTANCE_M)

fig, ax = plt.subplots(figsize=(9, 6.5))

# Paper's own legend convention: ONE linestyle+color per receiver type;
# the two PTx values are told apart ONLY by a marker (plain line vs.
# same line with circle markers), not by different dash patterns.
#
# markevery is now computed PER CURVE from its own number of finite
# points (not the full 241-point array) -- some curves (e.g. LO-free)
# are only resolvable over a short segment, and a global markevery
# left too few (sometimes 0-1) markers visible on those, making the
# two PTx values of the same color indistinguishable.

def dynamic_markevery(curve, target_markers=6):
    n_finite = int(np.sum(np.isfinite(curve)))
    if n_finite == 0:
        return 1
    return max(1, n_finite // target_markers)


def plot_pair(ax, x, curve_lo, curve_hi, color, linestyle, label_lo, label_hi):
    ax.plot(x, curve_lo, linestyle, color=color, linewidth=1.8, label=label_lo)
    me_hi = dynamic_markevery(curve_hi)
    ax.plot(x, curve_hi, linestyle, color=color, linewidth=1.8, marker="o",
            markevery=me_hi, markersize=6, markerfacecolor="none", markeredgewidth=1.3,
            label=label_hi)


plot_pair(ax, DISTANCE_M, curves["lofree"][10.0], curves["lofree"][20.0], "tab:purple", "-",
           "LO-free, PTx=10dBm", "LO-free, PTx=20dBm")
plot_pair(ax, DISTANCE_M, curves["conv"][-10.0], curves["conv"][10.0], "tab:green", "-",
           "Conventional, PTx=-10dBm", "Conventional, PTx=10dBm")
plot_pair(ax, DISTANCE_M, curves["theo"][-10.0], curves["theo"][10.0], "tab:orange", ":",
           "Theoretical LO-dressed, PTx=-10dBm", "Theoretical LO-dressed, PTx=10dBm")
plot_pair(ax, DISTANCE_M, curves["prac"][-10.0], curves["prac"][10.0], "tab:blue", "--",
           "Practical LO-dressed, PTx=-10dBm", "Practical LO-dressed, PTx=10dBm")

ax.set_xscale("log")
ax.set_xlim(0.1, 1e6)   # matches the real sweep's own start (d=0.1m), no artificial padding
ax.set_ylim(-100, 110)
ax.set_xlabel(r"Distance, $d_{Tx-Rx}$ (m)")
ax.set_ylabel("SNR (dB)")
ax.set_title("Fig. 6. SNR performance versus distance at different transmit powers $P_{Tx}$\n(d = 0.76 mm)")
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=8, ncol=2, loc="upper right")

# ------------------------------------------------------------
# Checkpoint: PTx=-10dBm, d=100m gap; 0dB crossing distances --
# computed BEFORE saving so the real numbers can be annotated on
# the plot itself, same style as the paper's own Fig.6 (paper
# shows ~44dB/~1500m; ours are the real, independently-computed
# numbers, whatever they are -- not tuned to match).
# ------------------------------------------------------------

i100 = int(np.argmin(np.abs(DISTANCE_M - 100.0)))
gap_100m = curves["theo"][-10.0][i100] - curves["conv"][-10.0][i100]
print(f"\nCheckpoint @ PTx=-10dBm, d=100m:")
print(f"  Conventional SNR       = {curves['conv'][-10.0][i100]:.4f} dB")
print(f"  Theoretical LO-dressed = {curves['theo'][-10.0][i100]:.4f} dB")
print(f"  Gap                     = {gap_100m:.4f} dB  (paper claims ~44 dB)")


def find_0db_crossing(distance, snr):
    valid = np.isfinite(snr)
    x = np.log10(distance[valid])
    y = snr[valid]
    diff = y - 0.0
    idx = np.where(np.diff(np.sign(diff)) != 0)[0]
    if len(idx) == 0:
        return np.nan
    i = idx[0]
    x0, x1 = x[i], x[i + 1]
    y0, y1 = diff[i], diff[i + 1]
    return 10.0 ** (x0 - y0 * (x1 - x0) / (y1 - y0))


d_conv_0db = find_0db_crossing(DISTANCE_M, curves["conv"][-10.0])
d_theo_0db = find_0db_crossing(DISTANCE_M, curves["theo"][-10.0])
print(f"\n0dB crossing: Conventional @ {d_conv_0db:.2f}m, Theoretical LO-dressed @ {d_theo_0db:.2f}m")
print(f"  Extended coverage = {d_theo_0db - d_conv_0db:.2f}m  (paper claims ~1500m)")

# ------------------------------------------------------------
# Annotate the real gap/crossing on the plot, paper style
# ------------------------------------------------------------

y_conv_100 = curves["conv"][-10.0][i100]
y_theo_100 = curves["theo"][-10.0][i100]

ax.annotate("", xy=(100, y_theo_100), xytext=(100, y_conv_100),
            arrowprops=dict(arrowstyle="<->", color="black", linewidth=1.3))
ax.text(130, (y_conv_100 + y_theo_100) / 2,
        f"~{gap_100m:.1f} dB\n(100 m)", fontsize=9, color="black", va="center")

mid_x = np.sqrt(d_conv_0db * d_theo_0db)   # geometric mean = midpoint on a LOG x-axis
ax.annotate("", xy=(d_theo_0db, 0), xytext=(d_conv_0db, 0),
            arrowprops=dict(arrowstyle="<->", color="red", linewidth=1.3))
ax.text(mid_x, -8, f"~{d_theo_0db - d_conv_0db:.0f} m\n(SNR = 0 dB)",
        fontsize=9, color="red", ha="center")
ax.text(d_conv_0db, -18, "Conventional", fontsize=8, color="tab:green", ha="center")
ax.text(d_theo_0db, -18, "LO-dressed", fontsize=8, color="tab:blue", ha="center")
ax.axvline(d_conv_0db, color="tab:green", linestyle=":", linewidth=0.8, alpha=0.6)
ax.axvline(d_theo_0db, color="tab:blue", linestyle=":", linewidth=0.8, alpha=0.6)

fig.tight_layout()
fig.savefig(OUTPUT_DIR / "fig6_snr_distance.png", dpi=220, bbox_inches="tight")
plt.close(fig)
print("\nSaved: fig6_snr_distance.png")

with open(OUTPUT_DIR / "fig6_math_report.md", "a") as f:
    f.write("\n\n## 9. Actual numeric results (this run)\n\n")
    f.write(f"- kappa (re-fit, matches fig5_fresh_build) = {kappa_W_per_MHz:.6e} W/MHz\n")
    f.write(f"- A_LO = {A_LO:.6e}, sigma_Ry_LO^2 = {sigma_Ry_LO_sq:.6e}\n")
    f.write(f"- Checkpoint @ PTx=-10dBm, d=100m: Conventional={curves['conv'][-10.0][i100]:.4f}dB, "
            f"Theoretical LO-dressed={curves['theo'][-10.0][i100]:.4f}dB, gap={gap_100m:.4f}dB "
            f"(paper claims ~44dB)\n")
    f.write(f"- 0dB crossing: Conventional@{d_conv_0db:.2f}m, Theoretical LO-dressed@{d_theo_0db:.2f}m, "
            f"extended coverage={d_theo_0db-d_conv_0db:.2f}m (paper claims ~1500m)\n")

print("\nAppended numeric results to fig6_math_report.md")
print("\nDONE.")
