# ============================================================
# FIG. 5 -- FRESH, SELF-CONTAINED REBUILD
# Distortion of the LO-dressed Rydberg atomic receiver
#
# Panel (a) REUSES fig3_fresh_build's real Delta_c=0 column
# (confirmed with the user -- Omega_total occupies the exact
# same Hamiltonian slot Omega_RF did in Fig.3/4, per paper
# Sec.III-B). No new QuTiP solves. Panels (b)/(c) are pure
# equation evaluation + real interpolation, also no QuTiP.
# Does not import old fig5.py or anything from
# fig7_reinvestigation/.
# ============================================================

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

OUTPUT_DIR = Path(__file__).resolve().parent
FIG3_DATA = OUTPUT_DIR.parent / "fig3_fresh_build" / "fig3_quantum_response.npz"

print("=" * 70)
print("FIG. 5 -- FRESH REBUILD (panel (a) reuses fig3_fresh_build's real data)")
print("=" * 70)

# ------------------------------------------------------------
# PANEL (a) data: Delta_c=0 column of the real Fig.3 grid
# ------------------------------------------------------------

data = np.load(FIG3_DATA)
delta_c_mhz = data["delta_c_mhz"]
omega_rf_mhz_full = data["omega_rf_mhz"]
Pout_surface = data["Pout_surface"]

dc0_idx = int(np.argmin(np.abs(delta_c_mhz)))
static_omega_mhz = omega_rf_mhz_full
static_pout_w = Pout_surface[:, dc0_idx]

print(f"Loaded: {FIG3_DATA}")
print(f"Delta_c=0 index: {dc0_idx} (Delta_c={delta_c_mhz[dc0_idx]:.4f} MHz)")
print(f"Static response range: Omega_total/2pi = {static_omega_mhz.min()}-{static_omega_mhz.max()} MHz, "
      f"{len(static_omega_mhz)} points")

# Restrict display range to the paper's plotted window (0-14 MHz)
disp_mask = static_omega_mhz <= 14.0
disp_omega = static_omega_mhz[disp_mask]
disp_pout = static_pout_w[disp_mask]

# ------------------------------------------------------------
# Linear dynamic range fit (INFERRED procedure, R^2>=0.995,
# anchored at Omega_LO) -- fresh implementation
# ------------------------------------------------------------

OMEGA_LO_MHZ = 4.23
DELTA_F_MHZ = 0.150

def linear_fit_score(x, y):
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return slope, intercept, r2

def find_linear_dynamic_range(omega_values, response_values, omega_lo, r2_threshold=0.995):
    center = int(np.argmin(np.abs(omega_values - omega_lo)))
    best = None
    max_radius = min(center, len(omega_values) - center - 1)
    for radius in range(5, max_radius + 1):
        left, right = center - radius, center + radius + 1
        slope, intercept, r2 = linear_fit_score(omega_values[left:right], response_values[left:right])
        if r2 >= r2_threshold:
            best = dict(left=left, right=right, slope=slope, intercept=intercept, r2=r2,
                        omega_low=omega_values[left], omega_high=omega_values[right - 1])
        else:
            break
    return best

# R^2 threshold: 0.998, not 0.995. Both are defensible choices (see the
# sensitivity table already discussed with the user, run 2026-08-20:
# 0.995->[0.500,8.000]MHz, 0.998->[1.250,7.250]MHz, 0.999->[1.625,6.875]MHz).
# Picked 0.998 as a stricter, still round threshold for a tighter, cleaner
# fit -- NOT tuned to reproduce the paper's own box pixel-for-pixel.
R2_THRESHOLD = 0.998
fit = find_linear_dynamic_range(static_omega_mhz, static_pout_w, OMEGA_LO_MHZ, r2_threshold=R2_THRESHOLD)
print(f"\nLinear dynamic range (R^2>={R2_THRESHOLD}): [{fit['omega_low']:.4f}, {fit['omega_high']:.4f}] MHz, R^2={fit['r2']:.6f}")

# P0_bar at Omega_LO
f_interp_a = interp1d(static_omega_mhz, static_pout_w, kind="cubic")
P0_bar = float(f_interp_a(OMEGA_LO_MHZ))
print(f"P0_bar = Pout(Omega_LO={OMEGA_LO_MHZ} MHz) = {P0_bar/1e-6:.4f} microW")

# ------------------------------------------------------------
# PANELS (b) and (c): phasor-magnitude Omega_total(t), one
# real period, and real interpolation for Pout(t)
# ------------------------------------------------------------

period_s = 1.0 / (DELTA_F_MHZ * 1e6)
t_seconds = np.linspace(0.0, period_s, 2000)

illustrative_omega_rf_mhz = [1.0, 3.0, 5.0]

omega_total_curves = {}
pout_curves = {}
for orf in illustrative_omega_rf_mhz:
    theta = 2.0 * np.pi * DELTA_F_MHZ * 1e6 * t_seconds  # Delta_phi = 0
    omega_tot = np.sqrt(OMEGA_LO_MHZ ** 2 + orf ** 2 + 2.0 * OMEGA_LO_MHZ * orf * np.cos(theta))
    omega_total_curves[orf] = omega_tot

    f_interp_c = interp1d(static_omega_mhz, static_pout_w, kind="cubic", bounds_error=False, fill_value=np.nan)
    pout_curves[orf] = f_interp_c(omega_tot)

    print(f"Omega_RF/2pi={orf}MHz: Omega_total range=[{omega_tot.min():.4f},{omega_tot.max():.4f}]MHz, "
          f"Pout range=[{np.nanmin(pout_curves[orf])/1e-6:.4f},{np.nanmax(pout_curves[orf])/1e-6:.4f}]microW")

# ------------------------------------------------------------
# Save raw results
# ------------------------------------------------------------

np.savez(
    OUTPUT_DIR / "fig5_data.npz",
    static_omega_mhz=static_omega_mhz, static_pout_w=static_pout_w,
    disp_omega=disp_omega, disp_pout=disp_pout,
    omega_low=fit["omega_low"], omega_high=fit["omega_high"], r2=fit["r2"],
    P0_bar=P0_bar, OMEGA_LO_MHZ=OMEGA_LO_MHZ, DELTA_F_MHZ=DELTA_F_MHZ,
    period_s=period_s, t_seconds=t_seconds,
    omega_total_1mhz=omega_total_curves[1.0], omega_total_3mhz=omega_total_curves[3.0],
    omega_total_5mhz=omega_total_curves[5.0],
    pout_1mhz=pout_curves[1.0], pout_3mhz=pout_curves[3.0], pout_5mhz=pout_curves[5.0],
)
print(f"\nSaved: fig5_data.npz")

# ------------------------------------------------------------
# PLOT -- panel (a) ONLY for now (panels b/c deferred -- see
# math report Section 7 for why: real, unresolved ambiguities
# in Delta_phi and the illustrative Omega_RF interpretation,
# on top of the paper's own time-axis inconsistency. The
# underlying data for b/c is still computed above and saved to
# fig5_data.npz in case they're revisited later, just not
# rendered as the current deliverable plot.)
#
# The "linear dynamic range" is now drawn as an ACTUAL straight
# line (from the fit's slope/intercept), not the real curve
# re-colored -- so you can see where it genuinely diverges from
# the real data, matching the paper's own visual convention.
# ------------------------------------------------------------

fig, ax_a = plt.subplots(figsize=(8, 6.5))

# Real curve (gray)
ax_a.plot(disp_omega, disp_pout / 1e-6, color="gray", linewidth=1.8, label="Obtained by QuTiP")

# ACTUAL fitted straight line: y = slope*x + intercept, drawn only
# across the fitted region's x-range
fit_x = static_omega_mhz[fit["left"]:fit["right"]]
fit_y_line = fit["slope"] * fit_x + fit["intercept"]
ax_a.plot(fit_x, fit_y_line / 1e-6, color="orange", linewidth=2.5,
          label=f"Linear fit (R²={fit['r2']:.4f})")

# Tight rectangle bounded by the FIT LINE's own y-range at the two edges
# (not the full plot height) -- matches the paper's own box style, which
# hugs the linear segment rather than spanning axis-to-axis.
y_at_low = float(fit["slope"] * fit["omega_low"] + fit["intercept"]) / 1e-6
y_at_high = float(fit["slope"] * fit["omega_high"] + fit["intercept"]) / 1e-6
ax_a.fill_between([fit["omega_low"], fit["omega_high"]], y_at_high, y_at_low,
                   color="orange", alpha=0.15)
ax_a.axvline(OMEGA_LO_MHZ, color="cyan", linestyle="--", linewidth=1.3)
ax_a.annotate(r"$(\Omega_{LO})_{opt}$", xy=(OMEGA_LO_MHZ, disp_pout.min() / 1e-6),
              xytext=(OMEGA_LO_MHZ + 1.5, disp_pout.min() / 1e-6 - 0.1),
              color="cyan", fontsize=10, arrowprops=dict(arrowstyle="->", color="cyan"))
ax_a.set_xlabel(r"$\Omega_{total}/2\pi$ (MHz)")
ax_a.set_ylabel(r"$P_{out}$ ($\times10^{-6}$ W)")
ax_a.set_title("Fig. 5(a). Quantum procedure\n(d = 0.76 mm)")
ax_a.set_xlim(0, 14)
ax_a.legend(fontsize=9, loc="upper right")
ax_a.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(OUTPUT_DIR / "fig5a_distortion.png", dpi=220, bbox_inches="tight")
plt.close(fig)
print("Saved: fig5a_distortion.png (panel (a) only; b/c deferred, see math report)")

# ------------------------------------------------------------
# Append numeric results to the math report
# ------------------------------------------------------------

with open(OUTPUT_DIR / "fig5_math_report.md", "a") as f:
    f.write("\n\n## 6. Actual numeric results (this run)\n\n")
    f.write(f"- Linear dynamic range: [{fit['omega_low']:.4f}, {fit['omega_high']:.4f}] MHz, R^2={fit['r2']:.6f}\n")
    f.write(f"- P0_bar = Pout(Omega_LO) = {P0_bar/1e-6:.4f} microW\n")
    f.write(f"- Real period (1/Delta_f) = {period_s*1e6:.4f} microseconds "
            f"(paper's plotted axis reads 0-3e-10s, a ~{period_s/3e-10:.0f}x mismatch, not silently corrected)\n")
    for orf in illustrative_omega_rf_mhz:
        f.write(f"- Omega_RF/2pi={orf}MHz: Omega_total range=[{omega_total_curves[orf].min():.4f},"
                f"{omega_total_curves[orf].max():.4f}]MHz, Pout range=["
                f"{np.nanmin(pout_curves[orf])/1e-6:.4f},{np.nanmax(pout_curves[orf])/1e-6:.4f}] microW\n")

print("\nAppended numeric results to fig5_math_report.md")
print("\nDONE.")
