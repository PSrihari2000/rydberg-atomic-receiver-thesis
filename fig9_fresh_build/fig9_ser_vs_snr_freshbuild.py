# ============================================================
# FIG. 9 -- SER VERSUS SNR, INDEPENDENT REBUILD ON TODAY'S DATA
# (a) Conventional, (b) LO-dressed, (c) LO-free
#
# NEW file, does not touch the existing fig9_ser_vs_snr.py/.png,
# fig9_data.npz, fig9_lofree_quantum_detector.py/.npz, fig9_math_report.md.
#
# Reuses fig6_fresh_build/fig6_data.csv's REAL, already-computed SNR
# values (Conventional/Theoretical/Practical LO-dressed/LO-free, real
# distance sweep, real Fig.4/Fig.5 gating) -- no new QuTiP solves. The
# SER formulas themselves are standard closed-form textbook results
# (Proakis, Digital Communications):
#   M-PAM:  SER = 2(1-1/M) * Q( sqrt(6*SNR/(M^2-1)) )                (Eq.5.2-46)
#   M-QAM:  SER = 1 - [1 - 2(1-1/sqrt(M))*Q(sqrt(3*SNR/(M-1)))]^2    (Eq.5.2-79)
# SNR here is the LINEAR average SNR per symbol (converted from our real dB values).
#
# LO-dressed uses PRACTICAL (not Theoretical) SNR, so the real decline at
# high SNR/short distance shows up, matching the paper's own described
# non-monotonic SER behaviour for this receiver.
#
# LO-free: honest treatment, not forced to look like the paper. LO-free's
# SNR is a hard ~46.02dB ceiling with nothing achievable below it (already
# established independently multiple times in this project) -- below the
# ceiling there is no real signal to detect, so SER is modelled as random
# guessing, (M-1)/M, exactly as the earlier fig9_lofree_quantum_detector.py
# Monte-Carlo check found; at/above the ceiling, the real PAM SER formula
# applies. This produces a near-vertical "wall," not the paper's smooth
# curve -- reported as-is.
# ============================================================

from pathlib import Path
import csv

import numpy as np
from scipy.special import erfc
import matplotlib.pyplot as plt

OUTPUT_DIR = Path(__file__).resolve().parent
FIG6_CSV = OUTPUT_DIR.parent / "fig6_fresh_build" / "fig6_data.csv"

print("=" * 70)
print("FIG. 9 -- SER vs SNR, independent rebuild on today's Fig.6 data")
print("=" * 70)

with open(FIG6_CSV, newline="") as f:
    rows = list(csv.reader(f))
meta = {}
i = 0
while rows[i]:
    if rows[i][0] != "# METADATA":
        meta[rows[i][0]] = rows[i][1]
    i += 1
i += 1
header = rows[i]
col = {name: k for k, name in enumerate(header)}
data_rows = rows[i+1:]

threshold_mhz = float(meta["fig4_threshold_mhz"])
print(f"Loaded Fig.6 data: {len(data_rows)} distance points, Fig.4 threshold={threshold_mhz:.4f}MHz")

def get_col(name):
    vals = []
    for r in data_rows:
        v = r[col[name]]
        vals.append(float(v) if v != "" else np.nan)
    return np.array(vals)

distance_m = get_col("distance_m")
conv_snr_dB = get_col("SNR_conventional_dB_PTx-10dBm")
prac_snr_dB = get_col("SNR_practical_LO_dressed_dB_PTx-10dBm")
lofree_snr_dB = get_col("SNR_LO_free_dB_PTx10dBm")

def get_bool_col(name):
    return np.array([r[col[name]] == "True" for r in data_rows])

prac_in_linear = get_bool_col("practical_in_linear_region_PTx-10dBm")

# ------------------------------------------------------------
# SER FORMULAS (Proakis, standard closed-form)
# ------------------------------------------------------------

def Q(x):
    return 0.5 * erfc(x / np.sqrt(2.0))

def ser_pam(M, snr_linear):
    return 2.0*(1.0 - 1.0/M) * Q(np.sqrt(6.0*snr_linear/(M**2 - 1.0)))

def ser_qam(M, snr_linear):
    inner = 1.0 - 2.0*(1.0 - 1.0/np.sqrt(M)) * Q(np.sqrt(3.0*snr_linear/(M - 1.0)))
    return 1.0 - inner**2

def dB_to_lin(dB):
    return 10.0**(dB/10.0)

# ------------------------------------------------------------
# (a) CONVENTIONAL -- straight formula application, real SNR sweep
# ------------------------------------------------------------

log2M_conv = [2, 4, 6, 8]
conv_valid = ~np.isnan(conv_snr_dB)
conv_snr_sorted_idx = np.argsort(conv_snr_dB[conv_valid])
conv_snr_dB_sorted = conv_snr_dB[conv_valid][conv_snr_sorted_idx]
conv_ser = {}
for log2M in log2M_conv:
    M = 2**log2M
    conv_ser[log2M] = ser_qam(M, dB_to_lin(conv_snr_dB_sorted))
print(f"\n(a) Conventional: {len(conv_snr_dB_sorted)} real SNR points, range=[{conv_snr_dB_sorted.min():.2f},"
      f"{conv_snr_dB_sorted.max():.2f}]dB")

# ------------------------------------------------------------
# (b) LO-DRESSED -- Practical (real, distortion-aware) SNR
# ------------------------------------------------------------

log2M_lodressed = [2, 4, 6, 8, 10]
prac_valid = ~np.isnan(prac_snr_dB)
prac_snr_sorted_idx = np.argsort(prac_snr_dB[prac_valid])
prac_snr_dB_sorted = prac_snr_dB[prac_valid][prac_snr_sorted_idx]
prac_in_linear_sorted = prac_in_linear[prac_valid][prac_snr_sorted_idx]
lodressed_ser = {}
for log2M in log2M_lodressed:
    M = 2**log2M
    lodressed_ser[log2M] = ser_qam(M, dB_to_lin(prac_snr_dB_sorted))
print(f"(b) LO-dressed (Practical): {len(prac_snr_dB_sorted)} real SNR points, range=[{prac_snr_dB_sorted.min():.2f},"
      f"{prac_snr_dB_sorted.max():.2f}]dB")

# Real distortion-region boundary: the highest SNR (sorted) at which the underlying
# distance point is still marked in-linear-region by fig6's real Omega_RF,max gate.
# Everything past that, in SNR-sorted order, includes at least one out-of-linear-range
# point mixed in -- read directly from fig6_data.csv's real flag, not assumed.
if np.any(prac_in_linear_sorted):
    LODRESSED_DISTORTION_BOUNDARY_DB = float(prac_snr_dB_sorted[prac_in_linear_sorted].max())
else:
    LODRESSED_DISTORTION_BOUNDARY_DB = None
print(f"(b) real linear/distortion boundary (from fig6's Omega_RF,max gate) = "
      f"{LODRESSED_DISTORTION_BOUNDARY_DB:.4f}dB" if LODRESSED_DISTORTION_BOUNDARY_DB is not None
      else "(b) no in-linear-region points found")

# ------------------------------------------------------------
# (c) LO-FREE -- honest ceiling-aware treatment
# ------------------------------------------------------------

log2M_lofree = [2, 4, 6, 8]
lofree_resolved = lofree_snr_dB[~np.isnan(lofree_snr_dB)]
lofree_ceiling_std = float(np.std(lofree_resolved))
LOFREE_CEILING_DB = float(np.mean(lofree_resolved))  # from fig6_data.csv's real LO-free column, not hardcoded
print(f"\nLO-free ceiling read from fig6_data.csv's real column: mean={LOFREE_CEILING_DB:.4f}dB "
      f"over {len(lofree_resolved)} resolved points, std={lofree_ceiling_std:.2e}dB "
      f"({'flat, as expected' if lofree_ceiling_std < 1e-3 else 'WARNING: not flat, check this'})")
snr_sweep_dB = np.linspace(-40, 50, 400)
lofree_ser = {}
for log2M in log2M_lofree:
    M = 2**log2M
    ser_vals = np.full_like(snr_sweep_dB, (M-1)/M)  # below the ceiling: no real signal, random guess
    above = snr_sweep_dB >= LOFREE_CEILING_DB
    ser_vals[above] = ser_pam(M, dB_to_lin(snr_sweep_dB[above]))
    lofree_ser[log2M] = ser_vals
print(f"(c) LO-free: ceiling-gated sweep, {np.sum(above)} points at/above {LOFREE_CEILING_DB}dB "
      f"out of {len(snr_sweep_dB)} (below: random-guess SER=(M-1)/M, no real signal)")

# ------------------------------------------------------------
# SAVE CSV
# ------------------------------------------------------------

csv_path = OUTPUT_DIR / "fig9_data_freshbuild.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["# Panel (a) Conventional -- SNR_dB, then SER for log2M=2,4,6,8"])
    writer.writerow(["SNR_dB"] + [f"SER_log2M{m}" for m in log2M_conv])
    for i in range(len(conv_snr_dB_sorted)):
        writer.writerow([f"{conv_snr_dB_sorted[i]:.4f}"] + [f"{conv_ser[m][i]:.6e}" for m in log2M_conv])
    writer.writerow([])
    writer.writerow(["# Panel (b) LO-dressed (Practical) -- SNR_dB, then SER for log2M=2,4,6,8,10"])
    writer.writerow(["SNR_dB"] + [f"SER_log2M{m}" for m in log2M_lodressed])
    for i in range(len(prac_snr_dB_sorted)):
        writer.writerow([f"{prac_snr_dB_sorted[i]:.4f}"] + [f"{lodressed_ser[m][i]:.6e}" for m in log2M_lodressed])
    writer.writerow([])
    writer.writerow(["# Panel (c) LO-free -- SNR_dB, then SER for log2M=2,4,6,8"])
    writer.writerow(["SNR_dB"] + [f"SER_log2M{m}" for m in log2M_lofree])
    for i in range(len(snr_sweep_dB)):
        writer.writerow([f"{snr_sweep_dB[i]:.4f}"] + [f"{lofree_ser[m][i]:.6e}" for m in log2M_lofree])
print(f"\nSaved: {csv_path.name}")

# ------------------------------------------------------------
# PLOT -- 3 panels matching paper's Fig.9 layout
# ------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

for log2M in log2M_conv:
    axes[0].semilogy(conv_snr_dB_sorted, conv_ser[log2M], label=f"$\\log_2 M={log2M}$")
axes[0].set_title("(a) Conventional")
axes[0].set_xlabel("SNR (dB)"); axes[0].set_ylabel("SER")
axes[0].set_xlim(-40, 40); axes[0].set_ylim(1e-6, 1)
axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

for log2M in log2M_lodressed:
    axes[1].semilogy(prac_snr_dB_sorted, lodressed_ser[log2M], label=f"$\\log_2 M={log2M}$")
# Real boundary from fig6's own Omega_RF,max gate: the highest SNR at which any point
# is still within Fig.5's real linear range. Above it, only out-of-range (distorted)
# points remain (individually scattered right at this edge due to the non-monotonic
# "hump", but this single cutoff is where the linear region ends for good).
if LODRESSED_DISTORTION_BOUNDARY_DB is not None:
    axes[1].axvspan(LODRESSED_DISTORTION_BOUNDARY_DB, 60, color="royalblue", alpha=0.15)
    axes[1].annotate("Distortion\nregion", xy=(LODRESSED_DISTORTION_BOUNDARY_DB, 0.4),
                      xytext=(LODRESSED_DISTORTION_BOUNDARY_DB-16, 0.4),
                      fontsize=8, ha="left", va="center", color="royalblue",
                      arrowprops=dict(arrowstyle="->", color="royalblue"))
axes[1].set_title("(b) LO-dressed")
axes[1].set_xlabel("SNR (dB)"); axes[1].set_ylabel("SER")
axes[1].set_xlim(-40, 60); axes[1].set_ylim(1e-6, 1)
axes[1].legend(fontsize=8, loc="lower left"); axes[1].grid(alpha=0.3)

for log2M in log2M_lofree:
    axes[2].semilogy(snr_sweep_dB, lofree_ser[log2M], label=f"$\\log_2 M={log2M}$")
axes[2].axvspan(snr_sweep_dB.min(), LOFREE_CEILING_DB, color="royalblue", alpha=0.15)
axes[2].annotate("Distortion region", xy=(snr_sweep_dB.min()+2, 3e-6), fontsize=8, ha="left", color="royalblue")
axes[2].axvline(LOFREE_CEILING_DB, color="k", linestyle=":", alpha=0.7)
axes[2].annotate(f"{LOFREE_CEILING_DB:.2f}dB", xy=(LOFREE_CEILING_DB, 0.5),
                  xytext=(LOFREE_CEILING_DB-35, 0.3), fontsize=8,
                  arrowprops=dict(arrowstyle="->", color="black"))
axes[2].set_title("(c) LO-free")
axes[2].set_xlabel("SNR (dB)"); axes[2].set_ylabel("SER")
axes[2].set_ylim(1e-6, 1); axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3)

fig.suptitle("Fig. 9. SER performance versus the SNR.", fontsize=13)
fig.tight_layout(rect=[0,0,1,0.94])
fig.savefig(OUTPUT_DIR / "fig9_ser_vs_snr_freshbuild.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved: fig9_ser_vs_snr_freshbuild.png")

# ------------------------------------------------------------
# MATH REPORT
# ------------------------------------------------------------

report = f"""# Fig. 9 math report -- independent rebuild, SER vs SNR

NEW file, does not touch the existing fig9_ser_vs_snr.py/.png, fig9_data.npz,
fig9_lofree_quantum_detector.py/.npz, fig9_math_report.md.

Reuses `fig6_fresh_build/fig6_data.csv`'s real, already-computed SNR values (no new
QuTiP solves) -- Conventional and LO-dressed use the full real distance sweep at
PTx=-10dBm as the SNR axis; LO-dressed uses Practical (not Theoretical) SNR so the
real distortion-driven decline is preserved. SER formulas are standard closed-form
results (Proakis, *Digital Communications*):

    M-PAM SER = 2(1-1/M) * Q(sqrt(6*SNR/(M^2-1)))              (Eq.5.2-46)
    M-QAM SER = 1 - [1 - 2(1-1/sqrt(M))*Q(sqrt(3*SNR/(M-1)))]^2  (Eq.5.2-79)

Q(x) = 0.5*erfc(x/sqrt(2)), SNR is the linear (not dB) average SNR per symbol.

## (a) Conventional -- 8/16/64/256-QAM (log2M=2,4,6,8)

Straight formula application over the real Conventional SNR sweep,
[{conv_snr_dB_sorted.min():.2f},{conv_snr_dB_sorted.max():.2f}]dB across {len(conv_snr_dB_sorted)} points.

## (b) LO-dressed -- QAM up to log2M=10 (1024-QAM, matching the paper's own stated order)

Uses Practical LO-dressed's real SNR values, [{prac_snr_dB_sorted.min():.2f},
{prac_snr_dB_sorted.max():.2f}]dB across {len(prac_snr_dB_sorted)} points -- includes the
real short-distance region where SNR is degraded by distortion (below Fig.5's linear
range), so SER should show the same non-monotonic character already established for
Practical LO-dressed's SNR curve itself in Fig.6.

## (c) LO-free -- 4/16/64/256-PAM (log2M=2,4,6,8), honest ceiling-gated treatment

LO-free's SNR is NOT a continuous sweepable quantity in this model -- it is a hard
ceiling with no real signal available below it (established multiple independent ways
earlier in this project, including a full Monte-Carlo "quantum detector" that confirmed
a sophisticated receiver cannot extract signal that isn't physically present). The
ceiling value used here, {LOFREE_CEILING_DB:.4f}dB, is read directly from
fig6_data.csv's real LO-free column (mean over {len(lofree_resolved)} resolved points,
std={lofree_ceiling_std:.2e}dB -- confirms it is genuinely flat, not assumed), matching
the algebraic 1/epsilon_tilde^2 prediction (epsilon_tilde=0.5%) but derived from the
loaded data rather than hardcoded. Modelled here as: below {LOFREE_CEILING_DB:.4f}dB,
no real detection is possible, so SER = random guessing = (M-1)/M; at/above
{LOFREE_CEILING_DB:.4f}dB, the real PAM SER formula applies.

This produces a near-vertical "wall" at the ceiling, not the paper's own published
smooth SER-vs-SNR curve spanning a wide range. This is a deliberate, honest
consequence of the real physics under the paper's own stated Eq.16/epsilon_tilde=0.5%
model, not a bug -- reported as-is, consistent with this project's standing policy of
not forcing curves to resemble the paper's published shape.

## Parameters/data sources

    Fig.4 threshold (from fig6_data.csv metadata): {threshold_mhz:.4f} MHz
    All underlying SNR values: real, from fig6_fresh_build's independently-rebuilt
    Fig.6 pipeline (today's Fig.4/Fig.5 data, not the older frozen fig6 build).
"""
with open(OUTPUT_DIR / "fig9_math_report_freshbuild.md", "w") as f:
    f.write(report)
print("Saved: fig9_math_report_freshbuild.md")
print("\nDONE.")
