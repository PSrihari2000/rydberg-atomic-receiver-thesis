# ============================================================
# FIG. 9 -- paper-axis-matched styling, using OUR real data
#
# NEW file, does not touch fig9_ser_vs_snr.py/.png, fig9_data.npz,
# fig9_lofree_quantum_detector.py/.npz, fig9_math_report.md, or
# fig9_ser_vs_snr_freshbuild.py/.png/csv/.md.
#
# Reuses fig6_fresh_build/fig6_data.csv's real SNR values (same source
# as fig9_ser_vs_snr_freshbuild.py) -- restyled to the paper's own
# axis ranges/shading convention: (a)/(b) SNR in [-40,40]dB,
# (c) SNR in [-40,20]dB, "Operating region"/"Distortion region"
# shading matching the paper's Fig.9 layout. This is a PRESENTATION
# change only (axis limits, shading) -- no new computation, no
# refitting, no tuning of any underlying value.
# ============================================================

from pathlib import Path
import csv

import numpy as np
from scipy.special import erfc
import matplotlib.pyplot as plt

OUTPUT_DIR = Path(__file__).resolve().parent
FIG6_CSV = OUTPUT_DIR.parent / "fig6_fresh_build" / "fig6_data.csv"

print("=" * 70)
print("FIG. 9 -- paper-axis-matched styling (real data, presentation only)")
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

def get_col(name):
    vals = []
    for r in data_rows:
        v = r[col[name]]
        vals.append(float(v) if v != "" else np.nan)
    return np.array(vals)

conv_snr_dB = get_col("SNR_conventional_dB_PTx-10dBm")
prac_snr_dB = get_col("SNR_practical_LO_dressed_dB_PTx-10dBm")
lofree_snr_dB_10 = get_col("SNR_LO_free_dB_PTx10dBm")
lofree_snr_dB_20 = get_col("SNR_LO_free_dB_PTx20dBm")

def Q(x): return 0.5 * erfc(x / np.sqrt(2.0))
def ser_pam(M, snr_linear): return 2.0*(1.0-1.0/M) * Q(np.sqrt(6.0*snr_linear/(M**2-1.0)))
def ser_qam(M, snr_linear):
    inner = 1.0 - 2.0*(1.0-1.0/np.sqrt(M)) * Q(np.sqrt(3.0*snr_linear/(M-1.0)))
    return 1.0 - inner**2
def dB_to_lin(dB): return 10.0**(dB/10.0)

# ------------------------------------------------------------
# Panel (a) Conventional -- paper's own axis: SNR in [-40,40]dB
# ------------------------------------------------------------
log2M_conv = [2, 4, 6, 8]
mask_a = ~np.isnan(conv_snr_dB) & (conv_snr_dB >= -40) & (conv_snr_dB <= 40)
snr_a = np.sort(conv_snr_dB[mask_a])
print(f"\n(a) Conventional: {mask_a.sum()} real points fall inside the paper's [-40,40]dB window "
      f"(full real range is [{np.nanmin(conv_snr_dB):.1f},{np.nanmax(conv_snr_dB):.1f}]dB)")
ser_a = {m: ser_qam(2**m, dB_to_lin(snr_a)) for m in log2M_conv}

# ------------------------------------------------------------
# Panel (b) LO-dressed (Practical) -- paper's own axis: SNR in [-40,40]dB
# ------------------------------------------------------------
log2M_lodressed = [2, 4, 6, 8, 10]
mask_b = ~np.isnan(prac_snr_dB) & (prac_snr_dB >= -40) & (prac_snr_dB <= 40)
snr_b = np.sort(prac_snr_dB[mask_b])
print(f"(b) LO-dressed (Practical): {mask_b.sum()} real points fall inside [-40,40]dB "
      f"(full real range is [{np.nanmin(prac_snr_dB):.1f},{np.nanmax(prac_snr_dB):.1f}]dB)")
ser_b = {m: ser_qam(2**m, dB_to_lin(snr_b)) for m in log2M_lodressed}

# ------------------------------------------------------------
# Panel (c) LO-free -- paper's own axis: SNR in [-40,20]dB (note: narrower
# than a/b in the paper's own Fig.9)
# ------------------------------------------------------------
log2M_lofree = [2, 4, 6, 8]
LOFREE_CEILING_DB = float(np.nanmean(np.concatenate([
    lofree_snr_dB_10[~np.isnan(lofree_snr_dB_10)], lofree_snr_dB_20[~np.isnan(lofree_snr_dB_20)]])))
snr_c = np.linspace(-40, 20, 400)  # paper's own displayed window for panel (c)
ser_c = {}
for m in log2M_lofree:
    M = 2**m
    vals = np.full_like(snr_c, (M-1)/M)
    above = snr_c >= LOFREE_CEILING_DB
    vals[above] = ser_pam(M, dB_to_lin(snr_c[above]))
    ser_c[m] = vals
n_above = int(np.sum(snr_c >= LOFREE_CEILING_DB))
print(f"(c) LO-free: real ceiling = {LOFREE_CEILING_DB:.4f}dB -- "
      f"{'OUTSIDE' if LOFREE_CEILING_DB > 20 else 'inside'} the paper's own [-40,20]dB window for this panel "
      f"({n_above}/{len(snr_c)} points in the shown window reach the operating region)")

# ------------------------------------------------------------
# PLOT -- paper's own axis ranges/shading
# ------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

for m in log2M_conv:
    axes[0].semilogy(snr_a, ser_a[m], label=f"log2M={m}")
axes[0].set_title("(a) Conventional")
axes[0].set_xlim(-40, 40); axes[0].set_ylim(1e-6, 1)
axes[0].set_xlabel("SNR (dB)"); axes[0].set_ylabel("SER")
axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

for m in log2M_lodressed:
    axes[1].semilogy(snr_b, ser_b[m], label=f"log2M={m}")
axes[1].set_title("(b) LO-dressed")
axes[1].set_xlim(-40, 40); axes[1].set_ylim(1e-6, 1)
axes[1].set_xlabel("SNR (dB)"); axes[1].set_ylabel("SER")
axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

for m in log2M_lofree:
    axes[2].semilogy(snr_c, ser_c[m], label=f"log2M={m}")
axes[2].axvspan(-40, LOFREE_CEILING_DB if LOFREE_CEILING_DB < 20 else 20, color="gray", alpha=0.15)
if LOFREE_CEILING_DB < 20:
    axes[2].axvspan(LOFREE_CEILING_DB, 20, color="purple", alpha=0.10)
axes[2].set_title(f"(c) LO-free (real ceiling={LOFREE_CEILING_DB:.1f}dB, "
                   f"{'off the right edge of this window' if LOFREE_CEILING_DB>20 else 'inside window'})")
axes[2].set_xlim(-40, 20); axes[2].set_ylim(1e-6, 1)
axes[2].set_xlabel("SNR (dB)"); axes[2].set_ylabel("SER")
axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3)

fig.suptitle("Fig. 9, paper's own axis ranges -- real data, no tuning", fontsize=13)
fig.tight_layout(rect=[0,0,1,0.94])
fig.savefig(OUTPUT_DIR / "fig9_paper_axis_matched.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("\nSaved: fig9_paper_axis_matched.png")
print("\nDONE.")
