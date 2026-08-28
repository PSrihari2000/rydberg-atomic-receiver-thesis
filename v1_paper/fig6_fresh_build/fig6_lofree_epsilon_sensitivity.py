# ============================================================
# EXPLORATORY -- sensitivity of the LO-free SNR curve to epsilon_tilde
# (the observation-uncertainty parameter, paper-stated 0.5%, cited to
# Sedlacek et al. 2012 Table 1's laser/detection technical noise for
# THEIR specific apparatus -- not a universal constant).
#
# NEW file, does not touch fig6_snr_vs_distance.py or any other
# existing fig6 file. Same Eq.8-16 pipeline and Fig.4 real threshold
# gating as the main build -- only epsilon_tilde is swept.
# ============================================================

from pathlib import Path
import csv

import numpy as np
import matplotlib.pyplot as plt

OUTPUT_DIR = Path(__file__).resolve().parent
FIG4_CSV = OUTPUT_DIR.parent / "fig4_fresh_build" / "fig4_classification.csv"

with open(FIG4_CSV, newline="") as f:
    rows = list(csv.reader(f))
threshold_mhz = float(rows[2][1])
print(f"Loaded Fig.4 threshold: {threshold_mhz:.4f} MHz")

e_charge = 1.6e-19
a0 = 5.2e-11
hbar = 1.054571817e-34
eta0 = 377.0
GTx_dBi = 2.15
sigma_BGN_dBm = -90.0
wp_RF = -1443.459 * e_charge * a0
wp_RF_over_hbar = wp_RF / hbar

def db_to_lin(db):
    return 10.0 ** (db / 10.0)

def dbm_to_w(dbm):
    return 1e-3 * 10.0 ** (dbm / 10.0)

GTx_lin = db_to_lin(GTx_dBi)
sigma_BGN_sq = dbm_to_w(sigma_BGN_dBm)

DISTANCE_M = np.logspace(np.log10(0.1), np.log10(1e6), 281)

def PRx_flux(PTx_W, d):
    return PTx_W * GTx_lin * eta0 / (4.0 * np.pi * d**2)

def snr_lofree(PTx_W, d, eps_tilde):
    PRx = PRx_flux(PTx_W, d)
    ERF = np.sqrt(PRx)
    Omega_RF_mhz = ERF * abs(wp_RF_over_hbar) / (2*np.pi) / 1e6
    sigma_UN_sq = (eps_tilde**2) * PRx
    sigma_Ry_sq = sigma_UN_sq + sigma_BGN_sq
    snr = PRx / sigma_Ry_sq
    resolved = Omega_RF_mhz >= threshold_mhz
    return (10.0*np.log10(snr) if resolved else np.nan)

EPS_VALUES = [0.001, 0.0025, 0.005, 0.01, 0.02, 0.05]  # 0.1% .. 5%, paper's 0.5% included
PTX_SET = [10.0, 20.0]

results = {}
for ptx in PTX_SET:
    PTx_W = dbm_to_w(ptx)
    for eps in EPS_VALUES:
        vals = np.array([snr_lofree(PTx_W, d, eps) for d in DISTANCE_M])
        results[(ptx, eps)] = vals
        n_res = np.sum(~np.isnan(vals))
        ceiling = 10*np.log10(1.0/eps**2)
        max_dist_resolved = DISTANCE_M[~np.isnan(vals)].max() if n_res else float("nan")
        min_val = np.nanmin(vals) if n_res else float("nan")
        max_val = np.nanmax(vals) if n_res else float("nan")
        print(f"PTx={ptx:g}dBm, eps_tilde={eps*100:.2f}%: resolved points={n_res}, "
              f"cutoff distance={max_dist_resolved:.4f}m (UNCHANGED by eps), "
              f"theoretical ceiling 1/eps^2={ceiling:.2f}dB, "
              f"actual curve range=[{min_val:.2f},{max_val:.2f}]dB")

# ------------------------------------------------------------
# SAVE CSV
# ------------------------------------------------------------
csv_path = OUTPUT_DIR / "fig6_lofree_epsilon_sensitivity.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    header = ["distance_m"] + [f"PTx{ptx:g}dBm_eps{eps*100:g}pct" for ptx in PTX_SET for eps in EPS_VALUES]
    writer.writerow(header)
    for i in range(len(DISTANCE_M)):
        row = [f"{DISTANCE_M[i]:.6f}"]
        for ptx in PTX_SET:
            for eps in EPS_VALUES:
                v = results[(ptx, eps)][i]
                row.append(f"{v:.4f}" if not np.isnan(v) else "")
        writer.writerow(row)
print(f"Saved: {csv_path.name}")

# ------------------------------------------------------------
# PLOT -- one panel per PTx, one line per epsilon_tilde
# ------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
colors = plt.cm.viridis(np.linspace(0, 0.9, len(EPS_VALUES)))

for ax, ptx in zip(axes, PTX_SET):
    for eps, c in zip(EPS_VALUES, colors):
        vals = results[(ptx, eps)]
        style = "-" if abs(eps - 0.005) < 1e-9 else "--"
        lw = 2.4 if abs(eps - 0.005) < 1e-9 else 1.4
        label = f"eps={eps*100:.2f}%" + ("  (paper's value)" if abs(eps-0.005) < 1e-9 else "")
        ax.plot(DISTANCE_M, vals, style, color=c, linewidth=lw, label=label)
    ax.set_xscale("log")
    ax.set_xlabel(r"Distance, $d_{Tx-Rx}$ (m)")
    ax.set_title(f"LO-free, PTx={ptx:g}dBm")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
axes[0].set_ylabel("SNR (dB)")
fig.suptitle("Exploratory: LO-free SNR sensitivity to epsilon_tilde (observation uncertainty)")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "fig6_lofree_epsilon_sensitivity.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved: fig6_lofree_epsilon_sensitivity.png")
print("\nDONE.")
