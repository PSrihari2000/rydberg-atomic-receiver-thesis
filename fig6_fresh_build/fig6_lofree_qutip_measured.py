# ============================================================
# EXPLORATORY -- LO-free SNR using Fig.4's REAL simulated (jagged)
# splitting instead of the clean Eq.8 theoretical Omega_RF.
#
# NEW file, does not touch fig6_snr_vs_distance.py or any other
# existing fig6 file. Purpose: show what the LO-free curve looks like
# under a more literal reading of the paper's "obtained through
# simulations using QuTiP" (Sec.V-B) -- for comparison only, not
# adopted into the main Fig.6 build unless requested.
#
# METHOD:
#   1. At each distance/PTx, compute the THEORETICAL Omega_RF from the
#      physical link budget (Eq.8-10) -- same as before.
#   2. Instead of feeding that theoretical Omega_RF straight into
#      Eq.16, look up Fig.4's REAL simulated separation_MHz (from
#      fig4_classification.csv -- genuine find_peaks() output on real
#      QuTiP steady-state solves, only defined where resolved=True) at
#      that operating point, and use THIS real, jagged value as the
#      "measured" Omega_RF fed into Eq.16 instead.
#   3. Gated to NaN wherever the theoretical Omega_RF falls below
#      Fig.4's real threshold (unresolved) OR outside Fig.4's actually
#      simulated Omega_RF grid (no real data to look up -- not
#      extrapolated).
# ============================================================

from pathlib import Path
import csv

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

OUTPUT_DIR = Path(__file__).resolve().parent
FIG4_CSV = OUTPUT_DIR.parent / "fig4_fresh_build" / "fig4_classification.csv"

print("=" * 70)
print("EXPLORATORY: LO-free SNR using Fig.4's real simulated splitting")
print("=" * 70)

# ------------------------------------------------------------
# LOAD FIG.4's REAL, PER-ROW SIMULATED DATA (not just the threshold)
# ------------------------------------------------------------

with open(FIG4_CSV, newline="") as f:
    rows = list(csv.reader(f))
threshold_mhz = float(rows[2][1])
header_idx = 4  # row: Omega_RF_MHz,n_peaks,left_MHz,right_MHz,separation_MHz,valley_position_MHz,resolved,reason
header = rows[header_idx]
col = {name: i for i, name in enumerate(header)}
data_rows = rows[header_idx + 1:]

omega_rf_grid, separation_grid, resolved_grid = [], [], []
for r in data_rows:
    if not r:
        continue
    omega_rf_grid.append(float(r[col["Omega_RF_MHz"]]))
    sep = r[col["separation_MHz"]]
    resolved = r[col["resolved"]] == "True"
    resolved_grid.append(resolved)
    separation_grid.append(float(sep) if (sep != "" and resolved) else np.nan)

omega_rf_grid = np.array(omega_rf_grid)
separation_grid = np.array(separation_grid)
resolved_grid = np.array(resolved_grid)
order = np.argsort(omega_rf_grid)
omega_rf_grid, separation_grid, resolved_grid = omega_rf_grid[order], separation_grid[order], resolved_grid[order]

valid = resolved_grid & ~np.isnan(separation_grid)
print(f"Loaded {len(omega_rf_grid)} real Fig.4 rows, Omega_RF grid=[{omega_rf_grid.min():.3f},"
      f"{omega_rf_grid.max():.3f}]MHz, {valid.sum()} resolved rows with a real measured separation, "
      f"threshold={threshold_mhz:.4f}MHz")

# Interpolator over ONLY the resolved rows (separation is only physically
# meaningful/measurable where two peaks are actually resolved)
measured_sep_interp = interp1d(omega_rf_grid[valid], separation_grid[valid],
                                bounds_error=False, fill_value=np.nan)
omega_rf_domain_max = omega_rf_grid[valid].max()
omega_rf_domain_min = omega_rf_grid[valid].min()

# ------------------------------------------------------------
# SHARED PARAMETERS (identical values to fig6_snr_vs_distance.py)
# ------------------------------------------------------------

e_charge = 1.6e-19
a0 = 5.2e-11
hbar = 1.054571817e-34
eta0 = 377.0
c_light = 2.998e8

GTx_dBi = 2.15
sigma_BGN_dBm = -90.0
eps_tilde = 0.005
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

# ------------------------------------------------------------
# VARIANT 1 (original, in the main Fig.6 script): theoretical Omega_RF
# straight into Eq.16
# ------------------------------------------------------------

def snr_lofree_formula(PTx_W, d):
    PRx = PRx_flux(PTx_W, d)
    ERF = np.sqrt(PRx)
    Omega_RF_mhz = ERF * abs(wp_RF_over_hbar) / (2*np.pi) / 1e6
    sigma_UN_sq = (eps_tilde**2) * PRx
    sigma_Ry_sq = sigma_UN_sq + sigma_BGN_sq
    snr = PRx / sigma_Ry_sq
    resolved = Omega_RF_mhz >= threshold_mhz
    return (10.0*np.log10(snr) if resolved else np.nan), Omega_RF_mhz

# ------------------------------------------------------------
# VARIANT 2 (this file): real Fig.4-measured splitting into Eq.16
# ------------------------------------------------------------

def snr_lofree_qutip_measured(PTx_W, d):
    PRx = PRx_flux(PTx_W, d)
    ERF = np.sqrt(PRx)
    Omega_RF_theoretical_mhz = ERF * abs(wp_RF_over_hbar) / (2*np.pi) / 1e6

    if (Omega_RF_theoretical_mhz < omega_rf_domain_min
            or Omega_RF_theoretical_mhz > omega_rf_domain_max
            or Omega_RF_theoretical_mhz < threshold_mhz):
        return np.nan, Omega_RF_theoretical_mhz, np.nan  # unresolved OR outside Fig.4's real simulated grid

    Omega_RF_measured_mhz = float(measured_sep_interp(Omega_RF_theoretical_mhz))
    if np.isnan(Omega_RF_measured_mhz):
        return np.nan, Omega_RF_theoretical_mhz, np.nan

    # Convert the REAL measured splitting back to an effective PRx/ERF
    # (inverting Eq.8), then evaluate Eq.16 with THIS measured value
    # standing in for both the signal and the observation-uncertainty term.
    Omega_RF_measured = 2*np.pi * Omega_RF_measured_mhz * 1e6
    ERF_measured = Omega_RF_measured / abs(wp_RF_over_hbar)
    PRx_measured = ERF_measured ** 2

    sigma_UN_sq = (eps_tilde**2) * PRx_measured
    sigma_Ry_sq = sigma_UN_sq + sigma_BGN_sq
    snr = PRx_measured / sigma_Ry_sq
    return 10.0*np.log10(snr), Omega_RF_theoretical_mhz, Omega_RF_measured_mhz

# ------------------------------------------------------------
# RUN BOTH VARIANTS, PTx = 10dBm, 20dBm (paper's LO-free legend)
# ------------------------------------------------------------

PTX_LOFREE_DBM = [10.0, 20.0]
results = {}
for ptx in PTX_LOFREE_DBM:
    PTx_W = dbm_to_w(ptx)
    f_vals = [snr_lofree_formula(PTx_W, d) for d in DISTANCE_M]
    q_vals = [snr_lofree_qutip_measured(PTx_W, d) for d in DISTANCE_M]
    results[f"formula_{ptx:g}dBm"] = np.array([v[0] for v in f_vals])
    results[f"qutip_{ptx:g}dBm"] = np.array([v[0] for v in q_vals])
    results[f"qutip_{ptx:g}dBm_theomhz"] = np.array([v[1] for v in q_vals])
    results[f"qutip_{ptx:g}dBm_measmhz"] = np.array([v[2] for v in q_vals])
    n_resolved_f = np.sum(~np.isnan(results[f"formula_{ptx:g}dBm"]))
    n_resolved_q = np.sum(~np.isnan(results[f"qutip_{ptx:g}dBm"]))
    print(f"PTx={ptx:g}dBm: formula-based resolved points={n_resolved_f}, "
          f"QuTiP-measured resolved points={n_resolved_q}")

# ------------------------------------------------------------
# SAVE CSV
# ------------------------------------------------------------

csv_path = OUTPUT_DIR / "fig6_lofree_qutip_measured.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["distance_m",
                      "SNR_formula_dB_10dBm", "SNR_formula_dB_20dBm",
                      "SNR_qutip_measured_dB_10dBm", "SNR_qutip_measured_dB_20dBm",
                      "Omega_RF_theoretical_MHz_10dBm", "Omega_RF_measured_MHz_10dBm",
                      "Omega_RF_theoretical_MHz_20dBm", "Omega_RF_measured_MHz_20dBm"])
    for i in range(len(DISTANCE_M)):
        def fmt(v):
            return f"{v:.4f}" if not np.isnan(v) else ""
        writer.writerow([f"{DISTANCE_M[i]:.6f}",
                          fmt(results["formula_10dBm"][i]), fmt(results["formula_20dBm"][i]),
                          fmt(results["qutip_10dBm"][i]), fmt(results["qutip_20dBm"][i]),
                          fmt(results["qutip_10dBm_theomhz"][i]), fmt(results["qutip_10dBm_measmhz"][i]),
                          fmt(results["qutip_20dBm_theomhz"][i]), fmt(results["qutip_20dBm_measmhz"][i])])
print(f"Saved: {csv_path.name}")

# ------------------------------------------------------------
# PLOT
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 6.5))
ax.plot(DISTANCE_M, results["formula_10dBm"], "--", color="mediumpurple", linewidth=1.4,
        label="LO-free (formula, Eq.16 w/ theoretical Omega_RF), PTx=10dBm")
ax.plot(DISTANCE_M, results["formula_20dBm"], "--", color="indigo", linewidth=1.4,
        label="LO-free (formula), PTx=20dBm")
ax.plot(DISTANCE_M, results["qutip_10dBm"], "-", color="darkorange", linewidth=1.8,
        label="LO-free (Fig.4 real measured splitting), PTx=10dBm")
ax.plot(DISTANCE_M, results["qutip_20dBm"], "-", color="firebrick", linewidth=1.8,
        label="LO-free (Fig.4 real measured splitting), PTx=20dBm")
ax.set_xscale("log")
ax.set_xlabel(r"Distance, $d_{Tx-Rx}$ (m)")
ax.set_ylabel("SNR (dB)")
ax.set_title("Exploratory: LO-free SNR vs distance -- formula vs Fig.4's real measured splitting")
ax.legend(fontsize=8, loc="upper right")
ax.grid(alpha=0.3, which="both")
ax.set_xlim(0.1, max(DISTANCE_M[~np.isnan(results['qutip_20dBm'])].max()*3, 20))
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "fig6_lofree_qutip_measured.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved: fig6_lofree_qutip_measured.png")
print("\nDONE.")
