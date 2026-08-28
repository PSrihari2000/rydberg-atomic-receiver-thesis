# ============================================================
# FIG. 6 -- d=0.36mm INDEPENDENT REBUILD -- SNR VS DISTANCE
# Identical method to fig6_fresh_build/fig6_snr_vs_distance.py:
# Conventional / Theoretical LO-dressed / Practical LO-dressed / LO-free,
# reusing THIS folder's own freshbuild_fig4_d036mm.csv threshold and
# freshbuild_fig5_d036mm.csv LO-dressed data -- ONLY d_probe changed
# from 0.76mm to 0.36mm. Does not touch fig6_fresh_build or the older
# qutip_d036mm files.
# ============================================================

from pathlib import Path
import time
import csv

import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

OUTPUT_DIR = Path(__file__).resolve().parent
FIG4_CSV = OUTPUT_DIR.parent / "fig4_d036mm" / "freshbuild_fig4_d036mm.csv"
FIG5_CSV = OUTPUT_DIR.parent / "fig5_d036mm" / "freshbuild_fig5_d036mm.csv"

print("=" * 70)
print("FIG. 6 -- d=0.36mm INDEPENDENT REBUILD -- SNR vs distance")
print("=" * 70)

with open(FIG4_CSV, newline="") as f:
    rows = list(csv.reader(f))
threshold_mhz = float(rows[3][1])
print(f"Loaded Fig.4-d036mm threshold: {threshold_mhz:.4f} MHz")

with open(FIG5_CSV, newline="") as f:
    rows = list(csv.reader(f))
meta = {}
i = 0
while rows[i]:
    if rows[i][0] != "# METADATA":
        meta[rows[i][0]] = float(rows[i][1])
    i += 1
i += 2  # blank + "# FIG.5(a)..." header row
i += 1  # column header row
fig5a_omega_mhz, fig5a_pout_uW = [], []
while i < len(rows) and rows[i]:
    fig5a_omega_mhz.append(float(rows[i][0]))
    fig5a_pout_uW.append(float(rows[i][1]))
    i += 1
fig5a_omega_mhz = np.array(fig5a_omega_mhz)
fig5a_pout_W = np.array(fig5a_pout_uW)*1e-6

Omega_LO_mhz = meta["Omega_LO_operating_MHz"]
lin_low_mhz = meta["linear_range_lower_MHz"]
lin_high_mhz = meta["linear_range_upper_MHz"]
kappa_slope_W_per_MHz = meta["linear_fit_slope_W_per_MHz"]
d_probe_mm = meta["d_probe_mm"]
print(f"Loaded Fig.5-d036mm: Omega_LO={Omega_LO_mhz}MHz, linear range=[{lin_low_mhz:.4f},"
      f"{lin_high_mhz:.4f}]MHz, kappa={kappa_slope_W_per_MHz:.6e}W/MHz, d_probe={d_probe_mm}mm")

cubic_fig5 = CubicSpline(fig5a_omega_mhz, fig5a_pout_W)

e_charge, a0, hbar, eps0, eta0 = 1.6e-19, 5.2e-11, 1.054571817e-34, 8.854e-12, 377.0
kB = 1.380649e-23
c_light = 2.998e8
Omega_p = 2.0*np.pi*8.0e6
Omega_c = 2.0*np.pi*1.0e6
gamma2 = 2.0*np.pi*5.2e6
wp_RF = -1443.459*e_charge*a0
wp_12 = (2.5*e_charge*a0)**2
lambda_p = 852e-9
kp_wave = 2.0*np.pi/lambda_p
N0, L_cell = 1.0e15, 1.0e-2
d_probe = d_probe_mm*1e-3
Pin = (np.pi/(2.0*eta0))*(d_probe*Omega_p*hbar/(2.0*np.sqrt(wp_12)))**2
C0 = -2.0*N0*wp_12/(eps0*hbar*Omega_p)
Omega_LO = 2.0*np.pi*Omega_LO_mhz*1e6
Delta_f = 150e3
wp_RF_over_hbar = wp_RF/hbar

GTx_dBi, GRx_dBi, GLNA_dB = 2.15, 2.15, 20.0
RL, D_resp, T_temp, eta_eff = 50.0, 0.55, 290.0, 0.8
sigma_BGN_dBm, eps_tilde, fRF, B_bw = -90.0, 0.005, 3.5e9, 1.0e6

def db_to_lin(db): return 10.0**(db/10.0)
def dbm_to_w(dbm): return 1e-3*10.0**(dbm/10.0)
GTx_lin, GRx_lin, GLNA_lin = db_to_lin(GTx_dBi), db_to_lin(GRx_dBi), db_to_lin(GLNA_dB)
lambda_RF = c_light/fRF
sigma_BGN_sq = dbm_to_w(sigma_BGN_dBm)

def basis4(): return [qt.basis(4,i) for i in range(4)]
def hamiltonian_lodressed(Omega_total_val):
    k1,k2,k3,k4 = basis4()
    H = qt.qzero(4)
    H += (Omega_p/2.0)*(k1*k2.dag()+k2*k1.dag())
    H += (Omega_c/2.0)*(k2*k3.dag()+k3*k2.dag())
    H += (Omega_total_val/2.0)*(k3*k4.dag()+k4*k3.dag())
    return H
def collapse_ops_lodressed():
    k1,k2,k3,k4 = basis4()
    return [np.sqrt(gamma2)*k1*k2.dag()]

n_qutip_solves = 0
def pout_at_omega_total(Omega_total_val):
    global n_qutip_solves
    H = hamiltonian_lodressed(Omega_total_val)
    rho_ss = qt.steadystate(H, collapse_ops_lodressed(), method="direct")
    n_qutip_solves += 1
    rho21 = complex(rho_ss[1,0])
    chi = C0*rho21
    return Pin*np.exp(-kp_wave*L_cell*np.imag(chi))

WIDE_SWEEP_MAX_MHZ = 35.0
wide_omega_mhz = np.linspace(0.05, WIDE_SWEEP_MAX_MHZ, 350)
wide_pout_W = np.zeros_like(wide_omega_mhz)
t0 = time.time()
for idx, w in enumerate(wide_omega_mhz):
    wide_pout_W[idx] = pout_at_omega_total(2.0*np.pi*w*1e6)
dt = time.time()-t0
print(f"{n_qutip_solves} fresh QuTiP solves for wide sweep, {dt:.1f}s")
cubic_wide = CubicSpline(wide_omega_mhz, wide_pout_W)
P0_bar = float(cubic_wide(Omega_LO_mhz))
print(f"P0_bar = {P0_bar/1e-6:.4f} microW")

kappa_true = kappa_slope_W_per_MHz/(2.0*np.pi*1e6)
Ibar = eta_eff*P0_bar*e_charge/(hbar*(c_light/lambda_p))
sigma_PSN_sq = 2.0*e_charge*B_bw*Ibar
sigma_TN_sq = 4.0*kB*T_temp*B_bw
A_LO = GLNA_lin*RL*D_resp**2*kappa_true**2*wp_RF_over_hbar**2
sigma_Ry_LO_sq = A_LO*sigma_BGN_sq + GLNA_lin*RL*D_resp**2*sigma_PSN_sq + sigma_TN_sq
print(f"kappa={kappa_true:.6e}, A_LO={A_LO:.6e}, sigma_Ry_LO^2={sigma_Ry_LO_sq:.6e}")

DISTANCE_M = np.logspace(np.log10(0.1), np.log10(1e6), 281)

def PRx_flux(PTx_W, d): return PTx_W*GTx_lin*eta0/(4.0*np.pi*d**2)
def omega_rf_from_PRx(PRx): return np.sqrt(PRx)*abs(wp_RF_over_hbar)

def snr_conventional_dB(PTx_W, d):
    Pc = (PTx_W*GTx_lin/(4.0*np.pi*d**2)) * (lambda_RF**2/(4.0*np.pi))
    gamma_conv = Pc/(GRx_lin*GLNA_lin*sigma_BGN_sq + sigma_TN_sq)
    return 10.0*np.log10(gamma_conv)

def snr_lofree_dB(PTx_W, d):
    PRx = PRx_flux(PTx_W, d)
    Omega_RF_mhz = omega_rf_from_PRx(PRx)/(2*np.pi)/1e6
    sigma_UN_sq = (eps_tilde**2)*PRx
    sigma_Ry_sq = sigma_UN_sq + sigma_BGN_sq
    snr = PRx/sigma_Ry_sq
    resolved = Omega_RF_mhz >= threshold_mhz
    return (10.0*np.log10(snr) if resolved else np.nan)

def snr_theoretical_dB(PTx_W, d):
    PRx = PRx_flux(PTx_W, d)
    return 10.0*np.log10(A_LO*PRx/sigma_Ry_LO_sq)

N_PERIOD = 512
t_grid = np.linspace(0.0, 1.0/Delta_f, N_PERIOD, endpoint=False)
psi_grid = 2.0*np.pi*Delta_f*t_grid

def snr_practical_dB(PTx_W, d):
    PRx = PRx_flux(PTx_W, d)
    Omega_RF = omega_rf_from_PRx(PRx)
    Omega_total = np.abs(Omega_LO + Omega_RF*np.exp(1j*psi_grid))
    Omega_total_mhz = Omega_total/(2*np.pi)/1e6
    if Omega_total_mhz.max() > WIDE_SWEEP_MAX_MHZ or Omega_total_mhz.min() < wide_omega_mhz.min():
        return np.nan
    Pout_t = cubic_wide(Omega_total_mhz)
    P_df = (2.0/N_PERIOD)*np.abs(np.sum(Pout_t*np.exp(-1j*psi_grid)))
    signal = GLNA_lin*RL*D_resp**2*P_df**2
    return 10.0*np.log10(signal/sigma_Ry_LO_sq)

PTX_CONV_DBM = PTX_THEO_DBM = PTX_PRAC_DBM = [-10.0, 10.0]
PTX_LOFREE_DBM = [10.0, 20.0]

results = {"distance_m": DISTANCE_M}
for ptx in PTX_CONV_DBM:
    PTx_W = dbm_to_w(ptx)
    results[f"conv_{ptx:g}dBm"] = np.array([snr_conventional_dB(PTx_W, d) for d in DISTANCE_M])
for ptx in PTX_THEO_DBM:
    PTx_W = dbm_to_w(ptx)
    results[f"theo_{ptx:g}dBm"] = np.array([snr_theoretical_dB(PTx_W, d) for d in DISTANCE_M])
for ptx in PTX_PRAC_DBM:
    PTx_W = dbm_to_w(ptx)
    results[f"prac_{ptx:g}dBm"] = np.array([snr_practical_dB(PTx_W, d) for d in DISTANCE_M])
for ptx in PTX_LOFREE_DBM:
    PTx_W = dbm_to_w(ptx)
    results[f"lofree_{ptx:g}dBm"] = np.array([snr_lofree_dB(PTx_W, d) for d in DISTANCE_M])

idx100 = int(np.argmin(np.abs(DISTANCE_M-100.0)))
conv_100 = results["conv_-10dBm"][idx100]
theo_100 = results["theo_-10dBm"][idx100]
print(f"\nCheckpoint @ d=100m, PTx=-10dBm: Conventional={conv_100:.4f}dB, "
      f"Theoretical LO-dressed={theo_100:.4f}dB, gap={theo_100-conv_100:.4f}dB")
print(f"(d=0.76mm fresh_build's real checkpoint gap: 28.8864dB)")

csv_path = OUTPUT_DIR / "freshbuild_fig6_d036mm.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["# METADATA"])
    writer.writerow(["d_probe_mm", f"{d_probe_mm:.4f}"])
    writer.writerow(["fig4_threshold_mhz", f"{threshold_mhz:.6f}"])
    writer.writerow(["kappa_W_per_rad_per_s", f"{kappa_true:.6e}"])
    writer.writerow(["P0_bar_W", f"{P0_bar:.6e}"])
    writer.writerow(["A_LO", f"{A_LO:.6e}"])
    writer.writerow(["sigma_Ry_LO_sq", f"{sigma_Ry_LO_sq:.6e}"])
    writer.writerow(["checkpoint_gap_100m_-10dBm_dB", f"{theo_100-conv_100:.6f}"])
    writer.writerow([])
    header = ["distance_m","SNR_conventional_dB_-10dBm","SNR_conventional_dB_10dBm",
              "SNR_theoretical_dB_-10dBm","SNR_theoretical_dB_10dBm",
              "SNR_practical_dB_-10dBm","SNR_practical_dB_10dBm",
              "SNR_LOfree_dB_10dBm","SNR_LOfree_dB_20dBm"]
    writer.writerow(header)
    for i in range(len(DISTANCE_M)):
        def fmt(v): return f"{v:.4f}" if not np.isnan(v) else ""
        writer.writerow([f"{DISTANCE_M[i]:.6f}",
                          fmt(results["conv_-10dBm"][i]), fmt(results["conv_10dBm"][i]),
                          fmt(results["theo_-10dBm"][i]), fmt(results["theo_10dBm"][i]),
                          fmt(results["prac_-10dBm"][i]), fmt(results["prac_10dBm"][i]),
                          fmt(results["lofree_10dBm"][i]), fmt(results["lofree_20dBm"][i])])
print(f"Saved: {csv_path.name}")

fig, ax = plt.subplots(figsize=(9,7))
ax.plot(DISTANCE_M, results["conv_-10dBm"], "-", color="green", linewidth=1.6, label="Conventional, PTx=-10dBm")
ax.plot(DISTANCE_M, results["conv_10dBm"], "-o", color="green", linewidth=1.6, markevery=20, markersize=4, label="Conventional, PTx=10dBm")
ax.plot(DISTANCE_M, results["theo_-10dBm"], "--", color="orange", linewidth=1.4, label="Theoretical LO-dressed, PTx=-10dBm")
ax.plot(DISTANCE_M, results["theo_10dBm"], "--o", color="orange", linewidth=1.4, markevery=20, markersize=4, label="Theoretical LO-dressed, PTx=10dBm")
ax.plot(DISTANCE_M, results["prac_-10dBm"], "-.", color="deepskyblue", linewidth=1.6, label="Practical LO-dressed, PTx=-10dBm")
ax.plot(DISTANCE_M, results["prac_10dBm"], "-.o", color="deepskyblue", linewidth=1.6, markevery=15, markersize=4, label="Practical LO-dressed, PTx=10dBm")
ax.plot(DISTANCE_M, results["lofree_10dBm"], "-", color="purple", linewidth=1.6, label="LO-free, PTx=10dBm")
ax.plot(DISTANCE_M, results["lofree_20dBm"], "-o", color="purple", linewidth=1.6, markevery=8, markersize=4, label="LO-free, PTx=20dBm")
ax.set_xscale("log")
ax.set_xlabel(r"Distance, $d_{Tx-Rx}$ (m)"); ax.set_ylabel("SNR (dB)")
ax.set_title(f"Fig. 6. SNR performance versus distance (d_probe={d_probe_mm:.2f}mm, independent rebuild)")
ax.legend(fontsize=8, loc="upper right"); ax.grid(alpha=0.3, which="both")
ax.set_xlim(0.1, 1e6)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "freshbuild_fig6_d036mm.png", dpi=220, bbox_inches="tight")
plt.close(fig)
print("Saved: freshbuild_fig6_d036mm.png")

report = f"""# Fig. 6 math report -- d=0.36mm independent rebuild

Identical method to `fig6_fresh_build/fig6_snr_vs_distance.py` (Conventional / Theoretical
LO-dressed / Practical LO-dressed / LO-free, same equations, same wide-QuTiP-sweep approach
for Practical LO-dressed) -- reusing THIS folder's own `freshbuild_fig4_d036mm.csv` threshold
and `freshbuild_fig5_d036mm.csv` LO-dressed data. ONLY d_probe changed from 0.76mm to 0.36mm.
Does not touch fig6_fresh_build or the older qutip_d036mm files.

## Result

    Fig.4-d036mm threshold used: {threshold_mhz:.4f} MHz
    kappa = {kappa_true:.6e} W/(rad/s), P0_bar = {P0_bar/1e-6:.4f} microW
    A_LO = {A_LO:.6e}, sigma_Ry_LO^2 = {sigma_Ry_LO_sq:.6e}
    Checkpoint @ d=100m, PTx=-10dBm: Conventional={conv_100:.4f}dB, Theoretical LO-dressed={theo_100:.4f}dB,
        gap={theo_100-conv_100:.4f}dB

## Comparison to d=0.76mm fresh_build

d=0.76mm fresh_build's real checkpoint gap = 28.8864dB (using the same GRx*GLNA-on-noise-only
Conventional formula, and the fRF=3.5GHz paper-stated value -- the fRF consistency fix from
`fig6_fresh_build/fig6_angle6_fRF_consistency_fix.py` is NOT applied here, for direct
apples-to-apples comparison with the main 0.76mm build's own real result). Conventional SNR is
diameter-independent (no d_probe dependence anywhere in its formula) -- any gap change here is
entirely due to the LO-dressed side's kappa/A_LO, which scale with probe diameter as shown in
`fig6_fresh_build/fig6_angle4_probe_diameter.py` (A_LO proportional to d_probe^4, confirmed
against real QuTiP data at both diameters).
"""
with open(OUTPUT_DIR / "freshbuild_fig6_d036mm_math_report.md", "w") as f:
    f.write(report)
print("Saved: freshbuild_fig6_d036mm_math_report.md")
print("\nDONE.")
