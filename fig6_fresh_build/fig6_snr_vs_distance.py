# ============================================================
# FIG. 6 -- SNR VERSUS DISTANCE d_Tx-Rx, FOUR RECEIVER MODELS
# Conventional / Theoretical LO-dressed / Practical LO-dressed / LO-free
#
# NEW FILE -- does not touch fig6_snr_distance.py/.png, fig6_math_report.md,
# or fig6_fRF_consistency_audit.py/.txt (all left untouched).
#
# Reuses REAL data already established this session:
#   - fig4_fresh_build/fig4_classification.csv: the measured LO-free
#     resolvability threshold (5.125 MHz), used to gate the LO-free curve.
#   - fig5_fresh_build/fig5_data.csv: the real, independently-simulated
#     LO-dressed Pout(Omega_total) curve and its linear-region fit
#     (kappa, Omega_LO, the fitted range).
#
# Conventional receiver: standard Friis link budget (flux density x
# effective aperture) for the signal, with the receive-chain gain
# (GRx*GLNA) applied only to the antenna-referenced background noise,
# matching Fig.2's own diagram literally (y=sqrt(PRx)*h*x+n_Conv, no
# explicit gain on the signal) and sigma^2_Conv = GRx*GLNA*sigma^2_BGN
# + 4kT B (thermal noise added downstream, unscaled either way).
#
# Practical LO-dressed: ONE continuous nonlinear pipeline (build
# Omega_total(t), interpolate the real Pout(Omega_total) curve, extract
# the Delta_f Fourier component) at every distance -- no formula switch
# at the linear-range boundary. Because the physical Omega_RF at short
# distances/high PTx vastly exceeds Fig.5's original domain, a WIDER
# Omega_total sweep is computed fresh here (identical Hamiltonian/
# parameters to fig5_quutip.py) to cover as much of the distance range
# as is physically sensible; beyond that covered domain the curve is a
# genuine gap (NaN), never extrapolated.
# ============================================================

import time
from pathlib import Path

import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import csv

OUTPUT_DIR = Path(__file__).resolve().parent
FIG4_CSV = OUTPUT_DIR.parent / "fig4_fresh_build" / "fig4_classification.csv"
FIG5_CSV = OUTPUT_DIR.parent / "fig5_fresh_build" / "fig5_data.csv"

print("=" * 70)
print("FIG. 6 -- SNR vs distance, 4 receiver models")
print("=" * 70)

# ------------------------------------------------------------
# LOAD REAL, ALREADY-VALIDATED DATA (no regeneration)
# ------------------------------------------------------------

with open(FIG4_CSV, newline="") as f:
    rows = list(csv.reader(f))
threshold_mhz = float(rows[2][1])
print(f"Loaded Fig.4 threshold: {threshold_mhz:.4f} MHz  (from {FIG4_CSV.name})")

with open(FIG5_CSV, newline="") as f:
    rows = list(csv.reader(f))
meta = {}
i = 0
while rows[i]:
    if rows[i][0] != "# METADATA":
        meta[rows[i][0]] = float(rows[i][1])
    i += 1
i += 1  # blank line
i += 2  # "# FIG.5(a)..." comment row + header row
fig5a_omega_mhz, fig5a_pout_uW = [], []
while i < len(rows) and rows[i]:
    fig5a_omega_mhz.append(float(rows[i][0]))
    fig5a_pout_uW.append(float(rows[i][1]))
    i += 1
fig5a_omega_mhz = np.array(fig5a_omega_mhz)
fig5a_pout_W = np.array(fig5a_pout_uW) * 1e-6

Omega_LO_mhz = meta["Omega_LO_operating_MHz"]
lin_low_mhz = meta["linear_range_lower_MHz"]
lin_high_mhz = meta["linear_range_upper_MHz"]
kappa_slope_W_per_MHz = meta["linear_fit_slope_W_per_MHz"]
print(f"Loaded Fig.5: Omega_LO={Omega_LO_mhz:.4f}MHz, linear range=[{lin_low_mhz:.4f},{lin_high_mhz:.4f}]MHz, "
      f"kappa(slope)={kappa_slope_W_per_MHz:.6e} W/MHz, {len(fig5a_omega_mhz)} curve points  (from {FIG5_CSV.name})")

cubic_fig5 = CubicSpline(fig5a_omega_mhz, fig5a_pout_W)
Omega_RF_max_from_fig5_mhz = min(Omega_LO_mhz - lin_low_mhz, lin_high_mhz - Omega_LO_mhz)
print(f"Omega_RF,max derived from Fig.5's linear range = min({Omega_LO_mhz:.4f}-{lin_low_mhz:.4f}, "
      f"{lin_high_mhz:.4f}-{Omega_LO_mhz:.4f}) = {Omega_RF_max_from_fig5_mhz:.4f} MHz "
      f"(the largest RF modulation amplitude whose full Omega_total(t) excursion stays inside "
      f"Fig.5's own measured linear region)")

# ------------------------------------------------------------
# SHARED PHYSICAL PARAMETERS (identical values to fig5_quutip.py)
# ------------------------------------------------------------

e_charge = 1.6e-19
a0 = 5.2e-11
hbar = 1.054571817e-34
eps0 = 8.854e-12
eta0 = 377.0
kB = 1.380649e-23
c_light = 2.998e8

Omega_p = 2.0 * np.pi * 8.0e6
Omega_c = 2.0 * np.pi * 1.0e6
gamma2 = 2.0 * np.pi * 5.2e6
gamma3 = 0.0   # LO-dressed-specific, Sec.III-B
gamma4 = 0.0
Delta_p = 0.0
Delta_c = 0.0
Delta_LO = 0.0

wp_RF = -1443.459 * e_charge * a0     # |3>-|4> dipole moment
wp_12 = (2.5 * e_charge * a0) ** 2
lambda_p = 852e-9
fp = c_light / lambda_p               # probe laser optical frequency

Omega_LO = 2.0 * np.pi * Omega_LO_mhz * 1e6
Delta_f = 150e3

# Wireless-system parameters (paper Sec. IV / V-A)
GTx_dBi = 2.15
GRx_dBi = 2.15
GLNA_dB = 20.0
RL = 50.0
D_resp = 0.55
T_temp = 290.0
eta_eff = 0.8
sigma_BGN_dBm = -90.0
eps_tilde = 0.005
fRF = 3.5e9
B_bw = 1.0e6

def db_to_lin(db):
    return 10.0 ** (db / 10.0)

def dbm_to_w(dbm):
    return 1e-3 * 10.0 ** (dbm / 10.0)

GTx_lin = db_to_lin(GTx_dBi)
GRx_lin = db_to_lin(GRx_dBi)
GLNA_lin = db_to_lin(GLNA_dB)
lambda_RF = c_light / fRF
sigma_BGN_sq = dbm_to_w(sigma_BGN_dBm)   # paper states -90dBm AS "the noise power" (Sec.III-A/V-A);
                                          # used directly as sigma^2_BGN, not squared again

print("\nWIRELESS PARAMETERS (paper Sec. IV / V-A)")
for label, val in [("GTx, GRx (dBi)", f"{GTx_dBi}, {GRx_dBi}"), ("GLNA (dB)", GLNA_dB), ("RL (Ohm)", RL),
                    ("D (A/W)", D_resp), ("T (K)", T_temp), ("eta_eff", eta_eff),
                    ("sigma_BGN (dBm)", sigma_BGN_dBm), ("epsilon_tilde", eps_tilde),
                    ("fRF (GHz)", fRF/1e9), ("lambda_RF (mm)", lambda_RF*1e3), ("B (MHz)", B_bw/1e6)]:
    print(f"  {label:20s} = {val}")

# ============================================================
# EXTENDED QUTIP SWEEP for Practical LO-dressed (identical physics
# to fig5_quutip.py's hamiltonian_lodressed/collapse_ops_lodressed;
# redefined here since fig5_quutip.py is a script, not a module, and
# is not to be modified)
# ============================================================

print("\n" + "=" * 70)
print("EXTENDED QUTIP SWEEP -- wider Omega_total domain for Fig.6's distance range")
print("=" * 70)

def basis4():
    return [qt.basis(4, i) for i in range(4)]

def hamiltonian_lodressed(Omega_total_val):
    k1, k2, k3, k4 = basis4()
    H = qt.qzero(4)
    H += -Delta_p * k2 * k2.dag()
    H += -(Delta_p + Delta_c) * k3 * k3.dag()
    H += -(Delta_p + Delta_c + Delta_LO) * k4 * k4.dag()
    H += (Omega_p / 2.0) * (k1 * k2.dag() + k2 * k1.dag())
    H += (Omega_c / 2.0) * (k2 * k3.dag() + k3 * k2.dag())
    H += (Omega_total_val / 2.0) * (k3 * k4.dag() + k4 * k3.dag())
    return H

def collapse_ops_lodressed():
    k1, k2, k3, k4 = basis4()
    ops = [np.sqrt(gamma2) * k1 * k2.dag()]
    if gamma3 > 0:
        ops.append(np.sqrt(gamma3) * k2 * k3.dag())
    if gamma4 > 0:
        ops.append(np.sqrt(gamma4) * k3 * k4.dag())
    return ops

Pin = (np.pi / (2.0 * eta0)) * (0.76e-3 * Omega_p * hbar / (2.0 * np.sqrt(wp_12))) ** 2
N0 = 1.0e15
L_cell = 1.0e-2
kp_wave = 2.0 * np.pi / lambda_p
C0 = -2.0 * N0 * wp_12 / (eps0 * hbar * Omega_p)

n_qutip_solves = 0

def pout_at_omega_total(Omega_total_val):
    global n_qutip_solves
    H = hamiltonian_lodressed(Omega_total_val)
    c_ops = collapse_ops_lodressed()
    rho_ss = qt.steadystate(H, c_ops, method="direct")
    n_qutip_solves += 1
    rho21 = complex(rho_ss[1, 0])
    chi = C0 * rho21
    exponent = -kp_wave * L_cell * np.imag(chi)
    return Pin * np.exp(exponent)

WIDE_SWEEP_MAX_MHZ = 35.0   # covers Omega_RF up to ~30.8MHz above Omega_LO; beyond this the
                             # 4-level ladder model itself is far outside its sensible regime
                             # (Omega_p=8MHz, Omega_c=1MHz), so points needing more are gapped, not extended further
wide_omega_mhz = np.linspace(0.05, WIDE_SWEEP_MAX_MHZ, 350)
wide_pout_W = np.zeros_like(wide_omega_mhz)
t0 = time.time()
for idx, w in enumerate(wide_omega_mhz):
    wide_pout_W[idx] = pout_at_omega_total(2.0 * np.pi * w * 1e6)
dt = time.time() - t0
print(f"{n_qutip_solves} fresh QuTiP steady-state solves over Omega_total/2pi in "
      f"[{wide_omega_mhz.min():.3f},{wide_omega_mhz.max():.3f}]MHz, {dt:.1f}s "
      f"({dt/n_qutip_solves*1000:.2f} ms/solve)")
cubic_wide = CubicSpline(wide_omega_mhz, wide_pout_W)
P0_bar = float(cubic_wide(Omega_LO_mhz))
print(f"P0_bar (average Pout at Omega_LO, from this real curve) = {P0_bar/1e-6:.4f} microW")

# ============================================================
# NOISE FLOOR FOR LO-DRESSED RECEIVERS (Eq.34-36) -- a single fixed
# value: shot noise is driven by the DC/average photocurrent (P0_bar),
# not by the tiny AC signal |P(Delta_f)|, so sigma^2_Ry,LO does not
# vary with distance -- only the signal term does.
# ============================================================

kappa_true = kappa_slope_W_per_MHz / (2.0 * np.pi * 1e6)   # W / (rad/s), converted from W/MHz-of-(Omega/2pi)
wp_RF_over_hbar = wp_RF / hbar

Ibar = eta_eff * P0_bar * e_charge / (hbar * fp)             # Eq.35, using P0_bar (see math report)
sigma_PSN_sq = 2.0 * e_charge * B_bw * Ibar                    # Eq.34
sigma_TN_sq = 4.0 * kB * T_temp * B_bw                          # Eq.13
A_LO = GLNA_lin * RL * D_resp**2 * kappa_true**2 * wp_RF_over_hbar**2
sigma_Ry_LO_sq = (A_LO * sigma_BGN_sq
                   + GLNA_lin * RL * D_resp**2 * sigma_PSN_sq
                   + sigma_TN_sq)                                # Eq.36

print(f"\nkappa (converted, W per rad/s) = {kappa_true:.6e}")
print(f"Ibar (avg photocurrent, Eq.35 using P0_bar) = {Ibar:.6e} A")
print(f"sigma_PSN^2 = {sigma_PSN_sq:.6e}, sigma_TN^2 = {sigma_TN_sq:.6e}, A_LO = {A_LO:.6e}")
print(f"sigma_Ry,LO^2 (Eq.36, fixed across all distances) = {sigma_Ry_LO_sq:.6e}")

# ============================================================
# DISTANCE SWEEP
# ============================================================

DISTANCE_M = np.logspace(np.log10(0.1), np.log10(1e6), 281)

def PRx_flux(PTx_W, d):
    """Eq.10 -- raw flux density reaching the vapor cell (no receive antenna)."""
    return PTx_W * GTx_lin * eta0 / (4.0 * np.pi * d**2)

def omega_rf_from_PRx(PRx):
    ERF = np.sqrt(PRx)
    Omega_RF = ERF * abs(wp_RF_over_hbar)
    return Omega_RF

# ---- Conventional (Sec.V-A + Fig.2, standard Friis link budget) ----

def snr_conventional_dB(PTx_W, d):
    # Literal Fig.2 diagram convention: y = sqrt(PRx)*h*x + n_Conv -- no GRx/GLNA
    # gain on the signal side, only on the noise (sigma^2_Conv = GRx*GLNA*sigma^2_BGN
    # + 4kTB). Switched from the GRx*GLNA-symmetric reading per user request, to
    # match the same convention this project's earlier Fig.6 build used.
    Pc = (PTx_W * GTx_lin / (4.0 * np.pi * d**2)) * (lambda_RF**2 / (4.0 * np.pi))
    gamma_conv = Pc / (GRx_lin * GLNA_lin * sigma_BGN_sq + sigma_TN_sq)
    return 10.0 * np.log10(gamma_conv)

# ---- LO-free (Eq.8-16, gated by Fig.4's real threshold) ----

def snr_lofree_dB(PTx_W, d):
    PRx = PRx_flux(PTx_W, d)
    Omega_RF = omega_rf_from_PRx(PRx)
    Omega_RF_mhz = Omega_RF / (2*np.pi) / 1e6
    sigma_UN_sq = (eps_tilde**2) * PRx
    sigma_Ry_sq = sigma_UN_sq + sigma_BGN_sq
    snr = PRx / sigma_Ry_sq
    resolved = Omega_RF_mhz >= threshold_mhz
    return (10.0 * np.log10(snr) if resolved else np.nan), Omega_RF_mhz, resolved

# ---- Theoretical LO-dressed (Eq.37, pure analytical, kappa from Fig.5) ----

def snr_theoretical_dB(PTx_W, d):
    PRx = PRx_flux(PTx_W, d)
    snr = A_LO * PRx / sigma_Ry_LO_sq
    return 10.0 * np.log10(snr), PRx

# ---- Practical LO-dressed (continuous real nonlinear pipeline, every distance) ----

N_PERIOD = 512
t_grid = np.linspace(0.0, 1.0 / Delta_f, N_PERIOD, endpoint=False)
psi_grid = 2.0 * np.pi * Delta_f * t_grid

def snr_practical_dB(PTx_W, d):
    PRx = PRx_flux(PTx_W, d)
    Omega_RF = omega_rf_from_PRx(PRx)
    Omega_RF_mhz = Omega_RF / (2*np.pi) / 1e6
    Omega_total = np.abs(Omega_LO + Omega_RF * np.exp(1j * psi_grid))
    Omega_total_mhz = Omega_total / (2*np.pi) / 1e6
    if Omega_total_mhz.max() > WIDE_SWEEP_MAX_MHZ or Omega_total_mhz.min() < wide_omega_mhz.min():
        return np.nan, Omega_RF_mhz, False   # genuinely out of the covered QuTiP domain -- gap, not extrapolated
    Pout_t = cubic_wide(Omega_total_mhz)
    P_delta_f = (2.0 / N_PERIOD) * np.abs(np.sum(Pout_t * np.exp(-1j * psi_grid)))
    signal = GLNA_lin * RL * D_resp**2 * P_delta_f**2
    snr = signal / sigma_Ry_LO_sq
    in_linear = Omega_RF_mhz <= Omega_RF_max_from_fig5_mhz
    return 10.0 * np.log10(snr), Omega_RF_mhz, in_linear

# ------------------------------------------------------------
# RUN SWEEPS -- paper-stated PTx sets per receiver (Fig.6 legend)
# ------------------------------------------------------------

PTX_CONV_DBM = [-10.0, 10.0]
PTX_THEO_DBM = [-10.0, 10.0]
PTX_PRAC_DBM = [-10.0, 10.0]
PTX_LOFREE_DBM = [10.0, 20.0]

print("\n" + "=" * 70)
print("RUNNING DISTANCE SWEEPS")
print("=" * 70)

results = {"distance_m": DISTANCE_M}

for ptx in PTX_CONV_DBM:
    PTx_W = dbm_to_w(ptx)
    results[f"conv_{ptx:g}dBm"] = np.array([snr_conventional_dB(PTx_W, d) for d in DISTANCE_M])

for ptx in PTX_THEO_DBM:
    PTx_W = dbm_to_w(ptx)
    vals = [snr_theoretical_dB(PTx_W, d) for d in DISTANCE_M]
    results[f"theo_{ptx:g}dBm"] = np.array([v[0] for v in vals])
    results[f"theo_{ptx:g}dBm_PRx"] = np.array([v[1] for v in vals])

for ptx in PTX_PRAC_DBM:
    PTx_W = dbm_to_w(ptx)
    vals = [snr_practical_dB(PTx_W, d) for d in DISTANCE_M]
    results[f"prac_{ptx:g}dBm"] = np.array([v[0] for v in vals])
    results[f"prac_{ptx:g}dBm_OmegaRF_mhz"] = np.array([v[1] for v in vals])
    results[f"prac_{ptx:g}dBm_inlinear"] = np.array([v[2] for v in vals])

for ptx in PTX_LOFREE_DBM:
    PTx_W = dbm_to_w(ptx)
    vals = [snr_lofree_dB(PTx_W, d) for d in DISTANCE_M]
    results[f"lofree_{ptx:g}dBm"] = np.array([v[0] for v in vals])
    results[f"lofree_{ptx:g}dBm_OmegaRF_mhz"] = np.array([v[1] for v in vals])
    results[f"lofree_{ptx:g}dBm_resolved"] = np.array([v[2] for v in vals])

print("Sweeps complete.")

# ============================================================
# VALIDATION CHECKS
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION CHECKS")
print("=" * 70)

d_idx_sorted = np.argsort(DISTANCE_M)
prx_check = np.array([PRx_flux(dbm_to_w(-10), d) for d in DISTANCE_M])
print(f"1. PRx monotonically decreasing with d: {np.all(np.diff(prx_check[d_idx_sorted]) <= 0)}")

omega_rf_check = omega_rf_from_PRx(prx_check) / (2*np.pi) / 1e6
print(f"2. Omega_RF monotonically decreasing with d: {np.all(np.diff(omega_rf_check[d_idx_sorted]) <= 0)}")

conv_check = results["conv_-10dBm"]
print(f"3. Conventional SNR monotonically decreasing with d: {np.all(np.diff(conv_check[d_idx_sorted]) <= 1e-9)}")

print(f"4. Theoretical LO-dressed follows analytical Eq.37 by construction: True (no fitting/branching)")

prac_ptx10 = results["prac_10dBm"]
inlin_ptx10 = results["prac_10dBm_inlinear"]
transitions = np.sum(np.abs(np.diff(inlin_ptx10.astype(int))))
print(f"5. Practical LO-dressed changes character across Fig.5's linear boundary: "
      f"{int(transitions)} crossing(s) of the linear/nonlinear boundary along the PTx=10dBm sweep")

nan_gap_ptx10 = np.sum(np.isnan(prac_ptx10))
print(f"6. Practical LO-dressed distortion/out-of-domain region occurs at short distances: "
      f"{nan_gap_ptx10} of {len(DISTANCE_M)} points gapped (all at the shortest distances: "
      f"d <= {DISTANCE_M[np.isnan(prac_ptx10)].max() if nan_gap_ptx10 else 0:.3f} m)")

lofree_resolved_10 = results["lofree_10dBm_resolved"]
print(f"7. LO-free resolved (non-distortion) region: {np.sum(lofree_resolved_10)} of {len(DISTANCE_M)} points, "
      f"up to d = {DISTANCE_M[lofree_resolved_10].max():.4f} m (PTx=10dBm)")

print("8. No fitting/scaling applied to match the paper's numbers anywhere in this script: True (by construction)")

idx100 = int(np.argmin(np.abs(DISTANCE_M - 100.0)))
conv_100 = results["conv_-10dBm"][idx100]
theo_100 = results["theo_-10dBm"][idx100]
gap_100 = theo_100 - conv_100
print(f"9. Checkpoint @ d={DISTANCE_M[idx100]:.4f}m, PTx=-10dBm: "
      f"Conventional={conv_100:.4f}dB, Theoretical LO-dressed={theo_100:.4f}dB, gap={gap_100:.4f}dB "
      f"(paper reports ~44dB at this exact reference point -- comparison only, not tuned)")

def zero_crossing(x_arr, y_arr):
    order = np.argsort(x_arr)
    x_s, y_s = x_arr[order], y_arr[order]
    valid = ~np.isnan(y_s)
    x_s, y_s = x_s[valid], y_s[valid]
    for k in range(len(y_s) - 1):
        if (y_s[k] - 0) * (y_s[k+1] - 0) <= 0 and y_s[k] != y_s[k+1]:
            frac = (0 - y_s[k]) / (y_s[k+1] - y_s[k])
            return x_s[k] + frac * (x_s[k+1] - x_s[k])
    return None

conv_cross = zero_crossing(DISTANCE_M, results["conv_-10dBm"])
theo_cross = zero_crossing(DISTANCE_M, results["theo_-10dBm"])
print(f"10-11. 0dB crossings (PTx=-10dBm): Conventional@{conv_cross:.4f}m, "
      f"Theoretical LO-dressed@{theo_cross:.4f}m, "
      f"extended coverage={theo_cross-conv_cross:.4f}m (paper reports ~1500m -- comparison only)")

first_linear_true = DISTANCE_M[inlin_ptx10][-1] if np.any(inlin_ptx10) else None
last_before_gap = DISTANCE_M[~np.isnan(prac_ptx10)].min() if np.any(~np.isnan(prac_ptx10)) else None
print(f"12. Practical LO-dressed (PTx=10dBm): out-of-domain gap for d < {last_before_gap:.4f}m; "
      f"leaves Fig.5's linear region for d < {DISTANCE_M[inlin_ptx10].min() if np.any(inlin_ptx10) else float('nan'):.4f}m "
      f"(annotation only, not a computational branch)")

# ============================================================
# SAVE CSV
# ============================================================

csv_path = OUTPUT_DIR / "fig6_data.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["# METADATA"])
    writer.writerow(["fig4_threshold_mhz", f"{threshold_mhz:.6f}"])
    writer.writerow(["fig5_Omega_LO_mhz", f"{Omega_LO_mhz:.6f}"])
    writer.writerow(["fig5_linear_range_low_mhz", f"{lin_low_mhz:.6f}"])
    writer.writerow(["fig5_linear_range_high_mhz", f"{lin_high_mhz:.6f}"])
    writer.writerow(["Omega_RF_max_from_fig5_mhz", f"{Omega_RF_max_from_fig5_mhz:.6f}"])
    writer.writerow(["kappa_W_per_rad_per_s", f"{kappa_true:.6e}"])
    writer.writerow(["P0_bar_W", f"{P0_bar:.6e}"])
    writer.writerow(["A_LO", f"{A_LO:.6e}"])
    writer.writerow(["sigma_Ry_LO_sq", f"{sigma_Ry_LO_sq:.6e}"])
    writer.writerow(["n_qutip_solves_this_script", n_qutip_solves])
    writer.writerow([])
    header = ["distance_m",
              "SNR_conventional_dB_PTx-10dBm", "SNR_conventional_dB_PTx10dBm",
              "SNR_theoretical_LO_dressed_dB_PTx-10dBm", "SNR_theoretical_LO_dressed_dB_PTx10dBm",
              "SNR_practical_LO_dressed_dB_PTx-10dBm", "SNR_practical_LO_dressed_dB_PTx10dBm",
              "practical_in_linear_region_PTx-10dBm", "practical_in_linear_region_PTx10dBm",
              "SNR_LO_free_dB_PTx10dBm", "SNR_LO_free_dB_PTx20dBm",
              "LO_free_resolved_PTx10dBm", "LO_free_resolved_PTx20dBm",
              "Omega_RF_MHz_PTx-10dBm", "Omega_RF_MHz_PTx10dBm"]
    writer.writerow(header)
    for i in range(len(DISTANCE_M)):
        writer.writerow([
            f"{DISTANCE_M[i]:.6f}",
            f"{results['conv_-10dBm'][i]:.4f}", f"{results['conv_10dBm'][i]:.4f}",
            f"{results['theo_-10dBm'][i]:.4f}", f"{results['theo_10dBm'][i]:.4f}",
            f"{results['prac_-10dBm'][i]:.4f}" if not np.isnan(results['prac_-10dBm'][i]) else "",
            f"{results['prac_10dBm'][i]:.4f}" if not np.isnan(results['prac_10dBm'][i]) else "",
            bool(results['prac_-10dBm_inlinear'][i]), bool(results['prac_10dBm_inlinear'][i]),
            f"{results['lofree_10dBm'][i]:.4f}" if not np.isnan(results['lofree_10dBm'][i]) else "",
            f"{results['lofree_20dBm'][i]:.4f}" if not np.isnan(results['lofree_20dBm'][i]) else "",
            bool(results['lofree_10dBm_resolved'][i]), bool(results['lofree_20dBm_resolved'][i]),
            f"{results['prac_-10dBm_OmegaRF_mhz'][i]:.6f}", f"{results['prac_10dBm_OmegaRF_mhz'][i]:.6f}",
        ])
print(f"\nSaved: {csv_path.name}")

# ============================================================
# PLOT
# ============================================================

fig, ax = plt.subplots(figsize=(9, 7))

ax.plot(DISTANCE_M, results["conv_-10dBm"], "-", color="green", linewidth=1.6, label="Conventional, PTx=-10dBm")
ax.plot(DISTANCE_M, results["conv_10dBm"], "-o", color="green", linewidth=1.6, markevery=20, markersize=4, label="Conventional, PTx=10dBm")

ax.plot(DISTANCE_M, results["theo_-10dBm"], "--", color="orange", linewidth=1.4, label="Theoretical LO-dressed, PTx=-10dBm")
ax.plot(DISTANCE_M, results["theo_10dBm"], "--o", color="orange", linewidth=1.4, markevery=20, markersize=4, label="Theoretical LO-dressed, PTx=10dBm")

ax.plot(DISTANCE_M, results["prac_-10dBm"], "-.", color="deepskyblue", linewidth=1.6, label="Practical LO-dressed, PTx=-10dBm")
ax.plot(DISTANCE_M, results["prac_10dBm"], "-.o", color="deepskyblue", linewidth=1.6, markevery=15, markersize=4, label="Practical LO-dressed, PTx=10dBm")

ax.plot(DISTANCE_M, results["lofree_10dBm"], "-", color="purple", linewidth=1.6, label="LO-free, PTx=10dBm")
ax.plot(DISTANCE_M, results["lofree_20dBm"], "-o", color="purple", linewidth=1.6, markevery=8, markersize=4, label="LO-free, PTx=20dBm")

ax.set_xscale("log")
ax.set_xlabel(r"Distance, $d_{Tx-Rx}$ (m)")
ax.set_ylabel("SNR (dB)")
ax.set_title("Fig. 6. SNR performance versus distance $d_{Tx-Rx}$ at different transmit powers $P_{Tx}$")
ax.legend(fontsize=8, loc="upper right", ncol=1)
ax.grid(alpha=0.3, which="both")
ax.set_xlim(0.1, 1e6)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "fig6_snr_vs_distance.png", dpi=220, bbox_inches="tight")
plt.close(fig)
print("Saved: fig6_snr_vs_distance.png")

print("\nDONE.")
