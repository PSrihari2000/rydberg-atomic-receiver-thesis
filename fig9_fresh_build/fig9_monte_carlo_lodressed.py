# ============================================================
# FIG. 9(b) -- GENUINE MONTE CARLO SYMBOL DETECTION, LO-dressed
#
# NEW file, does not touch any existing fig9_* file.
#
# HYPOTHESIS: the paper's Fig.9(b) shows SER *improving then worsening*
# with SNR -- not a plain S-curve. Our earlier fig9_ser_vs_snr_freshbuild.py
# cannot show this because it plugs a scalar SNR into the standard closed-
# form Q-function SER formula, which is provably monotonic. The paper's
# own text says LO-dressed/LO-free's distortion-region curves came from
# "QuTiP and Monte Carlo simulations" (Sec.V-C) -- this builds a genuine
# Monte Carlo symbol simulation instead of a formula lookup, to test
# whether the REAL nonlinear, possibly non-monotonic ORF -> |P(Delta_f)|
# mapping (via Fig.5's real curve) produces genuine detection ambiguity
# that a scalar-SNR formula can't capture -- which would explain the
# dip-then-worsen shape as a real, physical effect, not a plotting choice.
#
# METHOD (real, not approximated):
#   1. Fix M amplitude levels (M-PAM style, simplest case that isolates
#      the amplitude-detection-ambiguity mechanism cleanly).
#   2. At each power level (~ distance), the M transmit levels sit at M
#      different real Omega_RF values (linearly spaced fractions of the
#      peak Omega_RF at that operating point).
#   3. For EACH level, compute the REAL |P(Delta_f)| via the full
#      Omega_total(t) -> interpolate Fig.5's real curve -> Fourier
#      extraction pipeline (same as fig6_snr_vs_distance.py's Practical
#      LO-dressed branch -- reusing the same wide QuTiP sweep, no new
#      solves needed beyond that one sweep).
#   4. Add REAL Gaussian noise (variance = sigma_Ry,LO^2, the same real
#      noise floor used throughout) to the OBSERVED |P(Delta_f)|, many
#      trials.
#   5. Detect via NEAREST NEIGHBOR in the OBSERVED |P(Delta_f)| domain
#      (what a real receiver actually sees), not in Omega_RF domain.
#   6. Count REAL errors. This can show ambiguity-driven errors a
#      formula-based approach cannot, if the mapping is non-monotonic.
# ============================================================

from pathlib import Path
import csv
import time

import numpy as np
import qutip as qt
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

OUTPUT_DIR = Path(__file__).resolve().parent
FIG5_CSV = OUTPUT_DIR.parent / "fig5_fresh_build" / "fig5_data.csv"

print("=" * 70)
print("FIG. 9(b) -- genuine Monte Carlo symbol detection, LO-dressed")
print("=" * 70)

e_charge, a0, hbar, eps0, eta0 = 1.6e-19, 5.2e-11, 1.054571817e-34, 8.854e-12, 377.0
c_light = 2.998e8
Omega_p = 2.0*np.pi*8.0e6
Omega_c = 2.0*np.pi*1.0e6
gamma2 = 2.0*np.pi*5.2e6
wp_RF = -1443.459*e_charge*a0
wp_12 = (2.5*e_charge*a0)**2
lambda_p = 852e-9
kp_wave = 2.0*np.pi/lambda_p
d_probe = 0.76e-3
Pin = (np.pi/(2.0*eta0))*(d_probe*Omega_p*hbar/(2.0*np.sqrt(wp_12)))**2
N0, L_cell = 1.0e15, 1.0e-2
C0 = -2.0*N0*wp_12/(eps0*hbar*Omega_p)
wp_RF_over_hbar = wp_RF/hbar

with open(FIG5_CSV, newline="") as f:
    rows = list(csv.reader(f))
meta = {}
i = 0
while rows[i]:
    if rows[i][0] != "# METADATA":
        meta[rows[i][0]] = float(rows[i][1])
    i += 1
Omega_LO_mhz = meta["Omega_LO_operating_MHz"]
Omega_LO = 2.0*np.pi*Omega_LO_mhz*1e6
Delta_f = 150e3

# ------------------------------------------------------------
# Wide real QuTiP sweep (same physics as fig6_snr_vs_distance.py)
# ------------------------------------------------------------

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

WIDE_MAX_MHZ = 35.0
wide_omega_mhz = np.linspace(0.05, WIDE_MAX_MHZ, 350)
wide_pout_W = np.zeros_like(wide_omega_mhz)
t0 = time.time()
for idx, w in enumerate(wide_omega_mhz):
    wide_pout_W[idx] = pout_at_omega_total(2.0*np.pi*w*1e6)
print(f"{n_qutip_solves} fresh QuTiP solves, {time.time()-t0:.1f}s")
cubic_wide = CubicSpline(wide_omega_mhz, wide_pout_W)

# ------------------------------------------------------------
# Noise floor (same real formula as fig6_snr_vs_distance.py)
# ------------------------------------------------------------

GLNA_lin, RL, D_resp, T_temp, eta_eff, B_bw = 100.0, 50.0, 0.55, 290.0, 0.8, 1.0e6
kB = 1.380649e-23
fp = c_light/lambda_p
sigma_BGN_sq = 1e-3*10**(-90/10)
sigma_TN_sq = 4*kB*T_temp*B_bw
kappa_true = meta["linear_fit_slope_W_per_MHz"]/(2*np.pi*1e6)
P0_bar = float(cubic_wide(Omega_LO_mhz))
Ibar = eta_eff*P0_bar*e_charge/(hbar*fp)
sigma_PSN_sq = 2*e_charge*B_bw*Ibar
A_LO = GLNA_lin*RL*D_resp**2*kappa_true**2*wp_RF_over_hbar**2
sigma_Ry_LO_sq = A_LO*sigma_BGN_sq + GLNA_lin*RL*D_resp**2*sigma_PSN_sq + sigma_TN_sq
print(f"sigma_Ry_LO^2 = {sigma_Ry_LO_sq:.6e} (same real noise floor as fig6_snr_vs_distance.py)")

# ------------------------------------------------------------
# REAL nonlinear Omega_RF -> |P(Delta_f)| mapping (the actual physics)
# ------------------------------------------------------------

N_PERIOD = 512
t_grid = np.linspace(0.0, 1.0/Delta_f, N_PERIOD, endpoint=False)
psi_grid = 2.0*np.pi*Delta_f*t_grid

def P_delta_f_real(Omega_RF):
    """The REAL, possibly non-monotonic mapping -- full nonlinear pipeline."""
    Omega_total = np.abs(Omega_LO + Omega_RF*np.exp(1j*psi_grid))
    Omega_total_mhz = Omega_total/(2*np.pi)/1e6
    if Omega_total_mhz.max() > WIDE_MAX_MHZ or Omega_total_mhz.min() < wide_omega_mhz.min():
        return np.nan
    Pout_t = cubic_wide(Omega_total_mhz)
    return (2.0/N_PERIOD)*np.abs(np.sum(Pout_t*np.exp(-1j*psi_grid)))

# ------------------------------------------------------------
# Check: is the mapping monotonic? (the actual mechanism being tested)
# ------------------------------------------------------------

print("\nChecking whether Omega_RF -> |P(Delta_f)| is monotonic across a wide range...")
test_omega_rf_mhz = np.linspace(0.1, 30.0, 200)
test_P = np.array([P_delta_f_real(2*np.pi*w*1e6) for w in test_omega_rf_mhz])
valid = ~np.isnan(test_P)
diffs = np.diff(test_P[valid])
n_nonmonotonic = np.sum(diffs < 0)  # since kappa<0, |P(df)| should generally increase with Omega_RF if monotonic... check sign
print(f"  |P(Delta_f)| range: [{test_P[valid].min():.4e},{test_P[valid].max():.4e}]")
print(f"  Number of sign-changes in d|P(df)|/dOmega_RF: "
      f"{np.sum(np.diff(np.sign(diffs)) != 0)} (0 = perfectly monotonic)")
peak_idx = np.argmax(test_P[valid])
print(f"  Peak |P(Delta_f)| at Omega_RF/2pi={test_omega_rf_mhz[valid][peak_idx]:.4f}MHz -- "
      f"{'MAPPING IS NON-MONOTONIC (genuine ambiguity beyond this point)' if peak_idx < np.sum(valid)-5 else 'roughly monotonic over this range'}")

# ------------------------------------------------------------
# MONTE CARLO M-PAM SYMBOL DETECTION over a genuine power sweep
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("MONTE CARLO M-PAM detection, real nonlinear channel")
print("=" * 70)

M = 4  # log2M=2, matches the smallest order tested in fig9_ser_vs_snr_freshbuild.py
N_TRIALS = 3000
rng = np.random.default_rng(42)

# Sweep the OVERALL scale of Omega_RF (proxy for transmit power / distance);
# at each scale, the M levels are equally spaced fractions of that scale.
scale_mhz_grid = np.logspace(np.log10(0.002), np.log10(30.0), 60)

results_snr_dB = []
results_ser_mc = []
results_ser_formula = []

for scale_mhz in scale_mhz_grid:
    levels_mhz = np.linspace(scale_mhz/M, scale_mhz, M)  # M equally-spaced real amplitude levels
    levels_omega_rf = 2*np.pi*levels_mhz*1e6
    P_levels = np.array([P_delta_f_real(w) for w in levels_omega_rf])
    if np.any(np.isnan(P_levels)):
        continue
    signal_power_avg = np.mean(GLNA_lin*RL*D_resp**2*P_levels**2)
    snr_dB = 10*np.log10(signal_power_avg/sigma_Ry_LO_sq)

    # Monte Carlo: for each trial, pick a random true level, add real noise
    # to the OBSERVED photocurrent-power-equivalent quantity, and detect via
    # nearest neighbor among the REAL P_levels
    noise_std_in_P = np.sqrt(sigma_Ry_LO_sq / (GLNA_lin*RL*D_resp**2))  # noise std in |P(df)| units
    true_idx = rng.integers(0, M, N_TRIALS)
    observed_P = P_levels[true_idx] + rng.normal(0, noise_std_in_P, N_TRIALS)
    detected_idx = np.argmin(np.abs(observed_P[:, None] - P_levels[None, :]), axis=1)
    ser_mc = np.mean(detected_idx != true_idx)

    # Formula-based comparison (Proakis M-PAM, using the AVERAGE snr_dB as if the
    # channel were a clean linear AWGN channel -- what fig9_ser_vs_snr_freshbuild.py does)
    from scipy.special import erfc
    def Q(x): return 0.5*erfc(x/np.sqrt(2))
    ser_formula = 2.0*(1.0-1.0/M) * Q(np.sqrt(6.0*10**(snr_dB/10)/(M**2-1.0)))

    results_snr_dB.append(snr_dB)
    results_ser_mc.append(ser_mc)
    results_ser_formula.append(ser_formula)

results_snr_dB = np.array(results_snr_dB)
results_ser_mc = np.array(results_ser_mc)
results_ser_formula = np.array(results_ser_formula)

print(f"\n{'scale_mhz':>10} {'SNR(dB)':>10} {'SER_montecarlo':>16} {'SER_formula':>13}")
for i in range(0, len(results_snr_dB), 3):
    print(f"{'':>10} {results_snr_dB[i]:10.4f} {results_ser_mc[i]:16.4e} {results_ser_formula[i]:13.4e}")

# Does SER_mc show improve-then-worsen (non-monotonic) behavior vs SNR?
order = np.argsort(results_snr_dB)
ser_mc_sorted = results_ser_mc[order]
snr_sorted = results_snr_dB[order]
min_idx = np.argmin(ser_mc_sorted)
print(f"\nMinimum Monte Carlo SER = {ser_mc_sorted[min_idx]:.4e} at SNR={snr_sorted[min_idx]:.4f}dB")
worsens_after = np.any(ser_mc_sorted[min_idx+1:] > ser_mc_sorted[min_idx]*2) if min_idx < len(ser_mc_sorted)-3 else False
print(f"SER increases again (>2x the minimum) at higher SNR after that point (N_TRIALS={N_TRIALS}): {worsens_after}")
print("CAUTION -- this was checked against a HIGHER-trial-count re-run (50000 trials, dense scan)")
print("and did NOT reproduce: zero errors everywhere in that denser check. With only 3000 trials")
print("per point here, an apparent 'jump' is very likely small-sample noise, not a real effect.")
print("Treat any single-run 'worsens_after=True' from this script with that caveat -- verified")
print("separately with more trials before reporting as a real finding.")

# ------------------------------------------------------------
# PLOT
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8,6.5))
ax.semilogy(snr_sorted, ser_mc_sorted, "o-", color="purple", markersize=4, label="Monte Carlo (real nonlinear channel)")
ax.semilogy(results_snr_dB[order], results_ser_formula[order], "--", color="gray", label="Formula (scalar SNR -> Q-function)")
ax.set_xlabel("SNR (dB)")
ax.set_ylabel("SER")
ax.set_title(f"LO-dressed M-PAM (M={M}): genuine Monte Carlo vs formula-based SER")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "fig9_monte_carlo_lodressed.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("\nSaved: fig9_monte_carlo_lodressed.png")
print("\nDONE.")
