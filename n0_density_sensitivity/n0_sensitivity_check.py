# ============================================================
# N0 (ATOM DENSITY) SENSITIVITY CHECK
#
# Paper states N0 = 1e-15 m^-3 (literally, an impossible negative
# exponent) -- this project's established fix is a SIGN correction to
# +1e15 m^-3, used throughout fig3/4/5/6_fresh_build. Real comparable
# literature (Jing et al. 2020, whose parameters this paper's own Sec.IV
# borrows) reports ~3.5e16 m^-3 for a similar Cs vapor cell -- ~35x
# higher than what's used here.
#
# KEY FACT (proven below, not assumed): N0 enters ONLY through
# C0 = -2*N0*wp_12/(eps0*hbar*Omega_p), a purely classical constant
# applied AFTER the quantum steady-state rho21 is solved. rho21 itself
# (from the Hamiltonian/Lindblad structure) never involves N0 -- so
# changing N0 does NOT require re-running QuTiP. Since
#   Pout = Pin * exp(-kp*L*C0*Im(rho21))
# and C0 is proportional to N0, this gives an EXACT algebraic identity:
#   Pout_new = Pin * (Pout_old/Pin)^(N0_new/N0_old)
# allowing every already-computed real Pout value (Fig.3's full grid,
# Fig.5's real curve) to be rescaled to any N0 without new solves.
# Verified below against genuine fresh QuTiP spot-checks before trusting
# it for the full grids.
#
# NEW folder, does not touch any fig3/4/5/6_fresh_build file.
# ============================================================

from pathlib import Path
import csv

import numpy as np
import qutip as qt
from scipy.signal import find_peaks
from scipy.interpolate import CubicSpline

OUTPUT_DIR = Path(__file__).resolve().parent
FIG3_NPZ = OUTPUT_DIR.parent / "fig3_fresh_build" / "fig3_quantum_response.npz"
FIG5_CSV = OUTPUT_DIR.parent / "fig5_fresh_build" / "fig5_data.csv"
FIG6_CSV = OUTPUT_DIR.parent / "fig6_fresh_build" / "fig6_data.csv"

print("=" * 70)
print("N0 SENSITIVITY CHECK")
print("=" * 70)

e_charge, a0, hbar, eps0, eta0 = 1.6e-19, 5.2e-11, 1.054571817e-34, 8.854e-12, 377.0
L_cell = 1.0e-2
Omega_p = 2.0*np.pi*8.0e6
Omega_c = 2.0*np.pi*1.0e6
gamma2 = 2.0*np.pi*5.2e6
gamma3 = 2.0*np.pi*3.9e3
gamma4 = 2.0*np.pi*1.7e3
wp_RF = -1443.459*e_charge*a0
wp_12 = (2.5*e_charge*a0)**2
lambda_p = 852e-9
kp_wave = 2.0*np.pi/lambda_p
d_probe = 0.76e-3
Pin = (np.pi/(2.0*eta0))*(d_probe*Omega_p*hbar/(2.0*np.sqrt(wp_12)))**2

N0_BASELINE = 1.0e15
N0_CANDIDATES = [1.0e16, 3.5e16]   # Jing et al. 2020's own value

def C0_of(N0): return -2.0*N0*wp_12/(eps0*hbar*Omega_p)

# ------------------------------------------------------------
# STEP 1: VERIFY the exact rescaling identity against fresh QuTiP spot-checks
# ------------------------------------------------------------

def basis4(): return [qt.basis(4,i) for i in range(4)]
def hamiltonian(Omega_RF, Delta_c=0.0):
    k1,k2,k3,k4 = basis4()
    H = qt.qzero(4)
    H += -Delta_c*k3*k3.dag()
    H += -Delta_c*k4*k4.dag()
    H += (Omega_p/2.0)*(k1*k2.dag()+k2*k1.dag())
    H += (Omega_c/2.0)*(k2*k3.dag()+k3*k2.dag())
    H += (Omega_RF/2.0)*(k3*k4.dag()+k4*k3.dag())
    return H
def collapse_ops():
    k1,k2,k3,k4 = basis4()
    return [np.sqrt(gamma2)*k1*k2.dag(), np.sqrt(gamma3)*k2*k3.dag(), np.sqrt(gamma4)*k3*k4.dag()]

def pout_direct(Omega_RF, Delta_c, N0):
    H = hamiltonian(Omega_RF, Delta_c)
    rho_ss = qt.steadystate(H, collapse_ops(), method="direct")
    rho21 = complex(rho_ss[1,0])
    chi = C0_of(N0)*rho21
    return Pin*np.exp(-kp_wave*L_cell*np.imag(chi))

print("\nSTEP 1: verifying the exact rescaling identity against fresh QuTiP spot-checks")
test_points = [(2*np.pi*6e6, 0.0), (2*np.pi*3e6, 2*np.pi*4e6), (2*np.pi*10e6, -2*np.pi*3e6)]
max_reldiff = 0.0
for Omega_RF, Delta_c in test_points:
    Pout_old_direct = pout_direct(Omega_RF, Delta_c, N0_BASELINE)
    Pout_new_direct = pout_direct(Omega_RF, Delta_c, 3.5e16)   # genuine fresh QuTiP solve at the new N0
    Pout_new_predicted = Pin * (Pout_old_direct/Pin) ** (3.5e16/N0_BASELINE)
    reldiff = abs(Pout_new_direct - Pout_new_predicted) / Pout_new_direct
    max_reldiff = max(max_reldiff, reldiff)
    print(f"  Omega_RF/2pi={Omega_RF/2/np.pi/1e6:.1f}MHz, Delta_c/2pi={Delta_c/2/np.pi/1e6:.1f}MHz: "
          f"direct={Pout_new_direct/1e-6:.6f}uW, predicted={Pout_new_predicted/1e-6:.6f}uW, "
          f"reldiff={reldiff:.2e}")
print(f"Max relative difference: {max_reldiff:.2e} ({'CONFIRMED exact' if max_reldiff < 1e-9 else 'WARNING: not exact'})")

# ------------------------------------------------------------
# STEP 2: rescale Fig.3's REAL grid, redo Fig.4's threshold analysis
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 2: Fig.3/Fig.4 -- rescaled from real data, no new QuTiP solves")
print("=" * 70)

data = np.load(FIG3_NPZ)
delta_c_mhz = data["delta_c_mhz"]
omega_rf_mhz = data["omega_rf_mhz"]
Pout_surface_baseline = data["Pout_surface"]

def rescale_pout(Pout_old, N0_new, N0_old=N0_BASELINE):
    return Pin * (Pout_old/Pin) ** (N0_new/N0_old)

def measure_threshold(Pout_surface_uW_arr):
    row0 = Pout_surface_uW_arr[0, :]
    peak_idx0 = int(np.argmax(row0))
    peak_pos0 = delta_c_mhz[peak_idx0]
    peak_val0 = row0[peak_idx0]
    baseline0 = 0.5*(row0[0]+row0[-1])
    half_level0 = baseline0 + 0.5*(peak_val0-baseline0)
    def interp_crossing(x, y, level, i0, direction):
        i = i0; n = len(y)
        while 0 <= i+direction < n:
            y0, y1 = y[i], y[i+direction]
            if (y0-level)*(y1-level) <= 0 and y0 != y1:
                frac = (level-y0)/(y1-y0)
                return x[i] + frac*(x[i+direction]-x[i])
            i += direction
        return None
    right_cross = interp_crossing(delta_c_mhz, row0, half_level0, peak_idx0, +1)
    left_cross = interp_crossing(delta_c_mhz, row0, half_level0, peak_idx0, -1)
    if right_cross is None or left_cross is None:
        return None, None
    FWHM = (right_cross-peak_pos0) + (peak_pos0-left_cross)

    resolved = np.zeros(len(omega_rf_mhz), dtype=bool)
    for i in range(len(omega_rf_mhz)):
        row = Pout_surface_uW_arr[i, :]
        idx, _ = find_peaks(row)
        if len(idx) < 2:
            continue
        heights = row[idx]
        top2 = np.sort(idx[np.argsort(heights)[-2:]])
        li, ri = top2
        separation = delta_c_mhz[ri]-delta_c_mhz[li]
        between = row[li:ri+1]
        vloc = np.argmin(between)
        if not (0 < vloc < (ri-li)):
            continue
        if separation >= FWHM:
            resolved[i] = True
    threshold = omega_rf_mhz[resolved].min() if np.any(resolved) else None
    return FWHM, threshold

FWHM_base, threshold_base = measure_threshold(Pout_surface_baseline/1e-6)
print(f"N0=1.0e15 (baseline): FWHM={FWHM_base:.4f}MHz, threshold={threshold_base:.4f}MHz "
      f"(paper claims ~5.5MHz)")

fig4_results = {}
for N0_new in N0_CANDIDATES:
    Pout_rescaled = rescale_pout(Pout_surface_baseline, N0_new)
    FWHM_new, threshold_new = measure_threshold(Pout_rescaled/1e-6)
    fig4_results[N0_new] = dict(FWHM=FWHM_new, threshold=threshold_new)
    print(f"N0={N0_new:.2e}: FWHM={FWHM_new:.4f}MHz, threshold={threshold_new:.4f}MHz "
          f"(vs baseline {threshold_base:.4f}MHz, vs paper's ~5.5MHz)")

# ------------------------------------------------------------
# STEP 3: rescale Fig.5's REAL curve, redo linear-range fit
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 3: Fig.5 -- rescaled from real data, no new QuTiP solves")
print("=" * 70)

with open(FIG5_CSV, newline="") as f:
    rows = list(csv.reader(f))
meta = {}
i = 0
while rows[i]:
    if rows[i][0] != "# METADATA":
        meta[rows[i][0]] = float(rows[i][1])
    i += 1
i += 3
fig5a_omega_mhz, fig5a_pout_uW = [], []
while i < len(rows) and rows[i]:
    fig5a_omega_mhz.append(float(rows[i][0]))
    fig5a_pout_uW.append(float(rows[i][1]))
    i += 1
fig5a_omega_mhz = np.array(fig5a_omega_mhz)
fig5a_pout_W_baseline = np.array(fig5a_pout_uW)*1e-6
Omega_LO_mhz = meta["Omega_LO_operating_MHz"]

def linear_fit_score(x, y):
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope*x+intercept
    ss_res = np.sum((y-y_pred)**2); ss_tot = np.sum((y-np.mean(y))**2)
    return slope, intercept, (1.0-ss_res/ss_tot if ss_tot>0 else 1.0)

def find_linear_range(x_vals, y_vals, x_anchor, r2_threshold=0.998):
    center = int(np.argmin(np.abs(x_vals-x_anchor)))
    best=None
    max_radius = min(center, len(x_vals)-center-1)
    for radius in range(3, max_radius+1):
        left,right = center-radius, center+radius+1
        slope,intercept,r2 = linear_fit_score(x_vals[left:right], y_vals[left:right])
        if r2>=r2_threshold:
            best=dict(slope=slope, x_low=x_vals[left], x_high=x_vals[right-1], r2=r2)
        else:
            break
    return best

fit_base = find_linear_range(fig5a_omega_mhz, fig5a_pout_W_baseline, Omega_LO_mhz)
print(f"N0=1.0e15 (baseline): kappa={fit_base['slope']:.6e}W/MHz, "
      f"range=[{fit_base['x_low']:.4f},{fit_base['x_high']:.4f}]MHz")

fig5_results = {}
for N0_new in N0_CANDIDATES:
    pout_rescaled = rescale_pout(fig5a_pout_W_baseline, N0_new)
    fit_new = find_linear_range(fig5a_omega_mhz, pout_rescaled, Omega_LO_mhz)
    P0_bar_new = float(np.interp(Omega_LO_mhz, fig5a_omega_mhz, pout_rescaled))
    if fit_new is None:
        print(f"N0={N0_new:.2e}: NO linear region found (R2>=0.998 fails even at the smallest "
              f"3-point window -- the curve is too steep/nonlinear near Omega_LO at this N0), "
              f"P0_bar={P0_bar_new/1e-6:.4f}uW")
        fig5_results[N0_new] = None
        continue
    fit_new["P0_bar"] = P0_bar_new
    fig5_results[N0_new] = fit_new
    print(f"N0={N0_new:.2e}: kappa={fit_new['slope']:.6e}W/MHz, "
          f"range=[{fit_new['x_low']:.4f},{fit_new['x_high']:.4f}]MHz, P0_bar={P0_bar_new/1e-6:.4f}uW")

# ------------------------------------------------------------
# STEP 4: propagate to Fig.6's checkpoint gap
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 4: Fig.6 checkpoint gap (d=100m, PTx=-10dBm)")
print("=" * 70)

GLNA_lin, RL, D_resp, T_temp, eta_eff, B_bw = 100.0, 50.0, 0.55, 290.0, 0.8, 1.0e6
kB = 1.380649e-23
c_light = 2.998e8
fp = c_light/lambda_p
GTx_lin = 10**(2.15/10)
sigma_BGN_sq = 1e-3*10**(-90/10)
sigma_TN_sq = 4*kB*T_temp*B_bw
wp_RF_over_hbar = wp_RF/hbar

with open(FIG6_CSV, newline="") as f:
    rows6 = list(csv.reader(f))
conv_100m = None
for row in rows6:
    if row and row[0] == "100.000000":
        conv_100m = float(row[1])  # SNR_conventional_dB_PTx-10dBm column
        break
print(f"Conventional SNR @ d=100m, PTx=-10dBm (unaffected by N0): {conv_100m:.4f}dB (from fig6_data.csv)")

def theo_snr_at_100m(kappa_slope, P0_bar):
    kappa_true = kappa_slope/(2*np.pi*1e6)
    Ibar = eta_eff*P0_bar*e_charge/(hbar*fp)
    sigma_PSN_sq = 2*e_charge*B_bw*Ibar
    A_LO = GLNA_lin*RL*D_resp**2*kappa_true**2*wp_RF_over_hbar**2
    sigma_Ry_LO_sq = A_LO*sigma_BGN_sq + GLNA_lin*RL*D_resp**2*sigma_PSN_sq + sigma_TN_sq
    PRx = 1e-4*GTx_lin*eta0/(4*np.pi*100.0**2)
    return 10*np.log10(A_LO*PRx/sigma_Ry_LO_sq)

P0_bar_base = float(np.interp(Omega_LO_mhz, fig5a_omega_mhz, fig5a_pout_W_baseline))
theo_base = theo_snr_at_100m(fit_base["slope"], P0_bar_base)
gap_base = theo_base - conv_100m
print(f"\nN0=1.0e15 (baseline): Theoretical LO-dressed={theo_base:.4f}dB, gap={gap_base:.4f}dB "
      f"(paper claims ~44dB)")

for N0_new in N0_CANDIDATES:
    if fig5_results[N0_new] is None:
        print(f"N0={N0_new:.2e}: cannot compute Theoretical LO-dressed SNR -- no linear region "
              f"(hence no kappa) exists at this N0 under the same R2 criterion")
        continue
    theo_new = theo_snr_at_100m(fig5_results[N0_new]["slope"], fig5_results[N0_new]["P0_bar"])
    gap_new = theo_new - conv_100m
    print(f"N0={N0_new:.2e}: Theoretical LO-dressed={theo_new:.4f}dB, gap={gap_new:.4f}dB "
          f"(change vs baseline: {gap_new-gap_base:+.4f}dB)")

print("\nDONE.")
