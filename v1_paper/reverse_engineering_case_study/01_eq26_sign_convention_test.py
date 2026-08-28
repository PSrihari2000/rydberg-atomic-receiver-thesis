# ============================================================
# *** CASE STUDY -- NOT a validated result ***
#
# TEST: what if we used the paper's Eq.26 literal sign convention for
# Im(rho21) instead of QuTiP's (physically-required-for-absorption)
# sign? Earlier finding: fresh QuTiP rho21 matches Eq.26's MAGNITUDE
# exactly but has the OPPOSITE sign everywhere (see fig5_fresh_build's
# own math report). This flips Im(rho21) -> -Im(rho21) and propagates
# through Fig.3/4/5/6.
#
# EXACT algebraic shortcut (no new QuTiP solves needed): since
#   Pout_orig = Pin*exp(-kp*L*C0*Im(rho21))
# flipping Im(rho21) -> -Im(rho21) gives
#   Pout_flipped = Pin*exp(+kp*L*C0*Im(rho21)) = Pin^2 / Pout_orig
# Verified below against fresh QuTiP spot-checks before trusting it.
#
# Does not touch any frozen fresh_build file.
# ============================================================

from pathlib import Path
import csv

import numpy as np
import qutip as qt
from scipy.signal import find_peaks
from scipy.optimize import brentq

OUTPUT_DIR = Path(__file__).resolve().parent
FIG3_NPZ = OUTPUT_DIR.parent / "fig3_fresh_build" / "fig3_quantum_response.npz"
FIG5_CSV = OUTPUT_DIR.parent / "fig5_fresh_build" / "fig5_data.csv"

print("=" * 70)
print("*** CASE STUDY 01 -- Eq.26 sign convention flip test ***")
print("=" * 70)

e_charge, a0, hbar, eps0, eta0 = 1.6e-19, 5.2e-11, 1.054571817e-34, 8.854e-12, 377.0
c_light = 2.998e8
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
N0 = 1.0e15
Pin = (np.pi/(2.0*eta0))*(d_probe*Omega_p*hbar/(2.0*np.sqrt(wp_12)))**2
C0 = -2.0*N0*wp_12/(eps0*hbar*Omega_p)
wp_RF_over_hbar = wp_RF/hbar
fp = c_light/lambda_p

def basis4(): return [qt.basis(4,i) for i in range(4)]
def hamiltonian_lofree(Omega_RF, Delta_c=0.0):
    k1,k2,k3,k4 = basis4()
    H = qt.qzero(4)
    H += -Delta_c*k3*k3.dag()
    H += -Delta_c*k4*k4.dag()
    H += (Omega_p/2.0)*(k1*k2.dag()+k2*k1.dag())
    H += (Omega_c/2.0)*(k2*k3.dag()+k3*k2.dag())
    H += (Omega_RF/2.0)*(k3*k4.dag()+k4*k3.dag())
    return H
def collapse_ops_lofree():
    k1,k2,k3,k4 = basis4()
    return [np.sqrt(gamma2)*k1*k2.dag(), np.sqrt(gamma3)*k2*k3.dag(), np.sqrt(gamma4)*k3*k4.dag()]
def pout_direct(Omega_RF, Delta_c):
    H = hamiltonian_lofree(Omega_RF, Delta_c)
    rho_ss = qt.steadystate(H, collapse_ops_lofree(), method="direct")
    rho21 = complex(rho_ss[1,0])
    chi = C0*rho21
    return Pin*np.exp(-kp_wave*L_cell*np.imag(chi)), np.imag(rho21)

print("\nSTEP 1: verify the flip identity Pout_flipped = Pin^2/Pout_orig against fresh QuTiP")
test_points = [(2*np.pi*6e6, 0.0), (2*np.pi*3e6, 2*np.pi*4e6)]
for Omega_RF, Delta_c in test_points:
    Pout_orig, Im_rho21_orig = pout_direct(Omega_RF, Delta_c)
    Pout_flipped_predicted = Pin**2 / Pout_orig
    # direct: recompute Pout using -Im(rho21) explicitly
    Pout_flipped_direct = Pin*np.exp(-kp_wave*L_cell*C0*(-Im_rho21_orig))
    reldiff = abs(Pout_flipped_direct-Pout_flipped_predicted)/Pout_flipped_direct
    print(f"  Omega_RF/2pi={Omega_RF/2/np.pi/1e6:.1f}MHz: Im(rho21)={Im_rho21_orig:.4e}, "
          f"Pout_orig={Pout_orig/1e-6:.4f}uW, flipped(direct)={Pout_flipped_direct/1e-6:.4f}uW, "
          f"flipped(predicted)={Pout_flipped_predicted/1e-6:.4f}uW, reldiff={reldiff:.2e}")

def flip_pout(Pout_old):
    return Pin**2 / Pout_old

# ------------------------------------------------------------
# STEP 2: Fig.3/Fig.4 with the flipped sign
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 2: Fig.3/Fig.4 with Eq.26-literal (flipped) sign")
print("=" * 70)

data = np.load(FIG3_NPZ)
delta_c_mhz = data["delta_c_mhz"]
omega_rf_mhz = data["omega_rf_mhz"]
Pout_surface_baseline = data["Pout_surface"]
Pout_surface_flipped = flip_pout(Pout_surface_baseline)

print(f"Baseline Pout range: [{Pout_surface_baseline.min()/1e-6:.4f},{Pout_surface_baseline.max()/1e-6:.4f}]uW")
print(f"Flipped  Pout range: [{Pout_surface_flipped.min()/1e-6:.4f},{Pout_surface_flipped.max()/1e-6:.4f}]uW")
print("(flipping the sign turns absorption dips into GAIN peaks -- Pout > Pin is now possible, "
      "physically requires an unexplained gain mechanism)")

def measure_threshold(Pout_uW_arr):
    row0 = Pout_uW_arr[0, :]
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
        row = Pout_uW_arr[i, :]
        idx, _ = find_peaks(row)
        if len(idx) < 2: continue
        heights = row[idx]
        top2 = np.sort(idx[np.argsort(heights)[-2:]])
        li, ri = top2
        separation = delta_c_mhz[ri]-delta_c_mhz[li]
        between = row[li:ri+1]
        vloc = np.argmin(between)
        if not (0 < vloc < (ri-li)): continue
        if separation >= FWHM: resolved[i] = True
    threshold = omega_rf_mhz[resolved].min() if np.any(resolved) else None
    return FWHM, threshold

FWHM_f, threshold_f = measure_threshold(Pout_surface_flipped/1e-6)
print(f"Flipped-sign Fig.4 threshold: {'FWHM='+str(round(FWHM_f,4))+'MHz, threshold='+str(round(threshold_f,4))+'MHz' if threshold_f else 'NO THRESHOLD FOUND (peak-finding likely fails on gain-shaped curve)'}")

# ------------------------------------------------------------
# STEP 3: Fig.5/Fig.6 with the flipped sign
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 3: Fig.5/Fig.6 with Eq.26-literal (flipped) sign")
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
fig5a_pout_W_flipped = flip_pout(fig5a_pout_W_baseline)

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

fit_flipped = find_linear_range(fig5a_omega_mhz, fig5a_pout_W_flipped, Omega_LO_mhz)
if fit_flipped is None:
    print("Flipped-sign Fig.5: NO linear region found")
else:
    print(f"Flipped-sign Fig.5: kappa={fit_flipped['slope']:.6e}W/MHz, "
          f"range=[{fit_flipped['x_low']:.4f},{fit_flipped['x_high']:.4f}]MHz, R2={fit_flipped['r2']:.6f}")

    GLNA_lin, RL, D_resp, T_temp, eta_eff, B_bw = 100.0, 50.0, 0.55, 290.0, 0.8, 1.0e6
    kB = 1.380649e-23
    GTx_lin = 10**(2.15/10)
    GRx_lin = GTx_lin
    sigma_BGN_sq = 1e-3*10**(-90/10)
    sigma_TN_sq = 4*kB*T_temp*B_bw
    fRF = 3.5e9
    lambda_RF = c_light/fRF

    kappa_true = fit_flipped["slope"]/(2*np.pi*1e6)
    P0_bar = float(np.interp(Omega_LO_mhz, fig5a_omega_mhz, fig5a_pout_W_flipped))
    Ibar = eta_eff*P0_bar*e_charge/(hbar*fp)
    sigma_PSN_sq = 2*e_charge*B_bw*Ibar
    A_LO = GLNA_lin*RL*D_resp**2*kappa_true**2*wp_RF_over_hbar**2
    sigma_Ry_LO_sq = A_LO*sigma_BGN_sq + GLNA_lin*RL*D_resp**2*sigma_PSN_sq + sigma_TN_sq

    def snr_conv_dB(d, PTx_W=1e-4):
        Pc = (PTx_W*GTx_lin/(4*np.pi*d**2)) * (lambda_RF**2/(4*np.pi))
        return 10*np.log10(Pc/(GRx_lin*GLNA_lin*sigma_BGN_sq + sigma_TN_sq))
    def snr_theo_dB(d, PTx_W=1e-4):
        PRx = PTx_W*GTx_lin*eta0/(4*np.pi*d**2)
        return 10*np.log10(A_LO*PRx/sigma_Ry_LO_sq)

    gap_100m = snr_theo_dB(100.0) - snr_conv_dB(100.0)
    print(f"Flipped-sign checkpoint gap @ d=100m,PTx=-10dBm: {gap_100m:.4f}dB "
          f"(baseline real result: 28.8864dB, paper claims ~44dB)")

print("\nDONE.")
