# ============================================================
# EXPLORATORY -- does the probe beam diameter (0.76mm, paper-stated)
# affect Fig.6's LO-dressed SNR / the gap vs Conventional?
#
# NEW file, does not touch any other fig6 file. Fig.4's threshold was
# already shown to be diameter-independent (qutip_d036mm/, separate
# from this fig6 work). This checks the DOWNSTREAM wireless-system
# consequence for Fig.6: Pin (hence P0_bar, kappa) scales with
# d_probe^2 -- since kappa enters A_LO squared, A_LO should scale as
# d_probe^4, a real, checkable prediction. Runs a genuinely fresh
# LO-dressed QuTiP sweep at d_probe=0.36mm (identical Hamiltonian to
# fig5_quutip.py, only d_probe changed) to get real kappa/P0_bar there.
# ============================================================

from pathlib import Path
import time

import numpy as np
import qutip as qt

OUTPUT_DIR = Path(__file__).resolve().parent

print("=" * 70)
print("ANGLE 4: probe diameter dependence of Fig.6's LO-dressed SNR")
print("=" * 70)

e_charge, a0, hbar, eps0, eta0 = 1.6e-19, 5.2e-11, 1.054571817e-34, 8.854e-12, 377.0
Omega_p = 2*np.pi*8.0e6
Omega_c = 2*np.pi*1.0e6
gamma2 = 2*np.pi*5.2e6
wp_RF = -1443.459*e_charge*a0
wp_12 = (2.5*e_charge*a0)**2
lambda_p = 852e-9
kp_wave = 2*np.pi/lambda_p
N0, L_cell = 1.0e15, 1.0e-2
Omega_LO_mhz = 4.23
Omega_LO = 2*np.pi*Omega_LO_mhz*1e6
C0 = -2.0*N0*wp_12/(eps0*hbar*Omega_p)

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

def pin_for_diameter(d_probe):
    return (np.pi/(2.0*eta0))*(d_probe*Omega_p*hbar/(2.0*np.sqrt(wp_12)))**2

def run_sweep(d_probe, n_points=91, span_mhz=6.5):
    Pin = pin_for_diameter(d_probe)
    grid_mhz = np.linspace(Omega_LO_mhz-span_mhz, Omega_LO_mhz+span_mhz, n_points)
    pout_w = np.zeros_like(grid_mhz)
    t0=time.time()
    for i,w in enumerate(grid_mhz):
        H = hamiltonian_lodressed(2*np.pi*w*1e6)
        rho_ss = qt.steadystate(H, collapse_ops_lodressed(), method="direct")
        rho21 = complex(rho_ss[1,0])
        chi = C0*rho21
        pout_w[i] = Pin*np.exp(-kp_wave*L_cell*np.imag(chi))
    print(f"  d_probe={d_probe*1e3:.2f}mm: {n_points} QuTiP solves, {time.time()-t0:.1f}s, Pin={Pin/1e-6:.4f}uW")
    return grid_mhz, pout_w

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
            best=dict(slope=slope, intercept=intercept, r2=r2, x_low=x_vals[left], x_high=x_vals[right-1])
        else:
            break
    return best

print("\nRunning fresh LO-dressed sweeps at both diameters (identical physics, only d_probe differs)...")
grid_076, pout_076 = run_sweep(0.76e-3)
grid_036, pout_036 = run_sweep(0.36e-3)

fit_076 = find_linear_range(grid_076, pout_076, Omega_LO_mhz)
fit_036 = find_linear_range(grid_036, pout_036, Omega_LO_mhz)

kappa_076 = fit_076["slope"]/(2*np.pi*1e6)
kappa_036 = fit_036["slope"]/(2*np.pi*1e6)
P0_076 = float(np.interp(Omega_LO_mhz, grid_076, pout_076))
P0_036 = float(np.interp(Omega_LO_mhz, grid_036, pout_036))

print(f"\nd=0.76mm: kappa={kappa_076:.6e} W/(rad/s), P0_bar={P0_076/1e-6:.4f}uW, R2={fit_076['r2']:.6f}")
print(f"d=0.36mm: kappa={kappa_036:.6e} W/(rad/s), P0_bar={P0_036/1e-6:.4f}uW, R2={fit_036['r2']:.6f}")
print(f"\nkappa ratio (036/076) = {kappa_036/kappa_076:.4f}  vs predicted (0.36/0.76)^2 = {(0.36/0.76)**2:.4f}")
print(f"P0_bar ratio (036/076) = {P0_036/P0_076:.4f}  vs predicted (0.36/0.76)^2 = {(0.36/0.76)**2:.4f}")

# ------------------------------------------------------------
# Propagate to the SNR checkpoint
# ------------------------------------------------------------
GLNA_lin, RL, D_resp, T_temp, eta_eff, B_bw = 100.0, 50.0, 0.55, 290.0, 0.8, 1.0e6
wp_RF_over_hbar = wp_RF/hbar
kB = 1.380649e-23
sigma_BGN_sq = 1e-12
sigma_TN_sq = 4*kB*T_temp*B_bw
fp = 2.998e8/852e-9
GTx_lin = 10**(2.15/10)
eta0_link = 377.0
def dbm_to_w(dbm): return 1e-3*10**(dbm/10)

def snr_theoretical_checkpoint(kappa_true, P0_bar, d=100.0, PTx_dBm=-10.0):
    Ibar = eta_eff*P0_bar*e_charge/(hbar*fp)
    sigma_PSN_sq = 2*e_charge*B_bw*Ibar
    A_LO = GLNA_lin*RL*D_resp**2*kappa_true**2*wp_RF_over_hbar**2
    sigma_Ry_LO_sq = A_LO*sigma_BGN_sq + GLNA_lin*RL*D_resp**2*sigma_PSN_sq + sigma_TN_sq
    PRx = dbm_to_w(PTx_dBm)*GTx_lin*eta0_link/(4*np.pi*d**2)
    return 10*np.log10(A_LO*PRx/sigma_Ry_LO_sq), A_LO, sigma_Ry_LO_sq

snr_076, A_LO_076, sRy_076 = snr_theoretical_checkpoint(kappa_076, P0_076)
snr_036, A_LO_036, sRy_036 = snr_theoretical_checkpoint(kappa_036, P0_036)

print(f"\nA_LO(0.76mm)={A_LO_076:.4e}, A_LO(0.36mm)={A_LO_036:.4e}, "
      f"ratio={A_LO_036/A_LO_076:.4f} vs predicted (0.36/0.76)^4={(0.36/0.76)**4:.4f}")
print(f"sigma_Ry_LO^2(0.76mm)={sRy_076:.4e}, sigma_Ry_LO^2(0.36mm)={sRy_036:.4e}")
print(f"\nTheoretical LO-dressed SNR @ d=100m, PTx=-10dBm:")
print(f"  d_probe=0.76mm (paper's stated value): {snr_076:.4f} dB")
print(f"  d_probe=0.36mm (alternate):            {snr_036:.4f} dB")
print(f"  Difference: {snr_036-snr_076:.4f} dB")

conv_at_100m = -23.3293  # from the main build's real run, distance/probe-diameter-independent
print(f"\nConventional (unaffected by probe diameter) = {conv_at_100m:.4f} dB")
print(f"Gap at d_probe=0.76mm: {snr_076-conv_at_100m:.4f} dB  (main build's real result)")
print(f"Gap at d_probe=0.36mm: {snr_036-conv_at_100m:.4f} dB")
print("\nDONE.")
