# ============================================================
# FIG. 5 -- d=0.36mm INDEPENDENT REBUILD -- LO-DRESSED RECEIVER
# Identical method to fig5_fresh_build/fig5_quutip.py (fresh, independent
# QuTiP sweep over Omega_total, gamma3=gamma4=0 and all detunings=0 per
# Sec.III-B's LO-dressed-specific approximation, own linear dynamic range
# fit) -- ONLY d_probe changed from 0.76mm to 0.36mm. Does not touch
# fig5_fresh_build or the older qutip_d036mm files.
# ============================================================

import time
from pathlib import Path
import csv

import numpy as np
import qutip as qt
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

OUTPUT_DIR = Path(__file__).resolve().parent

e_charge, a0, hbar, eps0, eta0 = 1.6e-19, 5.2e-11, 1.054571817e-34, 8.854e-12, 377.0
L, N0 = 1.0e-2, 1.0e15
Omega_p = 2.0*np.pi*8.0e6
Omega_c = 2.0*np.pi*1.0e6
gamma2 = 2.0*np.pi*5.2e6
gamma3, gamma4 = 0.0, 0.0
Delta_p = Delta_c = Delta_LO = 0.0
wp_RF = -1443.459*e_charge*a0
wp_12 = (2.5*e_charge*a0)**2
lambda_p = 852e-9
kp = 2.0*np.pi/lambda_p
d_probe = 0.36e-3   # ONLY parameter changed from fig5_fresh_build

Pin = (np.pi/(2.0*eta0)) * (d_probe*Omega_p*hbar/(2.0*np.sqrt(wp_12)))**2
C0 = -2.0*N0*wp_12/(eps0*hbar*Omega_p)

Omega_LO = 2.0*np.pi*4.23e6
Delta_f = 150e3
Delta_phi = 0.0
illustrative_Omega_RF_mhz = [1.0, 3.0, 5.0]
Omega_LO_mhz = 4.23

print("=" * 70)
print(f"FIG. 5 -- d={d_probe*1e3:.2f}mm INDEPENDENT REBUILD -- LO-DRESSED")
print("=" * 70)
print(f"Pin = {Pin/1e-6:.4f} microW, C0={C0:.4e}")

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
    return Pin*np.exp(-kp*L*np.imag(chi))

sweep_max_mhz = Omega_LO_mhz + max(illustrative_Omega_RF_mhz) + 2.0
omega_total_sweep_mhz = np.linspace(0.05, sweep_max_mhz, 160)
Pout_a = np.zeros_like(omega_total_sweep_mhz)
t0 = time.time()
for i, w in enumerate(omega_total_sweep_mhz):
    Pout_a[i] = pout_at_omega_total(2.0*np.pi*w*1e6)
dt = time.time()-t0
print(f"Panel(a) sweep: {n_qutip_solves} solves in {dt:.1f}s, range [{omega_total_sweep_mhz.min():.3f},"
      f"{omega_total_sweep_mhz.max():.3f}]MHz")

cubic = CubicSpline(omega_total_sweep_mhz, Pout_a)

def linear_fit_score(x, y):
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope*x+intercept
    ss_res = np.sum((y-y_pred)**2); ss_tot = np.sum((y-np.mean(y))**2)
    return slope, intercept, (1.0-ss_res/ss_tot if ss_tot>0 else 1.0)

def find_linear_range(x_vals, y_vals, x_anchor, r2_threshold):
    center = int(np.argmin(np.abs(x_vals-x_anchor)))
    best=None
    max_radius = min(center, len(x_vals)-center-1)
    for radius in range(3, max_radius+1):
        left,right = center-radius, center+radius+1
        slope,intercept,r2 = linear_fit_score(x_vals[left:right], y_vals[left:right])
        if r2>=r2_threshold:
            best=dict(left=left,right=right,slope=slope,intercept=intercept,r2=r2,
                       x_low=x_vals[left], x_high=x_vals[right-1])
        else:
            break
    return best

R2_THRESHOLD = 0.998
fit = find_linear_range(omega_total_sweep_mhz, Pout_a, Omega_LO_mhz, R2_THRESHOLD)
y_low = fit["slope"]*fit["x_low"]+fit["intercept"]
y_high = fit["slope"]*fit["x_high"]+fit["intercept"]
print(f"Linear dynamic range: [{fit['x_low']:.4f},{fit['x_high']:.4f}]MHz, "
      f"slope={fit['slope']:.6e}W/MHz, R2={fit['r2']:.6f}")

period_s = 1.0/Delta_f
t_seconds = np.linspace(0.0, period_s, 4000)
panel_b = {}
for w_rf in illustrative_Omega_RF_mhz:
    Omega_RF_val = 2.0*np.pi*w_rf*1e6
    psi = 2.0*np.pi*Delta_f*t_seconds + Delta_phi
    Omega_complex = Omega_LO + Omega_RF_val*np.exp(1j*psi)
    Omega_total = np.abs(Omega_complex)
    panel_b[w_rf] = dict(Omega_total=Omega_total)

panel_c = {}
for w_rf in illustrative_Omega_RF_mhz:
    ot_mhz = panel_b[w_rf]["Omega_total"]/(2*np.pi)/1e6
    panel_c[w_rf] = cubic(ot_mhz)

Pout_at_LO = float(cubic(Omega_LO_mhz))
lo_cross = {}
for w_rf in illustrative_Omega_RF_mhz:
    Omega_RF_val = 2.0*np.pi*w_rf*1e6
    ratio = -Omega_RF_val/(2.0*Omega_LO)
    if abs(ratio) > 1:
        lo_cross[w_rf] = None
        continue
    psi0 = np.arccos(ratio); psi1 = 2*np.pi-psi0
    t0c = (psi0-Delta_phi)/(2*np.pi*Delta_f); t1c = (psi1-Delta_phi)/(2*np.pi*Delta_f)
    lo_cross[w_rf] = sorted([t0c % period_s, t1c % period_s])

extrema = {}
for w_rf in illustrative_Omega_RF_mhz:
    Omega_RF_val = 2.0*np.pi*w_rf*1e6
    ot_max_mhz = (Omega_LO+Omega_RF_val)/(2*np.pi)/1e6
    ot_min_mhz = abs(Omega_LO-Omega_RF_val)/(2*np.pi)/1e6
    extrema[w_rf] = dict(t_max=0.0, t_min=period_s/2.0, ot_max_mhz=ot_max_mhz, ot_min_mhz=ot_min_mhz,
                          pout_at_max=float(cubic(ot_max_mhz)), pout_at_min=float(cubic(ot_min_mhz)))

csv_path = OUTPUT_DIR / "freshbuild_fig5_d036mm.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["# METADATA"])
    writer.writerow(["d_probe_mm", f"{d_probe*1e3:.4f}"])
    writer.writerow(["Omega_LO_operating_MHz", f"{Omega_LO_mhz:.6f}"])
    writer.writerow(["linear_range_lower_MHz", f"{fit['x_low']:.6f}"])
    writer.writerow(["linear_range_upper_MHz", f"{fit['x_high']:.6f}"])
    writer.writerow(["linear_fit_slope_W_per_MHz", f"{fit['slope']:.6e}"])
    writer.writerow(["linear_fit_R2", f"{fit['r2']:.6f}"])
    writer.writerow(["n_qutip_solves", n_qutip_solves])
    writer.writerow([])
    writer.writerow(["# FIG.5(a): Omega_total_MHz, Pout_uW"])
    writer.writerow(["Omega_total_MHz", "Pout_uW"])
    for i in range(len(omega_total_sweep_mhz)):
        writer.writerow([f"{omega_total_sweep_mhz[i]:.4f}", f"{Pout_a[i]/1e-6:.6f}"])
    writer.writerow([])
    writer.writerow(["# FIG.5(b)/(c): time, case_Omega_RF_MHz, Omega_total_MHz, Pout_uW"])
    writer.writerow(["time_s", "case_Omega_RF_MHz", "Omega_total_MHz", "Pout_uW"])
    for w_rf in illustrative_Omega_RF_mhz:
        ot_mhz = panel_b[w_rf]["Omega_total"]/(2*np.pi)/1e6
        for i in range(len(t_seconds)):
            writer.writerow([f"{t_seconds[i]:.6e}", w_rf, f"{ot_mhz[i]:.6f}", f"{panel_c[w_rf][i]/1e-6:.6f}"])
print(f"Saved: {csv_path.name}")

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
ax_a, ax_c = axes[0,0], axes[0,1]
ax_b, ax_flow = axes[1,0], axes[1,1]
colors = {1.0: "red", 3.0: "green", 5.0: "blue"}

ax_a.plot(omega_total_sweep_mhz, Pout_a/1e-6, color="gray", linewidth=1.8, label="Obtained by QuTiP")
fit_x = omega_total_sweep_mhz[fit["left"]:fit["right"]]
fit_y = (fit["slope"]*fit_x+fit["intercept"])/1e-6
ax_a.plot(fit_x, fit_y, color="orange", linewidth=2.5, label=f"Linear fit (R2={fit['r2']:.4f})")
ax_a.axvline(Omega_LO_mhz, color="cyan", linestyle="--", linewidth=1.3)
ax_a.plot(Omega_LO_mhz, Pout_at_LO/1e-6, "o", color="cyan", markersize=7, zorder=5)
for w_rf in illustrative_Omega_RF_mhz:
    c = colors[w_rf]
    ax_a.plot(extrema[w_rf]["ot_max_mhz"], extrema[w_rf]["pout_at_max"]/1e-6, "o", color=c, markersize=6, zorder=5)
    ax_a.plot(extrema[w_rf]["ot_min_mhz"], extrema[w_rf]["pout_at_min"]/1e-6, "o", color=c, markersize=6, zorder=5)
ax_a.set_xlabel(r"$\Omega_{total}/2\pi$ (MHz)"); ax_a.set_ylabel(r"$P_{out}$ ($\mu$W)")
ax_a.set_title("Fig. 5(a)"); ax_a.legend(fontsize=8); ax_a.grid(alpha=0.3)
ax_a.set_xlim(0, omega_total_sweep_mhz.max())

for w_rf in illustrative_Omega_RF_mhz:
    c = colors[w_rf]
    ot_mhz = panel_b[w_rf]["Omega_total"]/(2*np.pi)/1e6
    ax_b.plot(ot_mhz, t_seconds/1e-6, color=c, linestyle="--", linewidth=1.2,
              label=r"$\Omega_{RF}/2\pi=$"+f"{w_rf} MHz")
ax_b.axvline(Omega_LO_mhz, color="cyan", linestyle="--", linewidth=1.3)
ax_b.set_xlabel(r"$|\Omega_{total}|/2\pi$ (MHz)"); ax_b.set_ylabel(r"t ($\mu$s)")
ax_b.set_title("Fig. 5(b)"); ax_b.legend(fontsize=8); ax_b.grid(alpha=0.3)
ax_b.set_xlim(0, omega_total_sweep_mhz.max()); ax_b.set_ylim(0, period_s/1e-6)

for w_rf in illustrative_Omega_RF_mhz:
    c = colors[w_rf]
    ax_c.plot(t_seconds/1e-6, panel_c[w_rf]/1e-6, color=c, linewidth=1.4,
              label=r"$\Omega_{RF}/2\pi=$"+f"{w_rf} MHz")
ax_c.axhline(Pout_at_LO/1e-6, color="cyan", linestyle=":", linewidth=1.0, alpha=0.7)
ax_c.set_xlabel(r"t ($\mu$s)"); ax_c.set_ylabel(r"$P_{out}$ ($\mu$W)")
ax_c.set_title("Fig. 5(c)"); ax_c.legend(fontsize=8); ax_c.grid(alpha=0.3)
ax_c.set_xlim(0, period_s/1e-6)

ax_flow.axis("off"); ax_flow.set_xlim(0,1); ax_flow.set_ylim(0,1)
from matplotlib.patches import FancyArrowPatch
flow_arrow = FancyArrowPatch((0.12,0.08),(0.88,0.62), connectionstyle="angle,angleA=90,angleB=0,rad=0",
                              arrowstyle="-|>", mutation_scale=35, linewidth=28, color="#c9c9f2", alpha=0.9, zorder=1)
ax_flow.add_patch(flow_arrow)
ax_flow.text(0.12,0.68,"(a) Quantum\nprocedure", ha="center", va="bottom", fontsize=11)
ax_flow.text(0.90,0.62,"(c) Output", ha="left", va="center", fontsize=11)
ax_flow.text(0.12,0.02,"(b) Input", ha="center", va="bottom", fontsize=11)

fig.suptitle(f"Distortion of the LO-dressed Rydberg atomic receiver (d={d_probe*1e3:.2f}mm)", fontsize=14)
fig.tight_layout(rect=[0,0,1,0.96])
fig.savefig(OUTPUT_DIR / "freshbuild_fig5_d036mm.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved: freshbuild_fig5_d036mm.png")

report = f"""# Fig. 5 math report -- d=0.36mm independent rebuild

Identical method to `fig5_fresh_build/fig5_quutip.py` (fresh, independent QuTiP
sweep over Omega_total, gamma3=gamma4=0 and Delta_p=Delta_c=Delta_LO=0 per
Sec.III-B's LO-dressed-specific approximation) -- ONLY d_probe changed from
0.76mm to 0.36mm. Does not touch fig5_fresh_build or the older qutip_d036mm files.

## Result

    Pin = {Pin/1e-6:.4f} microW (vs {(0.36/0.76)**2:.4f}x the 0.76mm value, by the proven d^2 scaling)
    n_qutip_solves = {n_qutip_solves}
    Linear dynamic range: [{fit['x_low']:.4f},{fit['x_high']:.4f}]MHz, R2={fit['r2']:.6f}
    kappa (slope) = {fit['slope']:.6e} W/MHz

## Comparison to d=0.76mm fresh_build

fig5_fresh_build's own kappa = -1.053982e-06 W/MHz, linear range [1.0344,7.3627]MHz.
Predicted kappa ratio (this project's d^2 scaling law, algebraically proven and
QuTiP-confirmed in fig6_fresh_build/fig6_angle4_probe_diameter.py): (0.36/0.76)^2 =
{(0.36/0.76)**2:.4f}. Actual ratio this run: {fit['slope']/-1.053982e-06:.4f}.
"""
with open(OUTPUT_DIR / "freshbuild_fig5_d036mm_math_report.md", "w") as f:
    f.write(report)
print("Saved: freshbuild_fig5_d036mm_math_report.md")
print("\nDONE.")
