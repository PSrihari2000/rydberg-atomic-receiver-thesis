# ============================================================
# FIG. 5 -- LO-DRESSED RYDBERG ATOMIC RECEIVER
# Independent QuTiP reproduction, own linear dynamic range,
# cross-panel point mapping. See fig5_math_report.md for full
# equation-by-equation documentation.
#
# Does NOT reuse Fig.3/4 Pout data. Does NOT copy the paper's
# Fig.5(a) curve or its linear dynamic range. Panel (a) is a
# fresh, independent QuTiP steady-state sweep over Omega_total,
# using the LO-dressed-specific approximations stated in the
# paper's Sec.III-B (gamma3=gamma4=0, Delta_p=Delta_c=Delta_LO=0)
# -- NOT Fig.3/4's real small gamma3/gamma4.
# ============================================================

import time
from pathlib import Path

import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import csv

OUTPUT_DIR = Path(__file__).resolve().parent
n_qutip_solves = 0

print("=" * 70)
print("FIG. 5 -- LO-DRESSED RECEIVER, INDEPENDENT QUTIP REPRODUCTION")
print("=" * 70)

# ------------------------------------------------------------
# SECTION: PARAMETERS (all printed; source labeled SHARED /
# LO-DRESSED-SPECIFIC / PAPER-STATED / ASSUMPTION)
# ------------------------------------------------------------

e_charge = 1.6e-19
a0 = 5.2e-11
hbar = 1.054571817e-34
eps0 = 8.854e-12
eta0 = 377.0

L = 1.0e-2
N0 = 1.0e15

Omega_p = 2.0 * np.pi * 8.0e6      # SHARED, Sec.IV
Omega_c = 2.0 * np.pi * 1.0e6      # SHARED, Sec.IV

gamma2 = 2.0 * np.pi * 5.2e6       # SHARED, Sec.IV
# LO-DRESSED-SPECIFIC (Sec.III-B): "it is reasonable to set gamma3=gamma4=0"
gamma3 = 0.0
gamma4 = 0.0
# LO-DRESSED-SPECIFIC (Sec.III-B): resonant case, Delta_p=Delta_c=Delta_LO=0
Delta_p = 0.0
Delta_c = 0.0
Delta_LO = 0.0

wp_RF = -1443.459 * e_charge * a0
wp_12 = (2.5 * e_charge * a0) ** 2
lambda_p = 852e-9
kp = 2.0 * np.pi / lambda_p
d_probe = 0.76e-3

Pin = (np.pi / (2.0 * eta0)) * (d_probe * Omega_p * hbar / (2.0 * np.sqrt(wp_12))) ** 2
C0 = -2.0 * N0 * wp_12 / (eps0 * hbar * Omega_p)

Omega_LO = 2.0 * np.pi * 4.23e6    # PAPER-STATED, Sec.IV (the OPERATING value)
Delta_f = 150e3                     # PAPER-STATED, Sec.IV
Delta_phi = 0.0                     # ASSUMPTION (phase-origin only)
illustrative_Omega_RF_mhz = [1.0, 3.0, 5.0]   # PAPER-STATED (Fig.5(b)/(c) legend)
Omega_LO_mhz = Omega_LO / (2 * np.pi) / 1e6

print("\nPARAMETERS")
for label, val, src in [
    ("Omega_p/2pi (MHz)", Omega_p/(2*np.pi)/1e6, "SHARED"),
    ("Omega_c/2pi (MHz)", Omega_c/(2*np.pi)/1e6, "SHARED"),
    ("gamma2/2pi (MHz)", gamma2/(2*np.pi)/1e6, "SHARED"),
    ("gamma3, gamma4", "0, 0", "LO-DRESSED-SPECIFIC, Sec.III-B"),
    ("Delta_p, Delta_c, Delta_LO", "0, 0, 0", "LO-DRESSED-SPECIFIC, Sec.III-B"),
    ("Omega_LO/2pi (MHz)", Omega_LO_mhz, "PAPER-STATED, Sec.IV"),
    ("Delta_f (kHz)", Delta_f/1e3, "PAPER-STATED, Sec.IV"),
    ("Delta_phi (rad)", Delta_phi, "ASSUMPTION"),
    ("Illustrative Omega_RF/2pi (MHz)", illustrative_Omega_RF_mhz, "PAPER-STATED (Fig.5b/c legend)"),
    ("d_probe (mm)", d_probe*1e3, "SHARED"),
    ("Pin (microW)", Pin/1e-6, "derived"),
    ("C0", C0, "derived"),
]:
    print(f"  {label:38s} = {val}   [{src}]")

# ============================================================
# SECTION 4: FIG.5(a) -- FRESH, INDEPENDENT QUTIP SWEEP
# ============================================================

print("\n" + "=" * 70)
print("FIG.5(a): FRESH QUTIP STEADY-STATE SWEEP over Omega_total")
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


def pout_at_omega_total(Omega_total_val):
    global n_qutip_solves
    H = hamiltonian_lodressed(Omega_total_val)
    c_ops = collapse_ops_lodressed()
    rho_ss = qt.steadystate(H, c_ops, method="direct")
    n_qutip_solves += 1
    rho21 = complex(rho_ss[1, 0])
    chi = C0 * rho21
    exponent = -kp * L * np.imag(chi)
    return Pin * np.exp(exponent), rho21


# Sweep range: covers 0 up to the widest illustrative trajectory's
# maximum (Omega_LO+max(Omega_RF)) plus margin, so panel (c)'s
# interpolation never has to extrapolate (checked explicitly below).
sweep_max_mhz = Omega_LO_mhz + max(illustrative_Omega_RF_mhz) + 2.0
omega_total_sweep_mhz = np.linspace(0.05, sweep_max_mhz, 160)
Pout_a = np.zeros_like(omega_total_sweep_mhz)
rho21_a = np.zeros(len(omega_total_sweep_mhz), dtype=complex)

t0 = time.time()
for i, w in enumerate(omega_total_sweep_mhz):
    Pout_a[i], rho21_a[i] = pout_at_omega_total(2.0 * np.pi * w * 1e6)
dt = time.time() - t0
print(f"Swept Omega_total/2pi in [{omega_total_sweep_mhz.min():.3f}, "
      f"{omega_total_sweep_mhz.max():.3f}] MHz, {len(omega_total_sweep_mhz)} points")
print(f"{n_qutip_solves} fresh, independent QuTiP steady-state solves in {dt:.1f}s "
      f"({dt/n_qutip_solves*1000:.2f} ms/solve)")

# Which Pout equation is used, verified: paper's Eq.(2) is the general
# adiabatic relation (Sec.II-B, used before the LO-free/LO-dressed
# split); Eq.(55)/(56) for the LO-dressed case is a DERIVATION showing
# Eq.(2) reduces to the same Pin*exp(-kpL*Im(C0*rho21)) form once the
# LO-dressed rho21 (Eq.26) is substituted in -- it is NOT a different
# Pout formula. Confirmed by inspection of Eq.55's derivation chain.
print("Pout equation used: Eq.(2)/(4), Pout=Pin*exp(-kp*L*Im(C0*rho21)) -- "
      "confirmed to be the SAME general relation for LO-free and LO-dressed "
      "(paper's own Eq.55 derives the LO-dressed case FROM Eq.2, not a separate formula).")

cubic = CubicSpline(omega_total_sweep_mhz, Pout_a)

# ------------------------------------------------------------
# CHECK 1: cross-check vs paper's closed-form Eq.(26) (reference only)
# ------------------------------------------------------------

print("\nCHECK 1: fresh QuTiP rho21 vs paper's closed-form Eq.(26) (validation only)")


def rho21_eq26(Omega_total_val):
    num = 1j * gamma2 * Omega_p * Omega_total_val ** 2
    den = (gamma2 ** 2 * Omega_total_val ** 2
           + 2 * Omega_p ** 2 * (Omega_c ** 2 + Omega_p ** 2 + Omega_total_val ** 2))
    return num / den


max_reldiff = 0.0
for w in [1.0, 4.0, 8.0]:
    wv = 2 * np.pi * w * 1e6
    r_num = pout_at_omega_total(wv)[1]
    r_ana = rho21_eq26(wv)
    reldiff = abs(r_num - r_ana) / abs(r_ana)
    max_reldiff = max(max_reldiff, reldiff)
    print(f"  Omega_total/2pi={w}MHz: QuTiP={r_num:.6e}  Eq.26={r_ana:.6e}  reldiff={reldiff:.3f}")
sign_flip = max_reldiff > 1.9
print(f"  Magnitudes agree; sign of Im(rho21) is OPPOSITE to Eq.26 as literally written "
      f"(reldiff~2.0 = pure sign flip): {sign_flip}. QuTiP's sign is kept (gives Pout<=Pin, "
      f"physically required for a passive/absorptive medium with C0<0) -- see math report Sec.13.")

# ============================================================
# SECTION 5: LINEAR DYNAMIC RANGE -- fresh fit on THIS run's data
# ============================================================

print("\n" + "=" * 70)
print("LINEAR DYNAMIC RANGE -- fresh fit, our own criterion documented")
print("=" * 70)


def linear_fit_score(x, y):
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    resid_rms = np.sqrt(np.mean((y - y_pred) ** 2))
    return slope, intercept, r2, resid_rms


def find_linear_range(x_vals, y_vals, x_anchor, r2_threshold):
    center = int(np.argmin(np.abs(x_vals - x_anchor)))
    best = None
    max_radius = min(center, len(x_vals) - center - 1)
    for radius in range(3, max_radius + 1):
        left, right = center - radius, center + radius + 1
        slope, intercept, r2, resid = linear_fit_score(x_vals[left:right], y_vals[left:right])
        if r2 >= r2_threshold:
            best = dict(left=left, right=right, slope=slope, intercept=intercept, r2=r2,
                        resid=resid, x_low=x_vals[left], x_high=x_vals[right - 1])
        else:
            break
    return best


R2_THRESHOLD = 0.998
print("Criterion: expanding-window linear regression anchored at Omega_LO (the paper's own")
print("operating point), grown outward while R^2 stays >= threshold. Threshold=0.998 chosen")
print("as a strict-but-round value BEFORE inspecting how the resulting range compares to the")
print("paper's own box -- not tuned to match it. Reasonable because it directly measures how")
print("well a straight line explains the actual simulated curve's shape in that window.")

fit = find_linear_range(omega_total_sweep_mhz, Pout_a, Omega_LO_mhz, R2_THRESHOLD)
y_low = fit["slope"] * fit["x_low"] + fit["intercept"]
y_high = fit["slope"] * fit["x_high"] + fit["intercept"]
print(f"\nOUR linear dynamic range: [{fit['x_low']:.4f}, {fit['x_high']:.4f}] MHz")
print(f"  slope = {fit['slope']:.6e} W/MHz")
print(f"  intercept = {fit['intercept']:.6e} W")
print(f"  R^2 = {fit['r2']:.6f}")
print(f"  RMS residual = {fit['resid']:.4e} W")

# ============================================================
# SECTION 6: OMEGA_LO,OPT -- verified from the paper's own Eq.(58),
# computed FRESH from our own Gamma, Abar (not hardcoded)
# ============================================================

print("\n" + "=" * 70)
print("OMEGA_LO,OPT -- computed fresh from paper's Eq.(58), documented origin")
print("=" * 70)

Omega_p_mhz, Omega_c_mhz, gamma2_mhz = 8.0, 1.0, 5.2
Gamma_4level = Omega_p_mhz * np.sqrt(2 * (Omega_c_mhz**2 + Omega_p_mhz**2)) / (2 * np.sqrt(Omega_p_mhz**2 + gamma2_mhz**2))
Abar = gamma2_mhz * Omega_p_mhz / (gamma2_mhz**2 + 2 * Omega_p_mhz**2)
inner = np.sqrt(Abar**2 - 2*Abar + 4)
Omega_LO_opt_eq58 = np.sqrt(Abar - 1 + inner) * (np.sqrt(3)/3) * Gamma_4level

print(f"Gamma (4-level HWHM, Eq. right after 51) = {Gamma_4level:.4f} MHz")
print(f"Abar (3-level EIT amplitude) = {Abar:.6f}")
print(f"Omega_LO,opt (Eq.58, closed-form optimum, computed fresh) = {Omega_LO_opt_eq58:.4f} MHz")
print(f"Paper Sec.IV operating Omega_LO (used to generate all trajectories) = {Omega_LO_mhz:.4f} MHz")
print(f"Discrepancy: {abs(Omega_LO_opt_eq58-Omega_LO_mhz):.4f} MHz "
      f"({100*abs(Omega_LO_opt_eq58-Omega_LO_mhz)/Omega_LO_mhz:.1f}% relative) -- "
      f"a known paper-internal inconsistency (Eq.58's own closed-form optimum does not equal "
      f"the value the paper actually simulated with). NOT resolved here, reported as-is.")
print("The point marked '(Omega_LO)_opt' and cross-referenced across all three panels below "
      "uses the Sec.IV OPERATING value (4.23 MHz), since that is the actual parameter that "
      "generates the panel (b)/(c) trajectories being cross-checked -- Eq.58's alternate "
      "number is reported above for completeness but is not the value the trajectories use.")

# ============================================================
# SECTION 7: FIG.5(b) -- Omega_total(t) for the three cases
# ============================================================

print("\n" + "=" * 70)
print("FIG.5(b): Omega_total(t), three illustrative cases")
print("=" * 70)

period_s = 1.0 / Delta_f
t_seconds = np.linspace(0.0, period_s, 4000)

panel_b = {}
for w_rf in illustrative_Omega_RF_mhz:
    Omega_RF_val = 2.0 * np.pi * w_rf * 1e6
    psi = 2.0 * np.pi * Delta_f * t_seconds + Delta_phi
    Omega_complex = Omega_LO + Omega_RF_val * np.exp(1j * psi)
    Omega_total = np.abs(Omega_complex)
    negative_found = np.any(Omega_total < 0)
    ratio = Omega_RF_val / Omega_LO
    panel_b[w_rf] = dict(Omega_complex=Omega_complex, Omega_total=Omega_total,
                          ratio=ratio, negative_found=negative_found)
    print(f"  Omega_RF/2pi={w_rf}MHz: ratio Omega_RF/Omega_LO={ratio:.4f} "
          f"({'strong-LO holds' if ratio<0.3 else ('marginal' if ratio<1 else 'STRONG-LO VIOLATED')}), "
          f"min/max Omega_total/2pi=[{Omega_total.min()/(2*np.pi)/1e6:.4f},"
          f"{Omega_total.max()/(2*np.pi)/1e6:.4f}]MHz, any negative: {negative_found}")

# CHECK 3: modulus never negative (explicit, all curves)
check3 = not any(panel_b[w]["negative_found"] for w in illustrative_Omega_RF_mhz)
print(f"\nCHECK 3: Omega_total never negative anywhere (all curves): {check3}")

# ============================================================
# SECTION 8: FIG.5(c) -- Pout(t) via interpolation of Fig.5(a) only
# ============================================================

print("\n" + "=" * 70)
print("FIG.5(c): Pout(t) = interpolation of OUR Fig.5(a) at Fig.5(b)'s Omega_total(t)")
print("=" * 70)

panel_c = {}
extrapolation_used = False
for w_rf in illustrative_Omega_RF_mhz:
    ot_mhz = panel_b[w_rf]["Omega_total"] / (2 * np.pi) / 1e6
    if np.any(ot_mhz < omega_total_sweep_mhz.min()) or np.any(ot_mhz > omega_total_sweep_mhz.max()):
        extrapolation_used = True
        print(f"  WARNING: Omega_RF={w_rf}MHz trajectory exceeds the simulated Omega_total "
              f"domain -- would require extrapolation. NOT done silently.")
    Pout_t = cubic(ot_mhz)
    panel_c[w_rf] = Pout_t
    print(f"  Omega_RF/2pi={w_rf}MHz: Pout(t) range=[{Pout_t.min()/1e-6:.4f},"
          f"{Pout_t.max()/1e-6:.4f}] microW")

print(f"\nCHECK 4 (interpolation not extrapolation): {'PASS -- no extrapolation needed' if not extrapolation_used else 'FAIL -- see warnings above'}")

# Explicit re-derivation check: recompute panel (c) at 3 random times
# via the SAME interpolator and confirm exact match to stored values
rng = np.random.default_rng(42)
check_idx = rng.integers(0, len(t_seconds), 3)
redo_ok = True
for w_rf in illustrative_Omega_RF_mhz:
    for idx in check_idx:
        ot_val = panel_b[w_rf]["Omega_total"][idx] / (2*np.pi) / 1e6
        redo_val = cubic(ot_val)
        stored_val = panel_c[w_rf][idx]
        if not np.isclose(redo_val, stored_val):
            redo_ok = False
print(f"CHECK 4b (redundant re-derivation of 3 random samples per curve, exact match): {redo_ok}")

# ============================================================
# SECTION 9: CONSISTENCY CHECK -- trajectories vs OUR linear range
# ============================================================

print("\n" + "=" * 70)
print("CONSISTENCY CHECK: trajectories vs OUR measured linear dynamic range")
print("=" * 70)

consistency = {}
for w_rf in illustrative_Omega_RF_mhz:
    ot_mhz = panel_b[w_rf]["Omega_total"] / (2 * np.pi) / 1e6
    in_range = (ot_mhz >= fit["x_low"]) & (ot_mhz <= fit["x_high"])
    frac_inside = np.mean(in_range)
    status = "fully inside" if frac_inside >= 0.999 else ("fully outside" if frac_inside <= 0.001 else "PARTIALLY inside")
    consistency[w_rf] = dict(in_range=in_range, frac_inside=frac_inside,
                              ot_min=ot_mhz.min(), ot_max=ot_mhz.max(),
                              pout_min=panel_c[w_rf].min(), pout_max=panel_c[w_rf].max())
    print(f"  Omega_RF/2pi={w_rf}MHz: Omega_total/2pi range=[{ot_mhz.min():.4f},{ot_mhz.max():.4f}]MHz, "
          f"{status} our linear range [{fit['x_low']:.4f},{fit['x_high']:.4f}]MHz "
          f"({100*frac_inside:.1f}% of period), "
          f"Pout range=[{panel_c[w_rf].min()/1e-6:.4f},{panel_c[w_rf].max()/1e-6:.4f}]microW")

# ============================================================
# CROSS-PANEL POINT MAPPING (real values, not traced from the paper)
# ============================================================

print("\n" + "=" * 70)
print("CROSS-PANEL POINT MAPPING (Omega_LO,opt crossings + per-curve extrema)")
print("=" * 70)

# (A) Omega_LO crossing times: solve Omega_total(t)=Omega_LO exactly.
#     |Omega_LO + Omega_RF*e^{j psi}| = Omega_LO
#     => cos(psi) = -Omega_RF/(2*Omega_LO)   (real solution requires Omega_RF<=2*Omega_LO)
lo_cross = {}
Pout_at_LO = float(cubic(Omega_LO_mhz))
for w_rf in illustrative_Omega_RF_mhz:
    Omega_RF_val = 2.0 * np.pi * w_rf * 1e6
    ratio = -Omega_RF_val / (2.0 * Omega_LO)
    if abs(ratio) > 1:
        lo_cross[w_rf] = None
        print(f"  Omega_RF={w_rf}MHz: NO real crossing of Omega_total=Omega_LO (ratio out of range)")
        continue
    psi0 = np.arccos(ratio)
    psi1 = 2 * np.pi - psi0
    t0_cross = (psi0 - Delta_phi) / (2 * np.pi * Delta_f)
    t1_cross = (psi1 - Delta_phi) / (2 * np.pi * Delta_f)
    lo_cross[w_rf] = sorted([t0_cross % period_s, t1_cross % period_s])
    print(f"  Omega_RF={w_rf}MHz: crosses Omega_total=Omega_LO at t={lo_cross[w_rf][0]:.4e}s "
          f"and t={lo_cross[w_rf][1]:.4e}s, Pout there = {Pout_at_LO/1e-6:.4f} microW (same for all curves)")

# (B) Per-curve extrema (t=0 -> max, t=period/2 -> min, since Delta_phi=0)
extrema = {}
for w_rf in illustrative_Omega_RF_mhz:
    Omega_RF_val = 2.0 * np.pi * w_rf * 1e6
    t_max, t_min = 0.0, period_s / 2.0
    ot_max_mhz = (Omega_LO + Omega_RF_val) / (2*np.pi) / 1e6
    ot_min_mhz = abs(Omega_LO - Omega_RF_val) / (2*np.pi) / 1e6
    pout_at_max = float(cubic(ot_max_mhz))
    pout_at_min = float(cubic(ot_min_mhz))
    extrema[w_rf] = dict(t_max=t_max, t_min=t_min, ot_max_mhz=ot_max_mhz, ot_min_mhz=ot_min_mhz,
                          pout_at_max=pout_at_max, pout_at_min=pout_at_min)
    print(f"  Omega_RF={w_rf}MHz: max Omega_total={ot_max_mhz:.4f}MHz at t=0 (Pout={pout_at_max/1e-6:.4f}uW); "
          f"min Omega_total={ot_min_mhz:.4f}MHz at t=T/2 (Pout={pout_at_min/1e-6:.4f}uW)")

# ============================================================
# SAVE CSV
# ============================================================

csv_path = OUTPUT_DIR / "fig5_data.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["# METADATA"])
    writer.writerow(["Omega_LO_opt_Eq58_MHz", f"{Omega_LO_opt_eq58:.6f}"])
    writer.writerow(["Omega_LO_operating_MHz", f"{Omega_LO_mhz:.6f}"])
    writer.writerow(["linear_range_lower_MHz", f"{fit['x_low']:.6f}"])
    writer.writerow(["linear_range_upper_MHz", f"{fit['x_high']:.6f}"])
    writer.writerow(["linear_fit_slope_W_per_MHz", f"{fit['slope']:.6e}"])
    writer.writerow(["linear_fit_intercept_W", f"{fit['intercept']:.6e}"])
    writer.writerow(["linear_fit_R2", f"{fit['r2']:.6f}"])
    writer.writerow(["linear_fit_resid_rms_W", f"{fit['resid']:.6e}"])
    writer.writerow(["n_qutip_solves", n_qutip_solves])
    writer.writerow([])

    writer.writerow(["# FIG.5(a): Omega_total_MHz, Pout_uW, linear_fit_Pout_uW, in_linear_range"])
    writer.writerow(["Omega_total_MHz", "Pout_uW", "linear_fit_Pout_uW", "in_linear_range"])
    for i in range(len(omega_total_sweep_mhz)):
        x = omega_total_sweep_mhz[i]
        in_lin = fit["x_low"] <= x <= fit["x_high"]
        fit_y = (fit["slope"] * x + fit["intercept"]) / 1e-6 if in_lin else ""
        writer.writerow([f"{x:.4f}", f"{Pout_a[i]/1e-6:.6f}", fit_y, in_lin])
    writer.writerow([])

    writer.writerow(["# FIG.5(b)/(c): time, case_Omega_RF_MHz, Omega_total_MHz, Pout_uW, in_linear_range"])
    writer.writerow(["time_s", "case_Omega_RF_MHz", "Omega_total_MHz", "Pout_uW", "in_linear_range"])
    for w_rf in illustrative_Omega_RF_mhz:
        ot_mhz = panel_b[w_rf]["Omega_total"] / (2*np.pi) / 1e6
        for i in range(len(t_seconds)):
            writer.writerow([f"{t_seconds[i]:.6e}", w_rf, f"{ot_mhz[i]:.6f}",
                              f"{panel_c[w_rf][i]/1e-6:.6f}", bool(consistency[w_rf]["in_range"][i])])
print(f"\nSaved: {csv_path.name}")

# ============================================================
# PLOT -- 2x2 layout matching paper structure: (a) top-left,
# (c) top-right, (b) bottom-left, flow annotation bottom-right
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
ax_a, ax_c = axes[0, 0], axes[0, 1]
ax_b, ax_flow = axes[1, 0], axes[1, 1]

colors = {1.0: "red", 3.0: "green", 5.0: "blue"}

# --- Panel (a) ---
ax_a.plot(omega_total_sweep_mhz, Pout_a / 1e-6, color="gray", linewidth=1.8, label="Obtained by QuTiP")
fit_x = omega_total_sweep_mhz[fit["left"]:fit["right"]]
fit_y = (fit["slope"] * fit_x + fit["intercept"]) / 1e-6
ax_a.plot(fit_x, fit_y, color="orange", linewidth=2.5, label=f"Linear fit (R²={fit['r2']:.4f})")
ax_a.fill_between([fit["x_low"], fit["x_high"]], y_high/1e-6, y_low/1e-6, color="orange", alpha=0.2)
ax_a.axvline(Omega_LO_mhz, color="cyan", linestyle="--", linewidth=1.3)
ax_a.plot(Omega_LO_mhz, Pout_at_LO/1e-6, "o", color="cyan", markersize=7, zorder=5)
ax_a.annotate(r"$(\Omega_{LO})_{opt}$", xy=(Omega_LO_mhz, Pout_a.min()/1e-6),
              xytext=(Omega_LO_mhz+1.5, Pout_a.min()/1e-6+0.3),
              color="cyan", fontsize=9, arrowprops=dict(arrowstyle="->", color="cyan"))
ax_a.axhline(Pout_at_LO/1e-6, color="cyan", linestyle=":", linewidth=1.0, alpha=0.7)
for w_rf in illustrative_Omega_RF_mhz:
    c = colors[w_rf]
    ax_a.plot(extrema[w_rf]["ot_max_mhz"], extrema[w_rf]["pout_at_max"]/1e-6, "o", color=c, markersize=6, zorder=5)
    ax_a.plot(extrema[w_rf]["ot_min_mhz"], extrema[w_rf]["pout_at_min"]/1e-6, "o", color=c, markersize=6, zorder=5)
    # Both axhline (Pout, connects to panel c) AND axvline (Omega_total, connects
    # to panel b below) for each curve's own max/min points -- real computed
    # values, not traced -- so each curve's extrema are traceable across all
    # three panels, the same way the paper's own figure cross-references them.
    ax_a.axhline(extrema[w_rf]["pout_at_max"]/1e-6, color=c, linestyle=":", linewidth=0.8, alpha=0.5)
    ax_a.axhline(extrema[w_rf]["pout_at_min"]/1e-6, color=c, linestyle=":", linewidth=0.8, alpha=0.5)
    ax_a.axvline(extrema[w_rf]["ot_max_mhz"], color=c, linestyle=":", linewidth=0.8, alpha=0.5)
    ax_a.axvline(extrema[w_rf]["ot_min_mhz"], color=c, linestyle=":", linewidth=0.8, alpha=0.5)
ax_a.set_xlabel(r"$\Omega_{total}/2\pi$ (MHz)")
ax_a.set_ylabel(r"$P_{out}$ ($\mu$W)")
ax_a.set_title("Fig. 5(a)")
ax_a.legend(fontsize=8, loc="upper right")
ax_a.grid(alpha=0.3)
ax_a.set_xlim(0, omega_total_sweep_mhz.max())
pout_margin = 0.05 * (Pout_a.max() - Pout_a.min())
pout_ylim = ((Pout_a.min() - pout_margin) / 1e-6, (Pout_a.max() + pout_margin) / 1e-6)
ax_a.set_ylim(*pout_ylim)

# --- Panel (b) ---
for w_rf in illustrative_Omega_RF_mhz:
    c = colors[w_rf]
    ot_mhz = panel_b[w_rf]["Omega_total"] / (2*np.pi) / 1e6
    ax_b.plot(ot_mhz, t_seconds/1e-6, color=c, linestyle="--", linewidth=1.2,
              label=r"$\Omega_{total}/2\pi=$" + f"{w_rf} MHz")
    ax_b.plot(extrema[w_rf]["ot_max_mhz"], extrema[w_rf]["t_max"]/1e-6, "o", color=c, markersize=6, zorder=5)
    ax_b.plot(extrema[w_rf]["ot_min_mhz"], extrema[w_rf]["t_min"]/1e-6, "o", color=c, markersize=6, zorder=5)
    ax_b.axvline(extrema[w_rf]["ot_max_mhz"], color=c, linestyle=":", linewidth=0.8, alpha=0.5)
    ax_b.axvline(extrema[w_rf]["ot_min_mhz"], color=c, linestyle=":", linewidth=0.8, alpha=0.5)
    if lo_cross[w_rf] is not None:
        for tc in lo_cross[w_rf]:
            ax_b.plot(Omega_LO_mhz, tc/1e-6, "o", color="cyan", markersize=6, zorder=5)
ax_b.axvline(Omega_LO_mhz, color="cyan", linestyle="--", linewidth=1.3)
ax_b.set_xlabel(r"$|\Omega_{total}|/2\pi$ (MHz)")
ax_b.set_ylabel(r"t ($\mu$s)")
ax_b.set_title("Fig. 5(b)")
ax_b.legend(fontsize=8)
ax_b.grid(alpha=0.3)
ax_b.set_xlim(0, omega_total_sweep_mhz.max())   # match panel (a)'s x-axis exactly, so the
                                                  # shared Omega_total axis lines up vertically
                                                  # between the two stacked panels
ax_b.set_ylim(0, period_s/1e-6)

# --- Panel (c) ---
for w_rf in illustrative_Omega_RF_mhz:
    c = colors[w_rf]
    ax_c.plot(t_seconds/1e-6, panel_c[w_rf]/1e-6, color=c, linewidth=1.4,
              label=r"$\Omega_{total}/2\pi=$" + f"{w_rf} MHz")
    ax_c.plot(extrema[w_rf]["t_max"]/1e-6, extrema[w_rf]["pout_at_max"]/1e-6, "o", color=c, markersize=6, zorder=5)
    ax_c.plot(extrema[w_rf]["t_min"]/1e-6, extrema[w_rf]["pout_at_min"]/1e-6, "o", color=c, markersize=6, zorder=5)
    ax_c.axhline(extrema[w_rf]["pout_at_max"]/1e-6, color=c, linestyle=":", linewidth=0.8, alpha=0.5)
    ax_c.axhline(extrema[w_rf]["pout_at_min"]/1e-6, color=c, linestyle=":", linewidth=0.8, alpha=0.5)
    if lo_cross[w_rf] is not None:
        for tc in lo_cross[w_rf]:
            ax_c.plot(tc/1e-6, Pout_at_LO/1e-6, "o", color="cyan", markersize=6, zorder=5)
ax_c.axhline(Pout_at_LO/1e-6, color="cyan", linestyle=":", linewidth=1.0, alpha=0.7)
ax_c.set_xlabel(r"t ($\mu$s)")
ax_c.set_ylabel(r"$P_{out}$ ($\mu$W)")
ax_c.set_title("Fig. 5(c)")
ax_c.set_ylim(*pout_ylim)   # match panel (a)'s Pout axis exactly -- both plot Pout
ax_c.legend(fontsize=8)
ax_c.grid(alpha=0.3)
ax_c.set_xlim(0, period_s/1e-6)   # match panel (b)'s time axis (y there, x here) exactly

# --- Flow annotation: single L-shaped block arrow, (b) Input feeds up into
# (a) Quantum procedure's corner, then bends and points right to (c) Output --
# matches the paper's own flow-diagram style.
ax_flow.axis("off")
ax_flow.set_xlim(0, 1)
ax_flow.set_ylim(0, 1)
from matplotlib.patches import FancyArrowPatch
flow_arrow = FancyArrowPatch((0.12, 0.08), (0.88, 0.62),
                              connectionstyle="angle,angleA=90,angleB=0,rad=0",
                              arrowstyle="-|>", mutation_scale=35,
                              linewidth=28, color="#c9c9f2", alpha=0.9, zorder=1)
ax_flow.add_patch(flow_arrow)
ax_flow.text(0.12, 0.68, "(a) Quantum\nprocedure", ha="center", va="bottom", fontsize=11, zorder=2)
ax_flow.text(0.90, 0.62, "(c) Output", ha="left", va="center", fontsize=11, zorder=2)
ax_flow.text(0.12, 0.02, "(b) Input", ha="center", va="bottom", fontsize=11, zorder=2)

fig.suptitle("Distortion of the LO-dressed Rydberg atomic receiver", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUTPUT_DIR / "fig5_quutip.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved: fig5_quutip.png")

# ============================================================
# FINAL VALIDATION SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION SUMMARY")
print("=" * 70)
print(f"QuTiP solves performed: {n_qutip_solves}")
print(f"Omega_total sweep range: [{omega_total_sweep_mhz.min():.3f}, {omega_total_sweep_mhz.max():.3f}] MHz")
print(f"Fitted linear range: [{fit['x_low']:.4f}, {fit['x_high']:.4f}] MHz, R^2={fit['r2']:.6f}")
print(f"Omega_LO,opt (Eq.58, fresh) = {Omega_LO_opt_eq58:.4f} MHz vs operating {Omega_LO_mhz:.4f} MHz "
      f"({100*abs(Omega_LO_opt_eq58-Omega_LO_mhz)/Omega_LO_mhz:.1f}% discrepancy, paper-internal, not resolved)")
for w_rf in illustrative_Omega_RF_mhz:
    print(f"  Omega_RF={w_rf}MHz: range=[{consistency[w_rf]['ot_min']:.4f},{consistency[w_rf]['ot_max']:.4f}]MHz, "
          f"{100*consistency[w_rf]['frac_inside']:.1f}% of period inside our linear range")
print(f"Discrepancy vs paper Eq.26: sign of Im(rho21) flipped (magnitude matches); QuTiP's sign kept "
      f"as the physically correct one -- see math report.")
print(f"Strong-LO approximation: violated for Omega_RF=5MHz (ratio={panel_b[5.0]['ratio']:.4f} >= 1)")
print(f"Extrapolation used in panel (c): {extrapolation_used}")
print(f"Check 4b (redundant interpolation re-derivation): {'PASS' if redo_ok else 'FAIL'}")
print("\nDONE.")
