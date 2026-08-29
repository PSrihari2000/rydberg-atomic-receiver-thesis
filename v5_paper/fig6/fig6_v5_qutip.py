# ============================================================
# FIG. 6 (v5 paper) -- Linear dynamic range of the LO-dressed
# Rydberg atomic receiver. Full 3-panel reproduction:
#   (a) Pout vs Omega_total  (real QuTiP)
#   (b) Omega_total vs time  (phasor form, transposed axes to
#       match the paper's own layout: t on y-axis, Omega_total
#       on x-axis, so it visually lines up under panel (a))
#   (c) Pout vs time (interpolated from panel (a)'s real curve)
#
# See fig6_v5_forensic_audit.md for the full equation/parameter
# audit this build follows. Key points carried over:
#   - Gamma = four-level Gamma (Eq.78, Appendix C), NOT Fig.4/5's
#     measured Gamma_ref -- confirmed correct context in the audit.
#   - epsilon (THD tolerance, Eq.58) has NO paper-stated value --
#     the paper itself calls it "the designer-specified tolerance".
#     The shaded LDR band below therefore uses a few explicitly
#     labeled ILLUSTRATIVE epsilon values, not one false-confident
#     paper-confirmed band.
#   - Omega_RF,min has no closed-form equation in the paper at all
#     (only "corresponds to the intrinsic sensitivity", no formula)
#     -- shown here only via the same illustrative-epsilon approach
#     for symmetry, clearly labeled, not derived from a real formula.
#   - Omega_total(t) uses the confirmed COMPLEX PHASOR form (not a
#     plain cosine) -- same form already validated in this
#     project's earlier v1 Fig.5 work.
#   - Delta_f=15kHz is paper-stated. Delta_phi has no paper value --
#     Delta_phi=0 used here, explicitly OUR ILLUSTRATIVE CHOICE.
#   - Time axis: plotted over one real period (1/Delta_f), NOT the
#     paper's own oddly-scaled ~1.5e-10s axis -- same already-
#     documented paper-internal time-axis inconsistency found for
#     v1's analogous Fig.5(b)/(c), not silently "fixed" to match.
# ============================================================

from pathlib import Path
import time

import numpy as np
import qutip as qt
import matplotlib.pyplot as plt

OUTPUT_DIR = Path(__file__).resolve().parent

# ------------------------------------------------------------
# PHYSICAL CONSTANTS / PARAMETERS (paper Sec.V-A, same as Fig.3-5)
# ------------------------------------------------------------
e_charge = 1.6e-19
a0 = 5.2e-11
hbar = 1.054571817e-34
eps0 = 8.854e-12
Z0 = 377.0

L_cell = 1.0e-2
N0 = 4.89e10 * 1e6

gamma2 = 2.0 * np.pi * 5.2e6
gamma3 = 2.0 * np.pi * 3.9e3
gamma4 = 2.0 * np.pi * 1.7e3

Omega_p = 2.0 * np.pi * 8.0e6
Omega_c = 2.0 * np.pi * 1.0e6

wp_12 = (2.5 * e_charge * a0) ** 2

lambda_p = 852e-9
kp = 2.0 * np.pi / lambda_p

Pin = 20.7e-6
C0 = -2.0 * N0 * wp_12 / (eps0 * hbar * Omega_p)

# ------------------------------------------------------------
# Analytic four-level Gamma (Eq.78, Appendix C)
# ------------------------------------------------------------
gamma2_mhz = 5.2
Omega_p_mhz = 8.0
Omega_c_mhz = 1.0
Gamma3_HWHM_mhz = (Omega_c_mhz**2 + Omega_p_mhz**2) / (2*np.sqrt(gamma2_mhz**2 + 2*Omega_p_mhz**2))
Gamma4_mhz = Omega_p_mhz * np.sqrt(2*(Omega_c_mhz**2 + Omega_p_mhz**2) / (2*Omega_p_mhz**2 + gamma2_mhz**2))
Omega_LO_opt_analytic_mhz = (np.sqrt(3)/3) * Gamma4_mhz

print("=" * 70)
print("FIG. 6 (v5) -- REAL QuTiP, LO-dressed, full 3-panel build")
print("=" * 70)
print(f"qutip version = {qt.__version__}")
print(f"Gamma (four-level, Eq.78) = {Gamma4_mhz:.4f} MHz")
print(f"Omega_LO,opt (analytic) = {Omega_LO_opt_analytic_mhz:.4f} MHz (paper states 4.23 MHz)")
print("=" * 70)


def basis4():
    return [qt.basis(4, i) for i in range(4)]


def hamiltonian(Omega_total, Delta_p=0.0, Delta_c=0.0, Delta_RF=0.0):
    k1, k2, k3, k4 = basis4()
    H = qt.qzero(4)
    H += -Delta_p * k2 * k2.dag()
    H += -(Delta_p + Delta_c) * k3 * k3.dag()
    H += -(Delta_p + Delta_c + Delta_RF) * k4 * k4.dag()
    H += (Omega_p / 2.0) * (k1 * k2.dag() + k2 * k1.dag())
    H += (Omega_c / 2.0) * (k2 * k3.dag() + k3 * k2.dag())
    H += (Omega_total / 2.0) * (k3 * k4.dag() + k4 * k3.dag())
    return H


def collapse_ops():
    k1, k2, k3, k4 = basis4()
    return [
        np.sqrt(gamma2) * k1 * k2.dag(),
        np.sqrt(gamma3) * k2 * k3.dag(),
        np.sqrt(gamma4) * k3 * k4.dag(),
    ]


C_OPS = collapse_ops()


def pout_at(Omega_total):
    H = hamiltonian(Omega_total)
    rho_ss = qt.steadystate(H, C_OPS, method="direct")
    rho21 = complex(rho_ss[1, 0])
    chi = C0 * rho21
    return Pin * np.exp(-kp * L_cell * np.imag(chi))


# ------------------------------------------------------------
# PANEL (a): real QuTiP sweep, Omega_total/2pi = 0 to 10 MHz
# ------------------------------------------------------------
omega_total_mhz = np.linspace(0.01, 10.0, 201)
omega_total_vals = 2.0 * np.pi * omega_total_mhz * 1e6

Pout_vals = np.zeros(len(omega_total_mhz))
t0 = time.time()
for i, Ot in enumerate(omega_total_vals):
    Pout_vals[i] = pout_at(Ot)
dt = time.time() - t0
print(f"\nPanel (a) sweep completed in {dt:.1f}s ({len(omega_total_mhz)} REAL qutip solves, "
      f"{dt/len(omega_total_mhz)*1000:.2f} ms/solve avg)")

dPout = np.gradient(Pout_vals, omega_total_mhz)
edge_mask = omega_total_mhz > 0.3
idx_opt_local = np.argmax(np.abs(dPout[edge_mask]))
Omega_LO_opt_numeric_mhz = omega_total_mhz[edge_mask][idx_opt_local]
print(f"Omega_LO,opt (NUMERICAL, real QuTiP) = {Omega_LO_opt_numeric_mhz:.4f} MHz (paper states 4.67 MHz)")

# tangent (linear approximation) lines at each Omega_LO candidate --
# this is what the paper's own "Omega_LO=4.67MHz"/"Omega_LO,opt=4.23MHz"
# traces actually are (confirmed from the rendered image: they hug the
# true curve near their own bias point and diverge away from it,
# exactly the signature of a straight tangent line, not a vertical line)
def tangent_line(bias_mhz):
    slope = np.interp(bias_mhz, omega_total_mhz, dPout)
    P0 = np.interp(bias_mhz, omega_total_mhz, Pout_vals)
    return P0 + slope * (omega_total_mhz - bias_mhz)

tangent_analytic = tangent_line(Omega_LO_opt_analytic_mhz)
tangent_numeric = tangent_line(Omega_LO_opt_numeric_mhz)

# ------------------------------------------------------------
# LINEAR DYNAMIC RANGE -- via the paper's ACTUAL Eq.57 formula
# (Omega_RF,max'' = 4|kappa0'/kappa0''|), NOT an ad-hoc tangent-
# line-vs-curve deviation (an earlier version of this script did
# that; it was WRONG -- comparing absolute deviation as a fraction
# of the PEAK gives a trivially-small, meaningless "match" once the
# curve decays toward zero, since both curve and tangent are tiny
# there regardless of whether the tangent means anything. Eq.57
# compares actual Taylor-series derivative magnitudes instead,
# which is what the paper's distortion analysis is actually about).
#
# Eq.57 needs NO epsilon -- it only needs kappa0' and kappa0''
# (real numeric derivatives of our own QuTiP curve). It becomes
# undefined (kappa0''=0) exactly AT an inflection point -- which is
# mathematically the SAME thing as "point of maximum |slope|", so
# it is undefined at our own numeric optimum (1.31MHz) for the same
# reason the paper says 2nd-order distortion vanishes at ITS optimum
# (4.23MHz) -- our curve reproduces that same structural property,
# just centered at a different Omega_total because our curve is
# sharper (see math report). Eq.57 DOES give a real, finite, real-
# derivative-based answer if evaluated at the analytic bias point
# (4.23MHz), which is not an inflection point of our own curve.
# ------------------------------------------------------------
d2Pout = np.gradient(dPout, omega_total_mhz)

def eq57_bound(bias_mhz):
    k1 = np.interp(bias_mhz, omega_total_mhz, dPout)
    k2 = np.interp(bias_mhz, omega_total_mhz, d2Pout)
    return 4 * abs(k1 / k2) if abs(k2) > 1e-9 else float("inf")

ORF_max_eq57_at_analytic = eq57_bound(Omega_LO_opt_analytic_mhz)
ORF_max_eq57_at_numeric = eq57_bound(Omega_LO_opt_numeric_mhz)
print(f"Eq.57 Omega_RF,max (epsilon-free) at analytic bias (4.23MHz) = {ORF_max_eq57_at_analytic:.4f} MHz")
print(f"Eq.57 Omega_RF,max (epsilon-free) at numeric bias (1.31MHz) = {ORF_max_eq57_at_numeric:.4f} MHz "
      f"(undefined/huge -- kappa0''~0 there, an inflection point, same structural reason the paper's "
      f"OWN optimum needs Eq.58/epsilon instead -- see math report)")

# ------------------------------------------------------------
# PANEL (b)/(c): Omega_total(t) and Pout(t), phasor form
# Delta_f=15kHz paper-stated. Delta_phi=0: OUR ILLUSTRATIVE CHOICE.
# Omega_LO = analytic optimum (4.23MHz, matches paper). Omega_RF
# values match the paper's own Fig.6(b)/(c) legend exactly: 0.5,
# 1.5, 3.0 MHz.
# ------------------------------------------------------------
Delta_f = 15e3  # Hz, paper-stated
Delta_phi = 0.0  # OUR ILLUSTRATIVE CHOICE -- no paper value exists
Omega_LO_for_time = Omega_LO_opt_analytic_mhz  # MHz

period = 1.0 / Delta_f  # seconds, the REAL period -- paper's own plotted axis
                         # (~1.5e-10s) doesn't match this by ~4 orders of
                         # magnitude, same already-documented time-axis
                         # inconsistency found for v1's analogous Fig.5(b)/(c)
t_vals = np.linspace(0, period, 400)
theta = 2 * np.pi * Delta_f * t_vals + Delta_phi

ORF_list_mhz = [0.5, 1.5, 3.0]  # matches paper's own panel (b)/(c) legend exactly
panel_bc_data = {}
for ORF_mhz in ORF_list_mhz:
    Omega_total_complex = Omega_LO_for_time + ORF_mhz * np.exp(1j * theta)
    Omega_total_t_mhz = np.abs(Omega_total_complex)
    Omega_total_t_mhz_clamped = np.clip(Omega_total_t_mhz, omega_total_mhz.min(), omega_total_mhz.max())
    Pout_t = np.interp(Omega_total_t_mhz_clamped, omega_total_mhz, Pout_vals)
    panel_bc_data[ORF_mhz] = (Omega_total_t_mhz, Pout_t)

print(f"\nPanel (b)/(c): Delta_f={Delta_f/1e3:.0f}kHz (paper-stated), "
      f"Delta_phi=0 (OUR illustrative choice), Omega_LO={Omega_LO_for_time:.2f}MHz, "
      f"real period={period*1e6:.3f} microseconds "
      f"(paper's own plotted axis ~1.5e-10s does not match this -- known paper-internal "
      f"time-axis inconsistency, not corrected here, see math report)")

# ------------------------------------------------------------
# Save data
# ------------------------------------------------------------
import csv
with open(OUTPUT_DIR / "fig6_v5_qutip_ldr_data.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["panel", "x", "y", "series"])
    for i in range(len(omega_total_mhz)):
        writer.writerow(["a", f"{omega_total_mhz[i]:.4f}", f"{Pout_vals[i]*1e6:.6f}", "EIT_real_qutip"])
        writer.writerow(["a", f"{omega_total_mhz[i]:.4f}", f"{tangent_analytic[i]*1e6:.6f}", "tangent_analytic"])
        writer.writerow(["a", f"{omega_total_mhz[i]:.4f}", f"{tangent_numeric[i]*1e6:.6f}", "tangent_numeric"])
    for ORF_mhz, (Ot, Pt) in panel_bc_data.items():
        for i in range(len(t_vals)):
            writer.writerow(["b", f"{t_vals[i]:.6e}", f"{Ot[i]:.6f}", f"ORF_{ORF_mhz}MHz"])
            writer.writerow(["c", f"{t_vals[i]:.6e}", f"{Pt[i]*1e6:.6f}", f"ORF_{ORF_mhz}MHz"])

np.savez(
    OUTPUT_DIR / "fig6_v5_qutip_ldr_response.npz",
    omega_total_mhz=omega_total_mhz, Pout_vals=Pout_vals, dPout=dPout,
    tangent_analytic=tangent_analytic, tangent_numeric=tangent_numeric,
    Gamma4_mhz=Gamma4_mhz, Omega_LO_opt_analytic_mhz=Omega_LO_opt_analytic_mhz,
    Omega_LO_opt_numeric_mhz=Omega_LO_opt_numeric_mhz,
    t_vals=t_vals, Delta_f=Delta_f, Delta_phi=Delta_phi,
)

# ------------------------------------------------------------
# PLOT -- 3-panel layout matching the paper's own arrangement:
#   (a) top-left      : Pout vs Omega_total
#   (c) top-right      : Pout vs t
#   (b) bottom-left     : t vs Omega_total (transposed, matches paper)
# ------------------------------------------------------------
fig = plt.figure(figsize=(12, 9))
ax_a = fig.add_subplot(2, 2, 1)
ax_c = fig.add_subplot(2, 2, 2)
ax_b = fig.add_subplot(2, 2, 3)

# Panel (a)
ax_a.plot(omega_total_mhz, Pout_vals * 1e6, color="purple", linewidth=2, label="EIT (real QuTiP)")
ax_a.plot(omega_total_mhz, tangent_numeric * 1e6, color="deepskyblue", linestyle="--", linewidth=1.3,
          label=rf"$\Omega_{{LO}}$={Omega_LO_opt_numeric_mhz:.2f} MHz (QuTiP)")
ax_a.plot(omega_total_mhz, tangent_analytic * 1e6, color="magenta", linestyle=":", linewidth=1.5,
          label=rf"$\Omega_{{LO,opt}}$={Omega_LO_opt_analytic_mhz:.2f} MHz")
ax_a.axvspan(Omega_LO_opt_analytic_mhz - ORF_max_eq57_at_analytic,
             Omega_LO_opt_analytic_mhz + ORF_max_eq57_at_analytic,
             color="lightblue", alpha=0.4,
             label=rf"Eq.57 LDR (real derivatives, no $\epsilon$): $\pm${ORF_max_eq57_at_analytic:.2f} MHz")
ax_a.set_xlabel(r"$\Omega_{total}/2\pi$ (MHz)")
ax_a.set_ylabel(r"$P_{out}$ ($\mu$W)")
ax_a.set_title("(a) Pout vs Omega_total -- real QuTiP")
ax_a.set_ylim(0, 9)
ax_a.legend(fontsize=7, loc="upper right")

# Panel (c): Pout vs t
colors = {0.5: "orange", 1.5: "green", 3.0: "red"}
styles = {0.5: ":", 1.5: "--", 3.0: "-."}
for ORF_mhz, (Ot, Pt) in panel_bc_data.items():
    ax_c.plot(t_vals * 1e6, Pt * 1e6, color=colors[ORF_mhz], linestyle=styles[ORF_mhz], linewidth=1.6,
              label=rf"$\Omega_{{RF}}$={ORF_mhz} MHz")
ax_c.set_xlabel(r"$t$ ($\mu$s)")
ax_c.set_ylabel(r"$P_{out}$ ($\mu$W)")
ax_c.set_title("(c) Pout vs t -- real period 1/Delta_f (see note)")
ax_c.legend(fontsize=8)

# Panel (b): t vs Omega_total (transposed, matches paper's own layout)
for ORF_mhz, (Ot, Pt) in panel_bc_data.items():
    ax_b.plot(Ot, t_vals * 1e6, color=colors[ORF_mhz], linestyle=styles[ORF_mhz], linewidth=1.6,
              label=rf"$\Omega_{{RF}}$={ORF_mhz} MHz")
ax_b.set_xlabel(r"$\Omega_{total}/2\pi$ (MHz)")
ax_b.set_ylabel(r"$t$ ($\mu$s)")
ax_b.set_title("(b) Omega_total vs t (transposed, matches paper layout)")
ax_b.legend(fontsize=8)

fig.tight_layout()
fig.savefig(OUTPUT_DIR / "fig6_v5_qutip_ldr.png", dpi=200, bbox_inches="tight")
plt.close(fig)

print("\nSaved: fig6_v5_qutip_ldr.png, fig6_v5_qutip_ldr_data.csv, fig6_v5_qutip_ldr_response.npz")
print("DONE.")
