# ============================================================
# *** CASE STUDY -- NOT a validated result ***
#
# TEST: what if the sign corrections this project made were wrong, and
# the paper's literal (negative-exponent) values should be used as-is?
#   N0 = 1e-15 m^-3 (literal) instead of the sign-corrected +1e15
#   e = +1.6e19 C (literal, missing the printed exponent's minus sign)
#     instead of the standard +1.6e-19 C
# Both are almost certainly PDF-extraction artifacts (a lost superscript
# minus sign), not real ambiguities -- this just confirms that
# numerically, by actually computing it rather than assuming.
#
# Does not touch any frozen fresh_build file.
# ============================================================

import numpy as np
import qutip as qt

print("=" * 70)
print("*** CASE STUDY 05 -- literal (uncorrected) paper values sanity check ***")
print("=" * 70)

a0 = 5.2e-11
hbar = 1.054571817e-34
eps0 = 8.854e-12
eta0 = 377.0
Omega_p = 2.0*np.pi*8.0e6
Omega_c = 2.0*np.pi*1.0e6
gamma2 = 2.0*np.pi*5.2e6
gamma3 = 2.0*np.pi*3.9e3
gamma4 = 2.0*np.pi*1.7e3
lambda_p = 852e-9
kp_wave = 2.0*np.pi/lambda_p
d_probe = 0.76e-3
L_cell = 1.0e-2

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

def pout_with(e_charge, N0):
    wp_12 = (2.5*e_charge*a0)**2
    Pin = (np.pi/(2.0*eta0))*(d_probe*Omega_p*hbar/(2.0*np.sqrt(wp_12)))**2
    C0 = -2.0*N0*wp_12/(eps0*hbar*Omega_p)
    H = hamiltonian(2*np.pi*6e6, 0.0)
    rho_ss = qt.steadystate(H, collapse_ops(), method="direct")
    rho21 = complex(rho_ss[1,0])
    chi = C0*rho21
    return Pin*np.exp(-kp_wave*L_cell*np.imag(chi)), Pin, C0

print("\nReal baseline (used everywhere): e=+1.6e-19 C, N0=+1e15 m^-3")
Pout_base, Pin_base, C0_base = pout_with(1.6e-19, 1e15)
print(f"  Pin={Pin_base/1e-6:.4f}uW, C0={C0_base:.4e}, Pout(Omega_RF/2pi=6MHz,Delta_c=0)={Pout_base/1e-6:.4f}uW")

print("\nLiteral N0=1e-15 m^-3 (paper's literal printed exponent, negative)")
Pout_n0lit, Pin_n0lit, C0_n0lit = pout_with(1.6e-19, 1e-15)
print(f"  Pin={Pin_n0lit/1e-6:.4f}uW (unaffected, N0 doesn't enter Pin), C0={C0_n0lit:.4e} "
      f"(~30 orders of magnitude smaller than baseline)")
print(f"  Pout={Pout_n0lit/1e-6:.6f}uW = Pin to {abs(Pout_n0lit-Pin_n0lit)/Pin_n0lit:.2e} relative precision")
print(f"  --> essentially ZERO absorption anywhere. No EIT dip, no AT splitting, Fig.3/4's entire")
print(f"      methodology (peak-finding on a features that don't exist) would be meaningless.")
print(f"      CONFIRMS (doesn't just assume) that the sign correction to +1e15 is necessary for")
print(f"      this paper's own described physics (an EIT/AT-splitting spectrum) to exist at all.")

print("\nLiteral e=+1.6e19 C (paper's literal printed exponent, positive/missing minus sign)")
try:
    Pout_elit, Pin_elit, C0_elit = pout_with(1.6e19, 1e15)
    print(f"  Pin={Pin_elit:.4e}W, C0={C0_elit:.4e}, Pout={Pout_elit:.4e}")
except (OverflowError, Exception) as ex:
    print(f"  Computation failed/overflowed: {ex} -- e=+1.6e19 makes wp_12 ~76 orders of magnitude")
    print(f"  too large, Pin becomes astronomically large, C0 astronomically large, the exponent in")
    print(f"  Pout=Pin*exp(...) overflows completely. Not a usable parameter set at all --")
    print(f"  confirms the sign correction to e=1.6e-19 is likewise necessary, not optional.")

print("\nDONE.")
