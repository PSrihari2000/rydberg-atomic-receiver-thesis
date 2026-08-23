# ============================================================
# INDEPENDENT, QuTiP-FREE VERIFICATION
#
# Solves the master equation steady state (paper Eq. 5-7) for
# ONE point (Omega_RF/2pi=6MHz, Delta_c=0 -- the same point
# already sanity-checked with QuTiP in fig3_hamiltonian_qutip.py)
# using ONLY plain NumPy linear algebra -- no qutip import at
# all. Then re-solves the SAME point with QuTiP, side by side,
# to prove both methods agree on the physical observable (Pout).
#
# This directly demonstrates what steadystate() does internally:
# build the 16x16 Liouvillian superoperator from the 4x4 H and
# collapse operators, vectorize rho, solve a linear system with
# the Tr(rho)=1 constraint. No stochastic simulation, no fitting.
# ============================================================

import numpy as np

# ------------------------------------------------------------
# Same physical parameters as fig3_hamiltonian_qutip.py
# ------------------------------------------------------------

e_charge = 1.6e-19
a0 = 5.2e-11
hbar = 1.054571817e-34
eps0 = 8.854e-12
eta0 = 377.0

L_cell = 1.0e-2
N0 = 1.0e15

gamma2 = 2.0 * np.pi * 5.2e6
gamma3 = 2.0 * np.pi * 3.9e3
gamma4 = 2.0 * np.pi * 1.7e3

Omega_p = 2.0 * np.pi * 8.0e6
Omega_c = 2.0 * np.pi * 1.0e6

wp_RF = -1443.459 * e_charge * a0
wp_12 = (2.5 * e_charge * a0) ** 2

lambda_p = 852e-9
kp = 2.0 * np.pi / lambda_p

d_probe = 0.76e-3
Pin = np.pi / (2.0 * eta0) * (d_probe * Omega_p * hbar / (2.0 * np.sqrt(wp_12))) ** 2
C0 = -2.0 * N0 * wp_12 / (eps0 * hbar * Omega_p)

# Test point (identical to the sanity check in fig3_hamiltonian_qutip.py)
Omega_RF_test = 2.0 * np.pi * 6.0e6
Delta_c_test = 0.0
Delta_p_test = 0.0
Delta_RF_test = 0.0

print("=" * 70)
print("INDEPENDENT NUMPY-ONLY MASTER-EQUATION SOLVE (no QuTiP)")
print("=" * 70)
print(f"Test point: Omega_RF/2pi = 6 MHz, Delta_c = 0")

# ------------------------------------------------------------
# STEP 1: build H (4x4), exactly matching paper Eq.(6) /
# hamiltonian() in fig3_hamiltonian_qutip.py, but as a plain
# numpy array instead of a qutip.Qobj.
# ------------------------------------------------------------

H = np.zeros((4, 4), dtype=complex)
H[1, 1] = -Delta_p_test
H[2, 2] = -(Delta_p_test + Delta_c_test)
H[3, 3] = -(Delta_p_test + Delta_c_test + Delta_RF_test)
H[0, 1] = H[1, 0] = Omega_p / 2.0
H[1, 2] = H[2, 1] = Omega_c / 2.0
H[2, 3] = H[3, 2] = Omega_RF_test / 2.0

print("\nHamiltonian H (rad/s), rows/cols = |1>,|2>,|3>,|4>:")
print(np.round(H.real / 1e6, 3), " (x 1e6 rad/s, real part -- H is real-symmetric here)")

# ------------------------------------------------------------
# STEP 2: collapse operators C2, C3, C4 (paper Eq.7), plain
# numpy 4x4 matrices.
# ------------------------------------------------------------

C2 = np.zeros((4, 4), dtype=complex); C2[0, 1] = np.sqrt(gamma2)   # |1><2|
C3 = np.zeros((4, 4), dtype=complex); C3[1, 2] = np.sqrt(gamma3)   # |2><3|
C4 = np.zeros((4, 4), dtype=complex); C4[2, 3] = np.sqrt(gamma4)   # |3><4|
collapse_ops = [C2, C3, C4]

# ------------------------------------------------------------
# STEP 3: build the 16x16 Liouvillian superoperator L, using
# the standard vectorization identity vec(A X B) = (B^T kron A) vec(X),
# with column-major (Fortran-order) vectorization of rho.
#
#   rho_dot = -i[H,rho] + sum_k ( Ck rho Ck^dag - 1/2{Ck^dag Ck, rho} )
#
# vec(-i[H,rho])        = -i (I kron H - H^T kron I) vec(rho)
# vec(Ck rho Ck^dag)    =  (Ck^* kron Ck) vec(rho)
# vec(-1/2 Ck^dag Ck rho) = -1/2 (I kron Ck^dag Ck) vec(rho)
# vec(-1/2 rho Ck^dag Ck) = -1/2 ((Ck^dag Ck)^T kron I) vec(rho)
# ------------------------------------------------------------

I4 = np.eye(4, dtype=complex)
Lsuper = -1j * (np.kron(I4, H) - np.kron(H.T, I4))

for Ck in collapse_ops:
    CkdCk = Ck.conj().T @ Ck
    Lsuper += np.kron(Ck.conj(), Ck)
    Lsuper -= 0.5 * np.kron(I4, CkdCk)
    Lsuper -= 0.5 * np.kron(CkdCk.T, I4)

print(f"\nLiouvillian superoperator L built: shape {Lsuper.shape} (16x16, complex)")

# ------------------------------------------------------------
# STEP 4: solve L . vec(rho) = 0 subject to Tr(rho)=1.
# Replace the last row of L with the trace constraint, solve
# the resulting linear system directly (np.linalg.solve --
# exactly the "direct" method QuTiP itself uses).
# ------------------------------------------------------------

A = Lsuper.copy()
b = np.zeros(16, dtype=complex)

# Trace constraint: vec(rho)[0]+vec(rho)[5]+vec(rho)[10]+vec(rho)[15] = 1
# (column-major vec: index k = i + 4*j corresponds to rho[i,j]; the
# diagonal rho[i,i] sits at k = i + 4*i = 5*i, i.e. k=0,5,10,15)
trace_row = np.zeros(16, dtype=complex)
trace_row[[0, 5, 10, 15]] = 1.0

A[-1, :] = trace_row
b[-1] = 1.0

vec_rho = np.linalg.solve(A, b)
rho_manual = vec_rho.reshape((4, 4), order="F")   # column-major un-vectorization

# ------------------------------------------------------------
# STEP 5: sanity checks on the manually-solved rho
# ------------------------------------------------------------

residual = Lsuper @ vec_rho   # should be ~0 in the 15 rows we did NOT overwrite
residual_physical = np.delete(residual, -1)   # drop the overwritten trace row

trace_val = np.trace(rho_manual)
hermiticity_error = np.max(np.abs(rho_manual - rho_manual.conj().T))
eigvals = np.linalg.eigvalsh((rho_manual + rho_manual.conj().T) / 2)

print(f"\nSanity checks on the manually-solved rho:")
print(f"  Master-equation residual (max |L.vec(rho)| over the 15 real physical rows) "
      f"= {np.max(np.abs(residual_physical)):.3e}")
print(f"  Trace(rho) = {trace_val:.10f}")
print(f"  Hermiticity error = {hermiticity_error:.3e}")
print(f"  Eigenvalues (should be in [0,1], summing to 1) = {np.round(eigvals, 6)}")

rho21_manual = rho_manual[1, 0]
chi_manual = C0 * rho21_manual
exponent_manual = -kp * L_cell * np.imag(chi_manual)
Pout_manual = Pin * np.exp(exponent_manual)

print(f"\nrho21 (manual, numpy-only) = {rho21_manual}")
print(f"Pout (manual, numpy-only)  = {Pout_manual/1e-6:.6f} microW")

# ------------------------------------------------------------
# STEP 6: NOW solve the identical point with QuTiP, side by side
# ------------------------------------------------------------

import qutip as qt

k1, k2, k3, k4 = [qt.basis(4, i) for i in range(4)]
H_qt = qt.Qobj(H)
c_ops_qt = [qt.Qobj(C2), qt.Qobj(C3), qt.Qobj(C4)]
rho_qt = qt.steadystate(H_qt, c_ops_qt, method="direct")
rho21_qt = complex(rho_qt[1, 0])
chi_qt = C0 * rho21_qt
Pout_qt = Pin * np.exp(-kp * L_cell * np.imag(chi_qt))

print("\n" + "=" * 70)
print("SIDE-BY-SIDE COMPARISON: manual numpy solve vs. QuTiP steadystate()")
print("=" * 70)
print(f"{'Quantity':<20}{'Manual (numpy)':<28}{'QuTiP':<28}")
print(f"{'rho21':<20}{str(rho21_manual):<28}{str(rho21_qt):<28}")
print(f"{'Pout (microW)':<20}{Pout_manual/1e-6:<28.8f}{Pout_qt/1e-6:<28.8f}")

rel_diff = abs(Pout_manual - Pout_qt) / abs(Pout_qt)
print(f"\nRelative difference in Pout: {rel_diff:.3e}")
print("CONCLUSION: " + (
    "IDENTICAL to numerical precision -- QuTiP's steadystate() is doing exactly "
    "the linear-algebra solve described above, confirmed independently."
    if rel_diff < 1e-8 else
    "DISCREPANCY -- investigate vectorization convention."
))
