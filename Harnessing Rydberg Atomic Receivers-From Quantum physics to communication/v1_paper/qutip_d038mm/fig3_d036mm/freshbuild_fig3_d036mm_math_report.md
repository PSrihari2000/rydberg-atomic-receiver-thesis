# Fig. 3 math report -- d=0.36mm independent rebuild

Identical physics/parameters/grid to `fig3_fresh_build/fig3_hamiltonian_qutip.py`
(paper Sec.IV, Eqs.2-7) -- the ONLY change is d_probe = 0.36mm instead of the
paper-literal 0.76mm. Every grid point is a fresh QuTiP steady-state solve.
Does not touch fig3_fresh_build or the older qutip_d036mm files.

## Hamiltonian (rotating frame, rad/s)

    H = -Delta_p|2><2| - (Delta_p+Delta_c)|3><3| - (Delta_p+Delta_c+Delta_RF)|4><4|
        + (Omega_p/2)(|1><2|+|2><1|) + (Omega_c/2)(|2><3|+|3><2|) + (Omega_RF/2)(|3><4|+|4><3|)

Steady state via QuTiP `steadystate` (direct method), collapse operators
sqrt(gamma2)|1><2|, sqrt(gamma3)|2><3|, sqrt(gamma4)|3><4|. Delta_p=Delta_RF=0.

Pout = Pin * exp(-kp*L*Im(C0*rho21))   (Eq.2/4)

## Parameters (paper Sec.IV, only d_probe changed)

    L=1cm, N0=1e15 m^-3, d_probe=0.36mm (vs paper-literal 0.76mm)
    Omega_p/2pi=8.0MHz, Omega_c/2pi=1.0MHz
    gamma2/2pi=5.2MHz, gamma3/2pi=3.9kHz, gamma4/2pi=1.7kHz
    wp_RF=-1443.459 e*a0, wp_12=(2.5 e*a0)^2, lambda_p=852nm

## Grid

    Delta_c/2pi: [-20,20]MHz, 201 points
    Omega_RF/2pi: [0,20]MHz, 161 points
    Total: 32361 fresh QuTiP steady-state solves, 2137.4s (66.05ms/solve)

## Results (this run)

    Pin = 8.767765e-06 W = 8.7678 microW
    C0 = -1.843619e-05
    Sanity check (Omega_RF/2pi=6MHz, Delta_c=0): Pout=7.548680microW,
        Tr(rho)=1.0, hermiticity_error=0.000e+00
    Peak Pout = 8.5919 microW at Omega_RF/2pi=0.125MHz,
        Delta_c/2pi=0.000MHz

## Comparison to the 0.76mm fresh_build

Pin scales as d_probe^2 (pure prefactor on Pout, proven algebraically and confirmed
numerically throughout this project -- see project memory). Expected ratio:
(0.36/0.76)^2 = 0.2244. Actual Pin ratio this run: to be compared against
fig3_fresh_build's own Pin once both are available side by side.
