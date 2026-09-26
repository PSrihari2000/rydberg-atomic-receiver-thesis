# Fig. 5 math report -- d=0.36mm independent rebuild

Identical method to `fig5_fresh_build/fig5_quutip.py` (fresh, independent QuTiP
sweep over Omega_total, gamma3=gamma4=0 and Delta_p=Delta_c=Delta_LO=0 per
Sec.III-B's LO-dressed-specific approximation) -- ONLY d_probe changed from
0.76mm to 0.36mm. Does not touch fig5_fresh_build or the older qutip_d036mm files.

## Result

    Pin = 8.7678 microW (vs 0.2244x the 0.76mm value, by the proven d^2 scaling)
    n_qutip_solves = 160
    Linear dynamic range: [1.0344,7.3627]MHz, R2=0.998178
    kappa (slope) = -2.364889e-07 W/MHz

## Comparison to d=0.76mm fresh_build

fig5_fresh_build's own kappa = -1.053982e-06 W/MHz, linear range [1.0344,7.3627]MHz.
Predicted kappa ratio (this project's d^2 scaling law, algebraically proven and
QuTiP-confirmed in fig6_fresh_build/fig6_angle4_probe_diameter.py): (0.36/0.76)^2 =
0.2244. Actual ratio this run: 0.2244.
