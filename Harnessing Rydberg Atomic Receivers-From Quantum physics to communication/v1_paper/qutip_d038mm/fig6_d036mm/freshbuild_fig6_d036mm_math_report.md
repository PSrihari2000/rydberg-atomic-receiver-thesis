# Fig. 6 math report -- d=0.36mm independent rebuild

Identical method to `fig6_fresh_build/fig6_snr_vs_distance.py` (Conventional / Theoretical
LO-dressed / Practical LO-dressed / LO-free, same equations, same wide-QuTiP-sweep approach
for Practical LO-dressed) -- reusing THIS folder's own `freshbuild_fig4_d036mm.csv` threshold
and `freshbuild_fig5_d036mm.csv` LO-dressed data. ONLY d_probe changed from 0.76mm to 0.36mm.
Does not touch fig6_fresh_build or the older qutip_d036mm files.

## Result

    Fig.4-d036mm threshold used: 5.1250 MHz
    kappa = -3.763838e-14 W/(rad/s), P0_bar = 8.0034 microW
    A_LO = 2.778820e-08, sigma_Ry_LO^2 = 2.937719e-14
    Checkpoint @ d=100m, PTx=-10dBm: Conventional=-23.3293dB, Theoretical LO-dressed=-3.3202dB,
        gap=20.0092dB

## Comparison to d=0.76mm fresh_build

d=0.76mm fresh_build's real checkpoint gap = 28.8864dB (using the same GRx*GLNA-on-noise-only
Conventional formula, and the fRF=3.5GHz paper-stated value -- the fRF consistency fix from
`fig6_fresh_build/fig6_angle6_fRF_consistency_fix.py` is NOT applied here, for direct
apples-to-apples comparison with the main 0.76mm build's own real result). Conventional SNR is
diameter-independent (no d_probe dependence anywhere in its formula) -- any gap change here is
entirely due to the LO-dressed side's kappa/A_LO, which scale with probe diameter as shown in
`fig6_fresh_build/fig6_angle4_probe_diameter.py` (A_LO proportional to d_probe^4, confirmed
against real QuTiP data at both diameters).
