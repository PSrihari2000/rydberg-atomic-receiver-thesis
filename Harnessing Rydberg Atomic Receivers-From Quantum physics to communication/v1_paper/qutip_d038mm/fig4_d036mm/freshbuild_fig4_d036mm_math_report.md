# Fig. 4 math report -- d=0.36mm independent rebuild

Re-analysis of THIS folder's own `freshbuild_fig3_d036mm_raw.npz` -- no new QuTiP
solves (Fig.4 is a top-down view of Fig.3, per the paper's own description). Same
FWHM-based resolvability criterion as `fig4_fresh_build` (separation >= 1 measured
FWHM of the Omega_RF=0 spectrum) -- plainly stated, not named "Rayleigh criterion."
Does not touch fig4_fresh_build or the older qutip_d036mm files.

## Result

    Measured HWHM = 2.5960 MHz -> FWHM = 5.1920 MHz
    Resolved rows: 120/161
    THRESHOLD (lowest resolved Omega_RF) = 5.1250 MHz

## Comparison to d=0.76mm

The d=0.76mm fresh_build threshold is 5.1250 MHz. Probe diameter enters Fig.3's
Pout only through Pin, a pure multiplicative prefactor with zero effect on where
peaks/valleys occur (proven algebraically and confirmed by direct byte-for-byte
comparison in the earlier qutip_d036mm/qutip_d076mm investigation this project
already did) -- so this threshold is expected to match 5.1250MHz exactly, and this
run independently re-confirms that on freshly-computed d=0.36mm data using TODAY's
FWHM-based method (the earlier investigation used the OLDER 5.5MHz-era method).
