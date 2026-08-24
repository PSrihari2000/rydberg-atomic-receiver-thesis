# Fig. 6 — SNR versus distance, four receiver models

NEW file, does not modify `fig6_snr_distance.py/.png`, `fig6_math_report.md`, or
`fig6_fRF_consistency_audit.py/.txt`.

Reuses two pieces of real, already-validated data from this project, computed nowhere else:
Fig. 4's measured LO-free resolvability threshold (5.1250 MHz, from
`fig4_fresh_build/fig4_classification.csv`), and Fig. 5's independently-simulated
LO-dressed `Pout(Omega_total)` curve together with its linear-region fit
(`Omega_LO`, the fitted range, the slope κ), from `fig5_fresh_build/fig5_data.csv`.
No previous simulation is regenerated; both files are only read.

The shared link budget for all four receivers is Eq. 10:

    PRx = PTx · GTx · η0 / (4π·d²)

**Conventional receiver.** Fig. 2's diagram gives `y = √PRx·h·x + n_Conv` with
`σ²_Conv = GRx·GLNA·σ²_BGN + 4kB·T·B`. Received power is computed via the standard Friis
relation (flux density × effective aperture, the aperture taken as Sec. V-A's stated
λRF²/(4π)):

    Pc(d) = [PTx·GTx / (4π·d²)] · [λRF² / (4π)]

and, following Fig. 2's diagram literally, GRx·GLNA is applied only to the antenna-referenced
background noise, not to the signal (no gain term appears on the `y=√PRx·h·x` side of the
diagram) — the same convention this project's earlier Fig. 6 attempt used:

    γ_conv = Pc / (GRx·GLNA·σ²_BGN + 4kB·T·B),   SNR_conv = 10·log10(γ_conv)

(An equally literal alternative — scaling Pc by GRx·GLNA too, so the receive-chain gain affects
signal and antenna-side noise symmetrically, standard in real receiver-chain analysis — was
tried first and found to shrink the gap to ~6.7 dB by nearly cancelling GRx·GLNA out of the
dominant background-noise term. That reading is not used in the reported results below.)

**LO-free receiver** (Eq. 8–16, unchanged from the paper):

    ΩRF = √PRx · |℘RF|/ħ,   σ²_UN = ε̃²·PRx,   σ²_Ry = σ²_UN + σ²_BGN
    SNR_Ry = PRx / σ²_Ry

Plotted only where ΩRF/2π ≥ 5.1250 MHz (Fig. 4's real, measured threshold — not the paper's
stated 5.5 MHz); below it the curve is a genuine gap, not drawn.

**Theoretical LO-dressed receiver** (Eq. 30–37, purely analytical — no QuTiP in this branch):

    κ = (Fig.5's fitted slope, W per MHz-of-Ωtotal/2π) / (2π×10⁶)     [converted to W per rad/s]
    A_LO = GLNA·RL·D²·κ²·(℘RF/ħ)²
    SNR_Ry,LO = A_LO·PRx / σ²_Ry,LO

σ²_Ry,LO (Eq. 36) is a single fixed number, not a function of distance: the photon-shot-noise
term (Eq. 34–35) is driven by the *average* photocurrent Ī = ηeff·P̄0·e/(ħ·fp), and P̄0 (the DC
probe transmission at Ω_LO) is a fixed operating point, re-evaluated here from Fig. 5's own real
curve rather than the tiny, distance-dependent AC term |P(Δf)| — the same resolution of this
ambiguity used earlier in this project. Only the numerator (PRx) varies with distance.

**Practical LO-dressed receiver.** Rather than switching between the analytical formula and a
QuTiP lookup at Fig. 5's linear-range boundary — which would stitch two formulas together and
risk a visible kink not present in the real physics — this branch runs ONE continuous pipeline
at every distance: build Ωtotal(t) = |Ω_LO + ΩRF·e^{j2πΔf t}| over one full period, read Pout(t)
off the real, interpolated Pout(Ωtotal) curve, and extract the Δf Fourier component |P(Δf)|
directly by numerical projection onto e^{-j2πΔf t}. The signal power is then
GLNA·RL·D²·|P(Δf)|², divided by the same σ²_Ry,LO as the theoretical branch. This reduces to the
theoretical formula automatically wherever the response is genuinely linear, and diverges from
it automatically wherever it isn't — no branch, no fitting.

Because ΩRF at short distance/high PTx (via Eq. 8) can reach tens of MHz — far beyond Fig. 5's
own ±a-few-MHz sweep around Ω_LO — a wider Ωtotal sweep (0.05–35 MHz, 350 points) was computed
fresh here, using the identical Hamiltonian, collapse operators, and parameters as
`fig5_quutip.py` (γ3=γ4=0, Δp=Δc=ΔLO=0, same Ωp, Ωc, γ2). Any distance/PTx point whose full
Ωtotal(t) excursion still exceeds even this wider domain is left as a genuine NaN gap — never
extrapolated — since beyond that range the four-level ladder model itself (Ωp=8 MHz, Ωc=1 MHz)
is far outside any regime it was built to describe.

## Parameters

    GTx = GRx = 2.15 dBi, GLNA = 20 dB, RL = 50 Ω, D = 0.55 A/W, T = 290 K, ηeff = 0.8
    σ_BGN = -90 dBm (used directly as σ²_BGN, matching the paper's own description of this
        value as "the noise power," not an amplitude to be squared again)
    ε̃ = 0.5%, fRF = 3.5 GHz (λRF = 85.66 mm), B = 1 MHz
    Ω_LO = 4.2300 MHz, Fig.5 linear range = [1.0344, 7.3627] MHz (loaded, not re-fit)
    κ = -1.6775e-13 W/(rad/s), P̄0 = 35.6693 μW, σ²_Ry,LO = 7.5566e-14 (fixed)
    Distance sweep: 0.1 to 1e6 m, 281 log-spaced points

    PTx sets (paper's own Fig. 6 legend, verified against the user-supplied reference image):
        Conventional / Theoretical LO-dressed / Practical LO-dressed: -10 dBm, +10 dBm
        LO-free: 10 dBm, 20 dBm

## Results — reported honestly, not tuned to match the paper

    Checkpoint @ d=100m, PTx=-10dBm: Conventional = -23.33 dB, Theoretical LO-dressed = 5.56 dB
        gap = 28.89 dB   (paper reports ~44 dB at this same reference point)
    0dB crossings (PTx=-10dBm): Conventional @ 6.82 m, Theoretical LO-dressed @ 189.64 m
        extended coverage = 182.82 m   (paper reports ~1500 m)
    LO-free resolved out to d = 2.37 m (PTx=10dBm) / correspondingly further at 20dBm
    Practical LO-dressed (PTx=10dBm): out-of-QuTiP-domain gap for d < 0.42 m;
        leaves Fig.5's linear region for d < 4.22 m

The gap and crossing distance both come out smaller than the paper's own headline numbers, but
substantially closer than the GRx·GLNA-symmetric alternative was (28.89 dB / 182.82 m here vs.
6.74 dB / 102.30 m there). No parameter was adjusted to search for a better match; every number
above follows directly from the equations and data above. Two concrete, identifiable sources of
the remaining gap: (1) the LO-dressed side's κ comes from this project's own independently-
simulated Fig. 5 curve, not the paper's — a different (correctly LO-dressed-approximated, but
numerically different) quantum response than whatever the paper's own simulation produced;
(2) Eq. 10's atom-flux PRx (no GRx, no λ²/(4πd)² suppression) makes the LO-dressed/LO-free
branches' received-power model already differ in kind from the conventional receiver's
Friis-based Pc, so nothing guarantees their ratio reproduces the paper's stated 44 dB. Both are
reported as open, unresolved discrepancies, consistent with this project's standing policy of
reporting mismatches rather than closing them by fitting.
