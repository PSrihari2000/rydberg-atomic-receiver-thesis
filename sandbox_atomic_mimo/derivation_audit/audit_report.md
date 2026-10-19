# Xu/Cui derivation audit, 14 October 2026

## Verdict and scope

The supplied 84-page derivation is a substantial and mostly correct pedagogical expansion, but it should not be described as completely error-free. Its later sections honestly distinguish several corrections and extra assumptions from the source papers. Two confirmed local errors need correction: the missing scalar in the second equality of Eq.(378), and the complex-amplitude/real-electric-field convention in the early physical derivation. Additional qualifications are needed for paper-compatible likelihood/CRLB labels and general algorithm statements. The separate EM-GS LaTeX source contains a confirmed SNR/RSR arithmetic error and needs a more explicit distinction between a complex FIM block and the full real/augmented information.

Read: all 84 derivation PDF pages; all 5 Xu PDF pages; all 15 Cui PDF pages; the current standalone em_gs_derivation.tex. Equations at critical locations were additionally inspected as rendered page images. This was a full textual/algebraic audit with targeted visual verification, not a certification that every rendered glyph on every page was visually compared. The 84-page PDF's corresponding main LaTeX source is absent from the current working directory; the compiled PDF is the source of its page/equation references. A LaTeX engine was not available to compile the separate EM-GS source.

The papers are the supplied published editions: Xu, IEEE WCL 14(9), 2957-2961, DOI 10.1109/LWC.2025.3584040; Cui, IEEE JSAC 43(3), 659-673, DOI 10.1109/JSAC.2025.3531528. File hashes are recorded in source_hashes.json.

## What is sound in the 84-page derivation

- The chosen excited/ground basis, free Hamiltonian, operator expansion under definite parity, interaction-picture transformation and resonant two-level ODE solution are internally consistent once a physically valid carrier/phasor convention is supplied. The probability sin^2(Omega*t/2) uses initial ground-state preparation and exact resonance; it is not the general detuned formula in Cui Eq.(7).
- The magnitude-model factorization and first-order strong-reference approximation, Eqs.(95)-(134), are correct for nonzero b and a small normalized perturbation. Page 31 explicitly acknowledges that retaining a quadratic term from the first-order square-root expansion is not a complete second-order expansion.
- The real rotated noise variance sigma_r^2=sigma_complex^2/2 in Eq.(131a)/(391)/(438) is correct.
- The dimensions, entrywise phase multiplication and linearized LS objective, Eqs.(135)-(206), are correct. Eq.(206) is the Wirtinger derivative with respect to G*, without the factor 2 of the equivalent real-coordinate gradient.
- The ordinary-transpose pilot model S^T*g_i and A=S* mapping, Eqs.(229)-(257), are correct. They preserve the meaning of each user's channel coefficient rather than silently conjugating the unknown.
- Biased GS spectral weighting, scale fit, phase alignment, per-cell LS update and full-matrix LS update, Eqs.(258)-(369), match the useful content of Cui Algorithm 1 after the channel-estimation mapping. Scale fitting assumes a nonzero predicted magnitude norm; phase alignment assumes the final eigenvector coordinate is nonzero. Degenerate cases need a stated fallback in executable code.
- The rank bound is correctly written as rank(G_k)<=min(L_k,I1,I2), not unconditional equality. Eqs.(373)-(382) explicitly require path polarization invariant across the aperture. That extra assumption is essential and is honestly stated; it is not established by Xu's cell-indexed random-polarization simulation description.
- Column-major mode-3 mapping, Eqs.(398)-(404), and the complex gradient, Eq.(412), are correct. Page 72 identifies the factor-of-two convention in Xu Eq.(15).
- The unprojected step-size proof S1-S9 is correct with L=lambda_max(S*S^H) and a positive step eta<=1/L. It proves the stated unprojected decrease, not by itself a global convergence theorem for nonconvex PGD. A projected-step decrease argument also needs a feasible starting iterate and an exact projection.
- The complex SVD uses V^H in Eqs.(424)/(426), and page 78 explicitly identifies Xu's V^T printing under the standard SVD convention.
- Eqs.(458)-(464b) provide the correct real-parameter FIM, its inverse when identifiable, and the augmented complex information structure. Eq.(466) correctly gives P>=2K as necessary dimension counting while additionally requiring full column rank of the real design matrix D.

## Corrections/qualifications needed in the 84-page derivation

| Location | Finding | Required revision |
|---|---|---|
| pp.9-15, especially Eqs.(15), (16), (26); explanation on p.14; propagated to Eqs.(36)-(40) | alpha_lk and s_k are complex, but alpha_lk*s_k*cos(omega*t+phase) is not a real physical electric field. The p.14 shorthand explanation does not fix the Fourier coefficients used later. Real dipole/polarization vectors do not make a complex scalar times cosine real. | Write an explicit real part of a complex carrier phasor, or use abs(alpha*s)*cos(carrier+spatial phase+symbol/path phase). Declare the carrier-sign convention and consistently track conjugations through the RWA and channel definition. |
| p.67, Eq.(378), second equality | The first formula G_k^(l)=beta_lk*a1*a2^T is correct, but the adjacent dimensional equality writes a1*a2^T=G_k^(l), omitting beta_lk. | Include beta_lk in the second product, or use a dimension-membership statement instead of equality. |
| pp.80-81, Eqs.(440), (453)-(454) and paper-equation attribution | These equations use corrected Gaussian normalization and a consistent sigma_r^2 convention. They are mathematically coherent, but are not literal replicas of Xu's printed Gaussian prefactor or unchanged use of sigma^2. | Explicitly label the normalization correction and the noise-variance reinterpretation. The paper does not explicitly state that it renames sigma_complex^2/2 as sigma^2. |
| p.78, algorithm (429a) | The text correctly says the projection is identity if L_k>=min(I1,I2), but the displayed general pseudocode still slices 1:L_k without a branch or clipping. | Use r_k=min(L_k,I1,I2) or an explicit identity-projection branch. The original 8x8, L=3..7 setting stays inside the permitted range. |
| p.84, discussion of a high-SNR floor | The claim that the model mismatch appears as a floor for any estimator is too absolute. An exactly real perturbation aligned with a positive reference has abs(b+a)=b+a, so it has no noiseless first-order mismatch. | Say finite-reference linearization can cause a persistent bias/floor for generic complex fields; it is not guaranteed for every channel or pilot/reference design. |
| p.84, constrained CRLB discussion | r*(I1+I2-r) complex degrees of freedom is the smooth rank-exactly-r manifold count. Counting alone does not derive a constrained bound, and rank-at-most-r is stratified at lower-rank points. | State the regular-point condition and use the appropriate tangent-space/Jacobian FIM if a constrained bound is actually needed. |
| pp.53-55, augmented phase ambiguity | Phase ambiguity belongs to the temporarily unconstrained augmented spectral problem. The original biased problem fixes the last augmented coordinate to 1. Phase rotation alone makes the approximate last coordinate real, not exactly 1. | Clarify that the initial direction is approximate and the actual reference remains known/fixed during the subsequent fits. This is not an error in the reproduced Cui initialization. |

An explicit counterexample to the p.14 claim: take real dipole/polarization couplings, two complex path coefficients c=[1,j], and spatial phases [0,pi/2]. The written C=sum(c_l*exp(-j*phase_l)) has magnitude 2. For a real positive-carrier field Re(sum(c_l*exp(j*(omega*t+phase_l)))), the resonant coefficient is sum(conj(c_l)*exp(-j*phase_l)), whose magnitude is zero. Thus the claim that restoring the real field cannot change abs(C) is not generally valid. One can preserve the desired negative-spatial-phase C by explicitly choosing the corresponding negative-carrier convention; the choice must be written, not inferred from the complex-cosine shorthand.

## Source-paper issues, separated from local mistakes

| Source location | Issue | How the local derivation handles it |
|---|---|---|
| Xu Eqs.(3)-(4), (6)-(7) | Complex symbol/path-amplitude cosine shorthand and a change between dipole transpose/conjugate-transpose and spatial-phase signs need consistent carrier/phasor definitions. These changes are not universally interchangeable for complex multiuser symbols. | The current p.14 explanation is insufficient; this part needs revision. |
| Xu Eqs.(15)/(24) | Different gradient scaling/notational conventions, including factor 2, appear between the 1D and 2D expressions. | Explicitly explained on derivation p.72. Both scales can be used with the corresponding step size. |
| Xu Eq.(16) and following rank statement | Cell-varying random polarization does not imply a rank-one path matrix; path count gives a rank upper bound rather than guaranteed equality. | Correctly supplemented by Eq.(374) and bounded in Eq.(382). |
| Xu feasible set below Eq.(24) | The paper writes a real-valued feasible set even though its channel and updates are complex. | Eq.(419) uses the appropriate complex domain, but could explicitly call out this correction. |
| Xu Eqs.(26)-(27) | Standard complex SVD requires V^H; the printed equations use V^T. | Explicitly corrected on p.78. A nonstandard definition of V could make a transpose usable, but the paper does not supply that convention. |
| Xu Eq.(28) | The Kronecker ordering I_Na tensor S^T agrees with vec(G_(3)), not conventional column-major vec of the original I1 x I2 x K tensor. | Eqs.(431)-(434) explicitly choose the compatible unfolded-vector order. |
| Xu Eq.(11) versus Eq.(28) | Real rotated noise is assigned sigma^2/2 in Eq.(11), but the same sigma^2 symbol is used as the real noise variance in Eq.(28) without an explicit redefinition. | sigma_r^2 is introduced correctly. The paper-variance reinterpretation needs clearer labeling. |
| Xu Eq.(29) | The printed real Gaussian normalizer has power M instead of M/2, where M=I1*I2*P. | Eq.(440) uses the correct power M/2. This prefactor is parameter-independent, so fixing it does not change the gradient, but it matters for a normalized pdf. |
| Xu Eqs.(30)-(31) | Inverting only one complex information block is generally weaker than the full real/augmented CRLB for real observations. | The 84-page PDF honestly adds the exact real FIM and off-diagonal blocks in Eqs.(458)-(464b). |
| Cui Eq.(26), surrounding spectral text | Some expressions use A^H with an augmented unknown; the augmented matrix bar(A)^H is dimensionally required. Algorithm 1 gives the consistent augmented construction. | The 84-page spectral derivation uses the correct augmented dimensions. |
| Cui Eq.(29), published p.667 | The displayed exponent lacks the squared residual. The polar joint density also needs Jacobian z. The squared residual is used in Appendix B and Lemma 1. | The separate EM-GS source explicitly identifies and corrects both issues at lines 381-384. |
| Cui Eq.(28) | Q alone is not generally a lower bound of log p for continuous latent phase; the entropy term is needed. | The separate EM-GS source correctly includes Q+H and proves the ascent property. |
| Cui Eqs.(33)-(34) | The complex FIM block I is not generally the complete augmented information; cross-score pseudo-covariance need not vanish. Its inverse is a potentially loose block-inverse lower benchmark, not generally the full real CRLB. | The separate EM-GS source reproduces the paper's block but does not develop this qualification/full bound. |

A block-inverse bound can remain a valid weaker information bound under regular unbiased-estimation conditions. The issue is its identification as the complete exact CRLB, not a claim that every block-inverse inequality is false.

## Separate EM-GS source: confirmed findings

File: Channel_Estimation_for_Rydberg_Atomic_Receivers/Derivation/em_gs_derivation.tex.

1. Joint polar density, Rician marginal, von Mises phase posterior, conditional mean, entropy-aware EM bound, square completion and unconstrained LS M-step are correct. The source clearly separates its own equation numbers from Cui equation numbers. It explicitly states that the QAM constraint is relaxed during the iterations.
2. Line 907 contains a confirmed arithmetic error: kappa is stated to scale as 2*K*RSR*SNR. For equal-power independent users and the stated total-user SNR/single-user RSR definitions, the strong-reference estimate is instead

   kappa ~= 2*P_reference/sigma^2 = 2*RSR*P_single/sigma^2 = (2/K)*RSR*SNR.

   RSR and SNR are in linear units. At K=3, RSR12 dB and SNR0 dB, the correct estimate is 10.566; the text gives 95.094, a factor 9 too large. The underlying EM update kappa=2*z*abs(lambda)/sigma^2 is correct; this error is in its later interpretation.
3. Lines 978-1013 reproduce Cui's I=sum(beta*a*a^H), then call trace(I^-1) the CRLB. Add the full real/augmented distinction. For one complex unknown and one amplitude observation, the scalar complex block can be positive while the 2x2 real FIM is rank one. A finite complex-block inverse does not establish identifiability of both real and imaginary parts.
4. The high-SNR 3 dB discussion and the comparison to Xu at lines 1016-1036 should be labeled as block-inverse/asymptotic comparisons. The exact phase-loss penalty depends on measurement/phase diversity, identifiability, and the full FIM. Xu's printed Eq.(31) has 4*sigma^2; the displayed 2*sigma_complex^2 form uses a corrected variance reinterpretation, not the literal paper expression.
5. At x=0 the integral proof containing 1/x should use continuity or the direct sine integral. Formulas dividing by abs(lambda) likewise need the zero-mean limiting case. The algorithm itself handles zero concentration via R(0)=0.
6. Statements that EM-GS is always better than biased GS should be qualified as the paper's observed simulation behavior, not a finite-sample universal MSE guarantee. EM likelihood ascent alone does not imply every iterate or local optimum has lower channel/symbol NMSE.

The LaTeX reference check found 68 unique labels with no undefined ref/eqref labels. This is a source-reference check, not a compiled layout/numbering verification.

## Equation-number mapping: Xu paper to the expanded PDF

These are corresponding constructions; corrected convention/extra assumptions are indicated above. Expanded equations deliberately have their own numbering.

| Xu equation | Expanded PDF equation(s) |
|---|---|
| 1 | 1 |
| 2 | 2 |
| 3 | 26, with the field-convention qualification |
| 4 | 16, with the field-convention qualification |
| 5 | 66, exact-resonance ground-state case |
| 6 | 72-74, with the phase/dipole convention qualification |
| 7 | 75 |
| 8 | 77 / 94 |
| 9 | 89 |
| 10 | 110-121 |
| 11 | 134 |
| 12 | 163 |
| 13 | 173 |
| 14 | 207 |
| 15 | 206 / 210, stated Wirtinger scaling |
| 16 | 375, supplemented by 374 |
| 17 | 383 |
| 18 | 392 |
| 19 | 397 |
| 20 | 401 |
| 21 | 404 |
| 22 | 406 |
| 23 | 420 |
| 24 | 412 |
| 25 | 421 |
| 26 | 424, standard complex SVD correction |
| 27 | 426 / 428 |
| 28 | 437 with vector order 431-434 |
| 29 | 440, normalization correction |
| 30 | 447-451, explicit real-noise variance |
| 31 | 453-454, block-inverse benchmark; full real bound 462-463a |

Cui Algorithm 1 is expanded as Eq.(369). Its spectral scaling Eq.(26) corresponds to expanded Eq.(279)/(354); its phase alignment Eq.(27) corresponds to expanded Eqs.(287)-(291)/(356)-(357). The 84-page PDF does not contain the full EM-GS development; that is in the separate EM-GS source.

## Numbering and independent numerical checks

The compiled 84-page PDF has 486 distinct displayed tags: main numbering through 466, two pairs 28a/b and 29a/b, the additional lettered tags, and S1-S9. No duplicate displayed tags or missing base-number groups were found; base 28 and 29 are intentionally represented by subparts. Displayed tags remain in order. PDF-tag enumeration cannot by itself verify the intended meaning of every prose reference; the main internal chains and paper mappings were reviewed separately.

Independent numerical checks (numerical_checks.py / numerical_checks.json):

- Gradient directional finite-difference relative discrepancy: 8.23e-10.
- Full-matrix and per-cell GS LS discrepancy: 4.10e-16.
- Mode-3/Kronecker mapping discrepancy: 2.43e-15.
- Step S9 decrease inequality holds on the sampled problem.
- Example real CRLB trace: 11.8046; complex-block inverse trace: 6.6188. The nonzero off-diagonal information explains the difference.
- Example Rician one-measurement complex block: 0.4733; exact real FIM rank: 1 for two real unknowns.
- The phasor and Eq.(378) counterexamples confirm the local issues above.

## Recommended presentation

Use four explicit labels throughout the derivation: paper equation; expanded algebra; added modeling assumption; corrected/reinterpreted paper expression. Preserve the sound GS/GD/PGD/EM steps, repair the confirmed local errors, and retain the exact real-FIM appendix. A mathematically corrected expansion can be honest even when it does not reproduce every printed expression literally; the important requirement is to identify each change and its reason.
