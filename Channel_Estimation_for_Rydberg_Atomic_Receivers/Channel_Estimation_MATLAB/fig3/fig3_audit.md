# Fig.3 Full Audit — Channel Estimation for Rydberg Atomic Receivers

Complete mathematical and implementation audit of the Fig.3 (1D array, NMSE vs SNR,
P∈{10,30}, GS/GD/CRLB) simulation, checked directly against:
- Xu, Zhang, Chen, Cheng, Liu, Wu, Ai, "Channel Estimation for Rydberg Atomic
  Receivers," IEEE WCL, Sept 2025 (`Channel_Estimation Paper.pdf`)
- Cui, Zeng, Huang, "Towards Atomic MIMO Receivers," IEEE JSAC, March 2025
  (`Towards_Atomic_MIMO_Receivers.pdf`)
- Netrapalli, Jain, Sanghavi, "Phase Retrieval using Alternating Minimization,"
  IEEE Trans. Signal Process. 2015 (Xu's ref [14] for the GS algorithm itself)

No code proposed in this document — audit only. Rule followed throughout: do not
modify equations or tune parameters to force a desired curve shape; every claim is
labeled DIRECTLY STATED IN PAPER / MATHEMATICALLY IMPLIED BY PAPER / IMPLEMENTATION
ASSUMPTION / NOT SPECIFIED BY PAPER.

---

## PART 1 — Literal Equation Table

| # | Item | Paper / Eq. | Literal equation | Dimensions | Transpose type | MATLAB equivalent |
|---|---|---|---|---|---|---|
| 1 | Exact observation | Xu Eq.(8) | y(i,p) = \|Σₖ g(i,k)s(k,p) + b(i,p) + n(i,p)\| | scalars; Y∈ℝ^{I×P} | — | `Y = abs(A_signal + B + N)` |
| 2 | Channel model | Xu Eq.(7) | g(i,k) = Σₗ (1/ℏ)μ_eg^T ε(i,k,l)α(l,k)e^{−j(i−1)φ(l,k)} | G∈ℂ^{I×K} | **plain (T)**, not H | loop, see Part 4 |
| 3 | Reference model | Xu Eq.(9) | b(i,p) = (s(b,p)/ℏ)μ_eg^T ε(b,i)α_b e^{−j(i−1)φ_b} | B∈ℂ^{I×P} | plain (T) | loop, see Part 4 |
| 4 | Strong-ref condition | Xu, text before Eq.(10) | \|b(i,p)\| ≫ \|Σₖg(i,k)s(k,p)+n(i,p)\| | scalar, per (i,p) | — | pointwise, see Part 3 |
| 5 | Taylor pre-drop | Xu Eq.(10) | y=\|b\|(1+\|a+n\|²/(2\|b\|²)+Re{(a+n)/b}) | scalar | — | not implemented (dropped term) |
| 6 | Linearized model | Xu Eq.(11) | y(i,p)=Re{e^{−jz(i,p)}Σg(i,k)s(k,p)}+\|b(i,p)\|+n̄(i,p) | scalar; n̄~N(0,σ²/2) | — | — |
| 7 | Matrix form | Xu Eq.(12) | Y=Re{Z∘(GS)}+\|B\|+N | Y∈ℝ^{I×P}, Z∈ℂ^{I×P} | — | `real(Z.*(G*S))` |
| 8 | GD objective | Xu Eq.(13) | argmin_G ‖Y−\|B\|−Re{Z∘(GS)}‖²_F | scalar | — | `norm(...,'fro')^2` |
| 9 | GD update | Xu Eq.(14) | G^{t+1}=G^t−η_t∇_{G*}L(G^t) | G∈ℂ^{I×K} | — | — |
| 10 | GD gradient | Xu Eq.(15) | ∇_GL(G^t)=−2([Y−\|B\|−Re{(G^tS)∘Z}]∘Z*)**S^H** | I×K | **Hermitian (H)** — paper prints S^H explicitly | `(...)*S'` |
| 11 | CRLB vectorized model | Xu Eq.(28) | ȳ=Re{z̄∘(I⊗S^T)ḡ}+\|b̄\|+n̄, n̄~N(0,σ²I) | — | S^T inside Kronecker | — |
| 12 | CRLB result | Xu Eq.(31) | CRB(g)⪰4σ²[I_{I1I2}⊗(S*S^T)^{-1}] | K×K per antenna | conj(S)·S^T | `4*sig2*pinv(conj(S)*S.')` |
| 13 | GS generic model | Cui Eq.(21) | z=\|A^Hs+b+w\| | A∈ℂ^{K×N} | **Hermitian (H)** | — |
| 14 | GS→channel-est. model | Cui Eq.(35) | z_n=\|**S^H**a_n+b_n+w_n\| | S∈ℂ^{K×P}, a_n∈ℂ^{K×1} | **Hermitian (H)** | see Part 6 — this is where it gets subtle |
| 15 | GS spectral init | Cui Eq.(26)‑(27) | r̄=\|v^HĀ\|z/‖Ā^Hv‖², s^0=e^{−i∠s̄⁰_{K+1}}s̄⁰(1:K) | Ā∈ℂ^{(K+1)×P} | Hermitian throughout | eig(M), phase-fix |
| 16 | GS iteration | Cui Alg.1 steps 6‑7 | θ^t=∠(A^Hs^{t−1}+b), s^t=(AA^H)^{−1}A(z∘e^{iθ^t}−b) | — | Hermitian | LS solve |
| 17 | Netrapalli init | Netrapalli Alg.1 | x⁰←top singular vector of Σᵢyᵢ²aᵢaᵢ^T | — | weighted by **yᵢ²** (squared) | not what we use |

**Key finding visible in this table:** row 10 (Xu's own GD gradient, Eq.15) explicitly
prints **S^H**. GD's transpose is not in dispute. The dispute is entirely about GS
(rows 13-14), addressed in Part 6.

---

## PART 2 — What observation does each algorithm actually see?

**Answer: Option A, and it is directly supported by the text — not a guess.**

Xu's own narrative order is unambiguous: Eq.(8) (exact, nonlinear) → strong-reference
assumption → Eq.(10)→(11) (Taylor-approximated) → Eq.(12) (matrix form GD fits).
Nowhere does the paper say GD is *given* a synthetically-linear measurement — Eq.(12)
is described as "This is the central equation **used by the GD method**" (i.e., GD's
fitting model), not as how the receiver generates data. The receiver only ever
produces Eq.(8)'s magnitude.

Confirming this further: Xu explicitly contrasts the two algorithms in the Fig.3
discussion — "the GS algorithm does not rely on an approximation of the channel
transmission model" — this sentence only makes sense if GS is being evaluated
against the *same* exact physical data GD is also fed, with GD accepting an
approximation error GS doesn't. If GD instead got a separately-manufactured "already
linear" measurement, that sentence would be meaningless (GD would have zero
approximation error by construction, and the paper's own explanation for the P=10
gap would be false).

**Mathematical validity of comparing them on the same Y:** yes, this is the only
valid comparison. Both algorithms are being asked to solve the same inverse-problem
instance; GD's disadvantage (a nonvanishing bias) is *itself* the finding Xu's Fig.3
is designed to expose.

**Verdict: Option A. DIRECTLY STATED IN PAPER** (via the "does not rely on an
approximation" sentence + the narrative order of Eq.8→10→11→12).

---

## PART 3 — Strong-reference condition: pointwise vs. average power

**What the Taylor expansion literally requires:** the pointwise condition (Part 1,
row 4) — |b(i,p)| ≫ |a(i,p)+n(i,p)|, for the *specific* (i,p) sample being expanded.
This is what makes √(1+x)≈1+x/2 accurate for that one sample's x=|a+n|²/|b|².

**What's appropriate for Monte Carlo simulation:** a pointwise inequality cannot be
verified as a single yes/no fact across 300 random trials — it fluctuates per draw.
The only way to characterize whether the approximation is *typically* good across
the simulation's own random distributions is the **average power ratio,
E|b|²/E|a+n|²** — and specifically **power** (squared modulus), not average of |b|
and |a+n| separately, because the dropped term itself, |a+n|²/(2|b|²), is *already*
a squared-magnitude ratio (Part 1, row 5). Checking E|b|²/E|a+n|² is checking the
expected size of exactly the term being discarded — not a looser proxy for it.

**On "RSR":** that term appears **nowhere** in Xu's paper. It is Cui's Eq.(37)
terminology, for a *different experiment* (Cui's signal-detection NMSE-vs-RSR sweep,
Fig.6/8 of Cui's paper). If a calibration parameter is needed to make Eq.(8)'s data
honor the strong-reference regime, it must be introduced and labeled explicitly as
**an implementation assumption invented for this simulation** — never as "the
paper's RSR."

**Labels:**
- Pointwise condition as the *mathematical requirement*: DIRECTLY STATED IN PAPER.
- Average-power-ratio as the *right way to check it in simulation*: MATHEMATICALLY
  IMPLIED (not stated, but forced by the structure of the dropped term and by
  Monte Carlo necessity).
- Any specific target dB value/calibration mechanism: IMPLEMENTATION ASSUMPTION,
  never "paper-defined."

---

## PART 4 — Channel/reference generation audit

| Item | Literal paper text | Verdict |
|---|---|---|
| Eq.(7) multipath sum | "Total channel = Path 1+Path 2+...+Path L_k" (structurally, Σₗ) | DIRECTLY STATED |
| α(l,k) distribution | Section V: "α(l,k)... follow 𝒞𝒩(0,1)" | DIRECTLY STATED |
| α_b distribution | Section V: "...and α_b follow 𝒞𝒩(0,1) and 𝒞𝒩(0,10), respectively" | DIRECTLY STATED — must be 𝒞𝒩(0,10) |
| Polarization ε(i,k,l), ε(b,i) | Section V, verbatim: "polarization direction is randomized, with ε(i,k,l), ε(b,i) following N(0,1/3)" | DIRECTLY STATED — antenna index **i is explicit** on both. Confirmed **not** a shared scalar. |
| θ(l,k) AOA distribution | Not given anywhere in Section V or Section II | NOT SPECIFIED — must be disclosed as an assumption |
| φ(l,k)=2πd/λ·cosθ formula | Section II, stated explicitly for Eq.(3)'s phase term, structurally identical to Eq.(7)'s exponent | DIRECTLY STATED (the *formula*) — numeric d/λ value: NOT SPECIFIED |
| φ_b formula | Not restated for the reference specifically, but same physical geometry | MATHEMATICALLY IMPLIED |

**On the suspicion that polarization "should be a single scalar shared across
antennas, only phase varies":** checked directly, again — this is **not**
supported. ε(i,k,l) carries the antenna index i in both the equation's own subscript
and in Section V's simulation-parameter sentence. A shared-scalar polarization is a
*different, undisclosed physical model* than what the paper specifies, however
physically intuitive it seems.

**Reference constant over pilots (s(b,p)=1):** Eq.(9) leaves s(b,p) general — Xu's
own text never sets it numerically. This is IMPLEMENTATION ASSUMPTION, justified
only by analogy to Cui's Eq.(35) treatment (b_n repeated identically across the
pilot dimension) — must be labeled as borrowed from Cui, not from Xu directly.

---

## PART 5 — ℏ, μ_eg audit

**Where it appears:** Eq.(7) and Eq.(9), as an identical multiplicative prefactor
(1/ℏ)μ_eg^T, applied to ε(i,k,l) in the channel and to ε(b,i) in the reference.
Section V gives it numerically: μ_eg=[0,1785.9qa₀,0]^T.

**Must it appear explicitly in code?** No — mathematical reason, not an assertion:
the prefactor (1/ℏ)μ_eg^T is **exactly the same complex scalar structure in both
Eq.(7) and Eq.(9)** — same μ_eg, same ℏ, no k/i/l/b-dependent subscript on either
factor. It therefore multiplies both g(i,k) and b(i,p) identically. Since it appears
squared (once from the true value, once from its own complex conjugate) in **every
quantity this simulation reports** — NMSE=‖G−Ĝ‖²/‖G‖² and the E|b|²/E|a+n|²
diagnostic alike — it cancels exactly in the numerator/denominator of both. Fig.3's
own axes (SNR in dB, NMSE in dB) are relative quantities by the paper's own
definition; no absolute-power benchmark is ever plotted that would expose the
missing scale.

**Can normalization "hide" this?** No normalization is even needed for the
cancellation argument — it's not a normalization trick, it's that the constant is
identical in both places it appears, so it drops out of any ratio algebraically,
before any code runs.

**Does omitting it change relative estimation performance?** No — by the above,
plus: multiplying G and B by any nonzero complex constant c leaves GS/GD/CRLB's
*behavior* unchanged under the natural SNR definition (σ²_complex is set relative to
E|A|², which also scales by |c|², so the noise-to-signal ratio at fixed dB is
preserved). This is a scale-invariance property of the whole pipeline, not specific
to ℏ,μ_eg.

**Verdict: MATHEMATICALLY IMPLIED that it can be omitted** (with the cancellation
reasoning stated explicitly, never asserted without proof).

---

## PART 6 — GS audit (the transpose question, resolved carefully)

**Q1: What is the measurement matrix, and S/S^T/S^H/conj(S)?**

Derive directly from Xu's own Eq.(8), not by importing Cui's Eq.(35) syntax blindly:

Per-antenna, Eq.(8)'s summation Σₖg(i,k)s(k,p) is literally the (1,p) entry of the
**row-vector–matrix product** g_i·S, where g_i=G(i,:) is 1×K and S is K×P. As a
column vector: (g_i·S)^T = **S^T**·g_i^T — this is a plain transpose identity
(reversing product order), requiring **no conjugation**, valid for any matrices
regardless of whether entries are complex.

So Xu's Eq.(8), rewritten per-antenna as a column-vector equation with x≡g_i^T (K×1,
the unknown, literally the transposed row — no conjugate), is:

y_i = |**S^T**x + b_i + n_i|

**This is S^T (plain), not S^H.** Cui's Eq.(35) prints S^H because *Cui's* channel
vector a_n is derived through a different chain (their Rabi-frequency formula,
Eq.14, uses μ_eg^H — Hermitian — while Xu's own channel definition, Eq.7, uses
μ_eg^T — plain). That's a genuine, confirmable difference in the two papers' own
notation (verified directly: Xu Eq.6's Ω_i uses μ_eg^H, but Xu Eq.7's g(i,k)
switches to μ_eg^T). Cui's a_n and Xu's g_i^T are not guaranteed to be the same
object up to conjugation without re-deriving Cui's whole chain — and doing so is
unnecessary, since Xu's Eq.(8) is self-contained and directly gives S^T.

**Verdict on `A_gs = S.'`: CORRECT — MATHEMATICALLY IMPLIED BY PAPER** (derived from
Xu's Eq.8 directly), not "directly stated" (Xu never writes out GS's matrix form at
all — it only says "GS [14] can be used"), and specifically **not** inherited from
Cui's Eq.(35) syntax, which uses a different underlying convention.

**Q2-9, remaining GS checks:**

| Check | Verdict |
|---|---|
| Unknown vector dimension | K×1 per antenna (channel vector for that antenna) — MATHEMATICALLY IMPLIED |
| Per-antenna independence | Yes — Eq.(8)'s relation for row i only involves row i of A, B, Y; no cross-antenna coupling — MATHEMATICALLY IMPLIED |
| Spectral init source | Cui Eq.(26)-(27), citing Candès-Li-Soltanolkotabi (Wirtinger Flow, Cui's own ref [35]) for the *general technique*; **Cui's own concrete formula weights by z_n (linear in magnitude)**, not z_n² | our code follows Cui's printed formula exactly (`diag(z_i)`, not `diag(z_i.^2)`) — CORRECT, matches Cui not Netrapalli |
| Netrapalli's own init (Alg.1, weight yᵢ²) | A *different* concrete spectral-init variant (squared weight) — this is the paper attached as Xu's ref [14] for GS itself, distinct from Cui's ref [35] for spectral init specifically | Since we implement Cui's Eq.(26) literally (linear weight), using Netrapalli's squared-weight variant instead would be a **different, unjustified substitution** |
| Projection/LS update | Cui Alg.1 steps 6-7, with A→S^T, b→b_i, s→x (re-derived convention) | matches, once S^T substitution (Q1) is applied consistently throughout — including inside `(A_gs'*A_gs)` for the LS solve, where `A_gs'` = (S^T)^H = conj(S) — this **is** correctly Hermitian, since it's a generic real least-squares normal-equation step, unrelated to the S^T-vs-S^H debate about the *observation* model |

**On Netrapalli-vs-Cui spectral init weighting (yᵢ² vs. zₙ):** a genuine,
previously-unflagged discrepancy between the two candidate reference algorithms.
Since Xu's paper gives no GS equations of its own, and Cui's Eq.(26) is the more
directly-relevant source (Cui's paper explicitly derives the *channel-estimation*
extension, Eq.35, that Netrapalli's paper never addresses — Netrapalli is pure
signal-vector phase retrieval, no reference/bias term at all), **Cui's linear-weight
formula is the correct one to use** — label this IMPLEMENTATION ASSUMPTION (a
justified choice between two candidate sources, not something either paper states
as the unique answer for *this* problem).

---

## PART 7 — GD audit

Fully consistent with Xu's literal Eq.(12)-(15) — Part 1, rows 7-10. Dimension
check, explicit:

- G: I×K, S: K×P → G·S: I×P
- Y, B: I×P (real/complex resp.), gradient: I×K (must match G)
- residual = Y−|B|−Re{(GS)∘Z}: I×P **real**
- residual∘Z*: I×P (elementwise, Z* complex conjugate, `conj(Z)`)
- (residual∘Z*)·S^H: (I×P)(P×K) = I×K ✓ matches G

**S^H here is correct and undisputed** — Xu prints S^H explicitly in Eq.(15) (Part 1,
row 10), unlike GS where Xu prints nothing at all and the transpose has to be
derived (Part 6). MATLAB: `S'` (Hermitian), confirmed correct, do not change to
`S.'`.

Step size / init / stopping: not specified numerically by the paper beyond init
variance 0.1 (Section V) — backtracking line search and 1e-8 tolerance are
IMPLEMENTATION ASSUMPTIONS, already disclosed as such.

---

## PART 8 — CRLB audit

Derived directly from Eq.(28)-(31)'s literal text (the σ² notation-collision
argument):

1. Observation model assumed: the **linearized** Eq.(12)/(20) model, vectorized as
   Eq.(28) — not the exact nonlinear Eq.(8).
2. Noise variance: Eq.(28)'s printed "σ²" is **not** Eq.(8)'s raw complex noise
   variance — it must equal Eq.(11)'s already-halved real-noise variance
   (σ²_complex/2), because Eq.(28) states n=vec(N)~N(0,σ²I) where N *is*
   Eq.(11)/(12)'s N, whose true variance was already fixed at σ²_complex/2 by
   Eq.(11)'s own text.
3. CRLB is a **total** bound (trace of the inverse FIM, summed over all I·K complex
   unknowns via the Kronecker structure), directly comparable to Frobenius-norm
   NMSE — Eq.(31)'s I_{I1I2}⊗(S*S^T)^{-1} structure, traced and summed over I
   antennas, gives exactly the scalar compared against Σ‖G−Ĝ‖²_F.
4. Scaling with P: implicit through (S*S^T)^{-1} — more pilot columns in S
   generally shrinks this inverse, matching the paper's own stated conclusion
   ("CRLB... reduces as the number of pilots grows").
5. MATLAB: `CRB_one = 4*(sigma2_complex/2)*pinv(conj(S)*S.'); total =
   I*real(trace(CRB_one))`.

No new issue found here — matches what's already implemented and disclosed.

---

## PART 9 — Diagnosing a reported GD≫GS, GD≈CRLB result

A result set was reported with GD landing within ~0.05dB of CRLB while GS sat flat
at +1 to +4dB regardless of SNR (e.g., P=30,SNR=30: GD≈−34.03dB vs CRLB≈−33.98dB).
This is not plausible under a faithful Option-A implementation, for a specific,
checkable reason:

**GD landing within 0.05dB of CRLB is the signature of GD being tested against its
own linearized-model-generated observation** (the Eq.11-built "Y_GD" bug found and
fixed earlier this session), **not** genuine convergence. CRLB is itself derived
entirely from the linearized model (Part 8, point 1) — if GD is handed data
manufactured to satisfy that same linearized model exactly, of course it can
approach the bound that model implies, because there's no model-mismatch error left
to bound. A GD that's honestly fed Eq.(8)'s true nonlinear magnitude cannot get
*that* close to a bound derived from an approximation it doesn't actually satisfy
exactly — the residual bias (Part 1, row 5's dropped term) sets a floor CRLB doesn't
know about.

Simultaneously, **GS pinned at a flat +1 to +4dB regardless of SNR** is consistent
with GS operating correctly on the *true* Eq.(8) data but hitting a genuine floor
from something else entirely — most plausibly the non-convex local-minima issue
(worst-case trials dominate the sum-of-squares NMSE) or a reference-dominance
shortfall (Part 3) making even GS's exact-nonlinear-model fit noisy.

**Recommended check, not a fix:** confirm in whatever script produced these numbers
whether `Y_GD` is built as `abs(A_signal+B+N)` (same variable as GS's Y) or
separately from `real(Z.*A_signal)+abs(B)+...`. If it's the latter, that single line
explains the entire pattern — not a GS bug, not a "GD genuinely wins" result.

A careful Option-A implementation (same Y, S^T for GS per Part 6, literal
per-antenna polarization), already run this session as `fig3_literal.m`, instead
shows GS beating GD at both P values, with genuine floors on both sides tied to the
measured E|b|²/E|a+n|² shortfall (Part 3) — not either qualitative pattern reported
in this Part.

---

## PART 10 — Structured summary

**A. Equations/dimensions:** Part 1 table — complete, no changes needed to what's
already coded for channel/reference/GD/CRLB (all independently re-derived from
literal paper text this session).

**B. GS observation:** Eq.(8), the true nonlinear |GS+B+N| — DIRECTLY IMPLIED
(Part 2).

**C. GD observation:** the *same* Eq.(8) data; GD fits it via the Eq.(12)/(13)
approximation but never receives separately-manufactured "clean" data — DIRECTLY
IMPLIED (Part 2).

**D. Same raw measurement for both?** Yes, required (Part 2).

**E. Channel generation:** Eq.(7), per-antenna-**and**-per-path polarization
ε(i,k,l)~N(0,1/3), α(l,k)~𝒞𝒩(0,1), Lₖ~U(3,7) — DIRECTLY STATED; AOA distribution and
d/λ value — NOT SPECIFIED, must stay disclosed assumptions (Part 4).

**F. Reference generation:** Eq.(9), per-antenna ε(b,i)~N(0,1/3), α_b~𝒞𝒩(0,10)
(single scalar, no per-antenna index on α_b itself — only ε(b,i) carries i) —
DIRECTLY STATED. s(b,p)=1 constant — IMPLEMENTATION ASSUMPTION borrowed from Cui.

**G. GS implementation:** A_gs=S^T (not S^H), x=g_i^T (no conjugate), re-derived
directly from Xu's Eq.(8) — MATHEMATICALLY IMPLIED, and explicitly **not** a blind
copy of Cui's Eq.(35) syntax, which uses a different (μ_eg^H-derived) convention.
Spectral init follows Cui's Eq.(26) linear-in-z_n weighting specifically, not
Netrapalli's squared-weight variant — IMPLEMENTATION ASSUMPTION (justified choice
between two candidate sources).

**H. GD implementation:** Eq.(13)-(15) literal, S^H confirmed correct (paper prints
it) — no dispute.

**I. CRLB:** Eq.(28)-(31), σ² notation-collision resolved (one halving from raw
complex noise, matching Eq.28's own redefinition) — no dispute, already correct.

**J. Bugs/inconsistencies found (this session, cumulative):**
1. Per-antenna polarization "fixed" to a shared scalar — was wrong, reversed.
2. α_b~𝒞𝒩(0,1) instead of 𝒞𝒩(0,10) — fixed.
3. Invented RSR=30dB rescaling not present in Xu's paper at all — removed.
4. Channel-power normalization not in the paper — removed.
5. An earlier version's GD fitted a separately-manufactured linearized "Y_GD"
   instead of the true Eq.(8) magnitude — fixed; this is the most likely
   explanation for Part 9's reported numbers if it has crept back into a newer
   script.
6. ℏ,μ_eg omission — mathematically inert (Part 5) but was previously
   under-disclosed as "everything literal."
7. GS's transpose (S.' vs Cui's printed S^H) — confirmed S.' is correct via direct
   re-derivation from Xu's own Eq.(8), not from Cui's Eq.(35) syntax.

No code changes are proposed in this document. Next step, if desired: apply this
audit's conclusions to a fresh implementation pass, with equation-number and
dimension comments on every block, only after this file has been reviewed.
