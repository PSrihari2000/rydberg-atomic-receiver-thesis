# Project Memory: Rydberg Atomic-Receiver Thesis

Last consolidated: 2026-10-10 (Asia/Kolkata)

This file carries forward project context from the local Claude Code project memory and available Claude Code session transcripts, together with Codex's prior thesis context. Treat it as a working map, not a substitute for checking the current source files, paper PDFs, MATLAB outputs, or Git state. Historical results and file status below can become stale.

## User and collaboration preferences

- The user is an M.Tech student in Signal Processing and Communications at IIT Bhubaneswar, researching Rydberg atomic receivers and channel estimation.
- Be rigorous and willing to challenge assumptions. Separate paper-stated facts, derived results, implementation choices, and unresolved questions.
- Explain math and code in small, explicit steps. The user often wants a high-school-level derivation with each algebraic property and notation change shown, especially when preparing to explain it to the advisor.
- Lead with the direct answer. Use short sentences, concrete examples and small tables where useful. Give one recommendation when asked which option to choose.
- In chat, avoid raw LaTeX source. Use readable Unicode notation; define symbols in bullets. In `.tex` files, use normal LaTeX.
- For the Xu channel-estimation thread, provide MATLAB code. The user runs MATLAB themselves. Keep scripts readable and step-by-step, with descriptive names and explicit loops; avoid compressed/vectorized code and unnecessary helper functions in reproduction scripts.
- For targeted slide or code requests, return the requested block/snippet without rewriting unrelated material.
- Guide-facing reports, derivations, slides and code comments must not mention Claude, ChatGPT, AI, or other tools. Use a professional solo-author voice; avoid “we/our” in submitted prose.
- Figures must come from genuine simulation outputs. Disclose assumptions and tuned settings; do not add arbitrary per-curve knobs to improve visual resemblance. Keep paper reproduction figures limited to the equations and curves actually in the paper.
- Before editing, inspect the current files and `git status`. Preserve unrelated user work. Do not commit unless the user asks.

## Project tracks

### Active: Xu et al. channel estimation

Paper: Bokai Xu et al., “Channel Estimation for Rydberg Atomic Receivers,” IEEE Wireless Communications Letters (2025), associated with arXiv:2503.08985.

- Main workspace: `D:\MTech_Thesis_Results`.
- Active code/document area: `Channel_Estimation_for_Rydberg_Atomic_Receivers/`.
- Git branch seen in the workspace: `fresh-figure-rebuild`; remote has been `origin` at `PSrihari2000/rydberg-atomic-receiver-thesis`.
- The 1D/2D derivation is `Channel_Estimation_for_Rydberg_Atomic_Receivers/Derivation/ce_derivation_1D_2D_PGD_CRLB.tex`. Claude's Oct. 8 memory says the 82-page document compiled successfully on Overleaf, ending at equation (466); the expanded gradient-step derivation was later rewritten in a step-by-step, exam-style format. Re-open the current `.tex` and compile/check the current version before relying on those details.
- The Oct. 8 project note said commit `4bc1e67` had been pushed and described `Simulation_Report/simulation_reference.tex` plus `rsr_check/` and the step-size derivation. That is historical status, not confirmation of the current checkout.
- As checked on 2026-10-10, the worktree had many deletions and modifications, plus untracked files. Do not restore, stage, or commit those changes without inspecting the current diff and user intent. In particular, old report, reference-PDF, MATLAB and Chen/v1 paths appeared deleted in that snapshot.

#### Reproduction files and reported settings (verify live scripts)

- Figs. 3–5 have literal and tuned MATLAB variants, with a separate converged Fig. 3 check. The compact naming convention is `figX/figX_literal/figX_literal.m` and `figX/figX_tuned/figX_tuned.m`; each script's `.png` and `.mat` output uses the same base name.
- **Literal** means paper-stated parameters and equations, with unstated assumptions identified. **Tuned** changes selected settings only; it does not change the paper's equations. A separate converged Fig. 3 run was kept to distinguish accuracy at convergence from a finite iteration budget.
- Claude's latest consolidated notes record these selected settings: RSR 50 dB, polarization per path, SNR per cell following Cui et al. Eq. (36), and a common RSR across the figures. Fig. 3 tuned uses GS iteration counts `[10 50]` and GD `[30 50]` for P = 10/30; Fig. 4 tuned uses 50 GD/PGD iterations for both P; Fig. 5 tuned PGD runs to convergence. These settings and outputs were selected for defensibility and qualitative reproduction; check the actual scripts before quoting or changing them.
- `G0 ~ CN(0, 0.1)` is stated in Xu et al. Sec. V. For a circular complex Gaussian with variance 0.1, each real component has variance 0.05; MATLAB initialization uses `sqrt(0.1/2)*(randn + 1j*randn)`.
- Xu et al. do not specify all implementation details. Notes identify `d/λ = 0.5`, 2D angle distributions, and several simulation conventions as assumptions, not paper-given facts. The current explanation for the 2D convention is elevation measured from an array axis over `(0, π)` and azimuth over `(0, 2π)`; match the angle formula and verify the source before asserting it.
- The tuned reference scaling used in the code is of the form `B = B*sqrt(RSR_linear*signalPower/currentReferencePower)`. The square root is over the entire ratio because `B` is an amplitude and power scales by amplitude squared. Confirm the target ratio is reference power divided by signal power before changing this code.
- The reported GD step size is `1/max(eig(S*S'))` for the corresponding 1D formulation, derived from the Lipschitz bound. Xu's 1D equation and gradient conventions differ by a factor of 2 depending on the gradient definition; check the exact objective and code convention before transferring the step size.
- `Y_centered = Y - abs(B)` is an optional cached subtraction. It is equivalent to writing `Y - abs(B)` directly in every residual. Omitting `abs(B)` is not equivalent.
- In the 2D mode-3 unfolding, `q = i1 + (i2 - 1)*I1` maps cell `(i1,i2)` to MATLAB-compatible column-major index `q`.

#### Findings and caveats recorded in Claude memory

- Literal simulations had weak reference-to-signal power (around −1.4 to −2.1 dB in recorded runs), which conflicts with the strong-reference condition used for the linearization and can cause an NMSE floor. The tuned RSR was raised to 50 dB; disclose that as an implementation choice, not as Xu's stated value.
- Per-path polarization was selected for a single plane wave/path. The notes say per-cell polarization in the 2D channel can destroy the intended low-rank structure; verify dimensions and assumptions before applying this claim to new models.
- The converged Fig. 3 check reportedly made GS and GD agree within Monte Carlo variation, while using fewer GS iterations. The finite-budget tuned plot was kept to reproduce the paper's ordering; explain this honestly.
- In the recorded Fig. 4 results, PGD was modestly better than GD. Fig. 5 was treated as a converged accuracy-versus-pilot-count experiment. Recheck current `.mat` outputs before quoting numeric values.
- The CRLB is a benchmark, not an estimator. Recorded notes flag possible looseness in the paper's CRLB for the real-valued linearized model, inconsistencies among published curves, and an SNR definition that Xu does not specify. Keep these as caveats until independently re-derived from the current model.
- No single SNR convention in the notes explained all figure-level offsets. The selected working SNR convention is per-cell, based on Cui et al.; label it as a chosen convention where Xu is silent.

#### Longer-term method comparison

- Advisor-directed future work: compare CNN/deep-learning channel estimation, QML, and Hankel-structured methods with GS/GD/PGD baselines; assess NMSE and computational effort (iterations/FLOPs); then evaluate data transmission using SER/BER versus SNR.
- Freeze one simulator and parameter set before comparing methods. Baselines should run to convergence where the comparison is intended to measure achievable accuracy; report finite-budget comparisons separately.
- Hankel-structured estimation is model-based, not machine learning. The notes warn its benefit may be small for an 8-element array and it can add runtime. Reassess against the exact array/path regime before proposing it as a lower-computation method.

### Earlier/parallel: Chen et al. “Harnessing Rydberg Atomic Receivers”

- This is a distinct receiver model and codebase from Xu channel estimation. Historical work used Python/QuTiP for the quantum-optics figures; do not carry those tools into the active Xu MATLAB thread by default.
- Claude's Aug. 2026 memory says the Chen/v1 Fig. 3–5 and Fig. 9 work had scripts and a LaTeX derivation, while Figs. 6–8 remained unfinished at that time. It records a model distinction between LO-dressed atomic mixing and the magnitude-only biased phase-retrieval receiver studied by Xu/Cui. Treat the reported state as historical; inspect `Harnessing Rydberg Atomic Receivers-From Quantum physics to communication/` before resuming.
- Reported Fig. 9 work required an explicitly disclosed SNR-to-RF-field calibration bridge because the paper's listed parameters did not reproduce its transition points. Do not silently reuse or retune those assumptions.

## Documents and advisor work

- The 2026 mid-review report was recorded as finalized at 17 pages on 2026-09-23. The Beamer deck was maintained in Overleaf; the local copy might not match it.
- Report/deck scope was Xu channel estimation plus Chen et al. v1, with user-generated simulation plots. Future work was set as CNN, QML and Hankel channel estimation, then SER/BER evaluation.
- The user often needs a one-sentence, technically honest explanation to give the advisor. Explain where every parameter came from and explicitly flag choices the paper does not state.
- The derivation and `Simulation_Report/simulation_reference.tex` have different status in Claude's Oct. 8 notes: the derivation had been compiled by the user on Overleaf; the simulation reference report had not yet been compiled. Check present files and Overleaf state.

## Additional context from other saved sessions

- Earlier sessions covered array geometry and angle conventions, code walkthroughs, resume editing, LinkedIn questions, phone-to-laptop control, and a separate Chen/v1 simulation. These are not automatically part of the active thesis task; use their local transcripts only when relevant.
- The local Claude Code transcripts were found under `C:\Users\P SRIHARI\.claude\projects\` (18 JSONL files across repo, OneDrive and scratch-project folders). Claude project memory for this repository was under `C:\Users\P SRIHARI\.claude\projects\d--MTech-Thesis-Results\memory\`.

## Resume checklist

1. Run `git status --short --branch` and inspect relevant diffs before editing.
2. Confirm which thread/paper the task concerns (Xu MATLAB or Chen/QuTiP).
3. Check the actual paper page/equation and live script; memory is a pointer, not evidence.
4. Keep assumptions, settings and simulation outputs traceable and honestly labeled.
5. Make only the requested change, preserve user files, and do not commit unless asked.

## DL channel-estimation track: verified status 2026-10-26

- Active scripts/notebooks and reports are under `v2/`. Read each current file before relying on older run descriptions.
- V12 pilot code: `v2/colab_unfolded_dl_v12_cnn_pilot.py`; Colab launcher: `v2/DL_v12_cnn_pilot_colab.ipynb`; experiment notes: `v2/V12_README.md`.
- V12 completed on a Colab T4, using one seed (20261026), for P=10 and P=30, with a loss-only arm and a CNN residual arm. It trained/selected on train and validation only; test was not loaded/evaluated. Downloaded artifacts are in `v2/v12_results_bundle/` (four selected `.pt` checkpoints, four per-epoch JSON histories, and `summary_v12_validation_pilot.json`).
- Verified V12 selected validation DL-minus-GS gaps (dB):
  - P=10 loss, best epoch 8, worst gap +0.61842; per SNR −5/0/5/10/15/20/25/30: −9.4724, −5.6774, −3.3837, −1.4399, −0.7139, +0.0248, +0.6184, +0.6178.
  - P=10 CNN, best epoch 1, worst gap +0.62223; per SNR: −9.4784, −5.6611, −3.3581, −1.4220, −0.7208, +0.00025, +0.5887, +0.6222.
  - P=30 loss, best epoch 6, worst gap −0.000422; per SNR: −2.8452, −1.0945, −0.2940, −0.06692, −0.00759, −0.00603, −0.000422, −0.000426.
  - P=30 CNN selected epoch 0 (untrained CNN residual is zero); all later trained checkpoints had worse worst-case validation gap. Its selected curve is the source model's initial curve: −2.8389, −1.0846, −0.2871, −0.06419, −0.00691, −0.00609, −0.000412, −0.000559.
- Interpretation: V12 did not close P=10's high-SNR gap (about +0.59 to +0.62 dB at 25–30 dB). P=30 loss arm is essentially tied with GS at high SNR; sub-millidecibel gains are not meaningful evidence of superiority. Results are one-seed validation-only; do not claim test/generalization wins.
- Next recommended action: diagnose P=10 train-versus-validation NMSE by SNR and pilot-condition group using selected V11/V12 models, without touching test data. If train is better than GS but validation is worse, investigate data coverage/generalization; if both lag, investigate estimator/optimization limitations before another large training run. Then test one targeted change and validate with multiple seeds; reserve test for a frozen final comparison.
- V4–V11 DL artifacts have now been gathered for GitHub publication in the active V12 sync worktree: scripts, Colab notebooks, selected and resume checkpoints, summaries, histories/reviews, and comparison/audit CSVs (97 files, about 9.5 MB). This includes historical test-result artifacts from V6–V9; label each result by its run/version and do not treat it as a final independent comparison without checking its evaluation protocol. Publication is pending GitHub authentication.
