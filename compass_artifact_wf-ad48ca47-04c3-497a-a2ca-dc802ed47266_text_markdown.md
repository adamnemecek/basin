# Hybrid/Memetic CMA-ES with Gradient or Quasi-Newton Local Refinement: An Annotated Bibliography

## TL;DR
- The canonical reference for "CMA-ES outside, local refiner inside" is the **MA-LS-Chains** family of Molina, Lozano, García-Martínez & Herrera (HM 2008; *Evolutionary Computation* 2010) — implemented in the R package `Rmalschains` — together with **Hansen's 2011 INRIA report "Injecting External Solutions Into CMA-ES"** (arXiv:1110.4181), which is the foundational mechanism for Lamarckian re-injection of locally refined solutions back into a CMA-ES distribution.
- Strict published pipelines that chain CMA-ES specifically with **Levenberg–Marquardt** are surprisingly sparse — most works compare the two as competing optimizers (Bledsoe et al. 2011; Grbić et al. 2016; Bhattacharjee et al. 2020). The cleanest chained schemes use **NEWUOA** (HCMA: Loshchilov, Schoenauer & Sebag, GECCO BBOB 2013), **L-BFGS-B / BOBYQA / Simplex** (Atia et al. 2017), or **coordinate-wise Mtsls1** (Liao & Stützle's iCMAESILS, CEC 2013).
- For software, **pagmo2/pygmo** is the most idiomatic out-of-the-box choice via the `mbh` (Monotonic Basin Hopping, generalised) meta-algorithm wrapping the `cmaes` UDA with NLopt's L-BFGS / BOBYQA / SLSQP wrappers; **pycma** exposes the `es.inject()` hook and SciPy-compatible result objects, so a `cma.fmin2` → `scipy.optimize.minimize(method='L-BFGS-B')` two-stage script is a few lines of code.

## Key Findings

1. **The dominant "memetic CMA-ES" architecture in the literature is inverted relative to the user's framing.** Molina–Lozano-style memetic algorithms use a steady-state GA at the outer level and **CMA-ES as the local search operator**, exploiting CMA-ES's strong quadratic-model behaviour near optima. The user's intended pattern — CMA-ES outer, gradient/quasi-Newton inner — is supported in the literature mainly through Hansen's injection protocol (A4) and the HCMA/iCMAESILS/Atia line of work (B1–B4).

2. **When CMA-ES IS used as the outer global optimizer**, the most influential hybrids are **HCMA** (NEWUOA initial probe → STEP → BIPOP-saACM-ES-k) and **iCMAESILS** (IPOP-CMA-ES competing against an Mtsls1 iterated local search). Notably, neither uses a true gradient-based or quasi-Newton inner step; the community has preferred derivative-free trust-region (NEWUOA) or coordinate-wise (STEP, Mtsls1) polish.

3. **Hansen's "Injection" paper (arXiv:1110.4181) is the foundational primitive.** Any externally refined solution — explicitly including "a gradient or a Newton step, a surrogate model optimizer or any other oracle or search mechanism" — can be substituted for one of the λ sampled candidates each generation, with a tight step-length renormalization that prevents the covariance update from being destabilised. This is the right building block for a custom CMA-ES → L-BFGS-B chain.

4. **For nonlinear least-squares applications**, explicit CMA-ES → Levenberg–Marquardt chained pipelines are rare in the peer-reviewed literature. The closest documented chained scheme is Atia et al. 2017 (CMA-ES → L-BFGS-B 2-stage memetic framework). Head-to-head comparisons (Bledsoe et al. 2011, *Ann. Nucl. Energy*; Grbić et al. 2016, *IFAC-PapersOnLine*) establish the empirical rationale: CMA-ES wins as dimension and multimodality grow; LM wins on smooth low-dimensional residual problems — which is exactly the complementarity that motivates chaining them.

5. **Practical recommendation:** Run BIPOP-CMA-ES (or IPOP-aCMA-ES) until stagnation, then warm-start either L-BFGS-B (`scipy.optimize.minimize`) or, if the objective is a sum-of-squared-residuals, Levenberg–Marquardt (`scipy.optimize.least_squares(method='lm')`). This is the simplest scheme, is consistent with Hansen's own published advice, and replicates the structure of Atia et al.'s 2-stage framework.

## Details

### A. Foundational memetic-CMA-ES papers

**A1. Molina, Lozano, García-Martínez & Herrera (2008).** "Memetic Algorithm for Intense Local Search Methods Using Local Search Chains." In Blesa et al. (eds.) *Hybrid Metaheuristics (HM 2008)*, LNCS 5296, Springer, pp. 58–71. DOI: 10.1007/978-3-540-88439-2_5. — Introduces **MA-LSCh-CMA**: a steady-state GA with BLX-α crossover at the outer level, with **CMA-ES as the intense local search operator**. Innovation: "local search chains" — when CMA-ES is re-invoked on a previously improved chromosome, it resumes from the stored final CMA-ES strategy state (σ, C, evolution paths) rather than restarting. Lamarckian; population 60; mutation probability 0.125; LS intensity I_str = 500 evaluations per chain segment; balance ratio r_L/G = 0.5.

**A2. Molina, Lozano, Sánchez & Herrera (2010).** "Memetic algorithms for continuous optimisation based on local search chains." *Evolutionary Computation* 18(1): 27–63. DOI: 10.1162/evco.2010.18.1.18102. — Journal expansion of A1 with a thorough CEC 2005 benchmark study, full pseudo-code for the chain mechanism, and the algorithm directly implemented in the R package **Rmalschains** on CRAN.

**A3. Auger & Hansen (2005).** "Performance evaluation of an advanced local search evolutionary algorithm." *IEEE CEC 2005*, pp. 1777–1784. — Establishes that even pure CMA-ES with restarts (IPOP-CMA-ES) acts as a strong global–local hybrid in its own right: each restart with increasing population size functions as a global re-exploration after a local-refinement phase. Conceptual ancestor of BIPOP-CMA-ES.

**A4. Hansen (2011).** "Injecting External Solutions Into CMA-ES." INRIA Research Report RR-7748; arXiv:1110.4181. https://arxiv.org/abs/1110.4181. — Defines the protocol for substituting an externally obtained solution (gradient/Newton step, surrogate optimum, repaired infeasible point, or L-BFGS-B output) for one of the λ candidates. Key contribution: tight renormalization of step length. Explicitly enables Lamarckian memetic CMA-ES: *"an improved solution, for example the result of a local search started from a solution sampled by CMA-ES (Lamarckian learning), which allows to use CMA-ES in the context of memetic algorithms."* Must-read for any user building a custom CMA-ES → LM/BFGS chain.

### B. Modern hybrids with CMA-ES as outer global

**B1. Loshchilov, Schoenauer & Sebag (2013).** "BI-population CMA-ES Algorithms with Surrogate Models and Line Searches." In *GECCO '13 Companion*, ACM, pp. 1177–1184. DOI: 10.1145/2464576.2482696. HAL: hal-00818596. PDF: https://numbbo.github.io/gforge/data/media/pdf2013/w0306-loshchilov.pdf. — Defines **HCMA = BIPOP-saACM-ES-k + STEP + NEWUOA**. Per the authors' GECCO 2013 slides (slide 21, verbatim): *"1. NEWUOA with m = 2n+1 for 10n function evaluations. 2. BIPOP-saACM-ES-k and STEP with n_MinIterSTEP = 10 (e.g., 10n evaluations)."* The inner derivative-free refiner is **NEWUOA** (Powell's trust-region quadratic-model method); it runs as an initial 10·n-evaluation probe, after which the BIPOP-aCMA-ES surrogate-assisted main loop takes over. HCMA was reported as the best overall performer on BBOB-2009/2010/2012/2013 at budgets ≥ 100·n.

**B2. Liao & Stützle (2013).** "Benchmark results for a simple hybrid algorithm on the CEC 2013 benchmark set for real-parameter optimization." *IEEE CEC 2013*, Cancún, pp. 1938–1944. DOI: 10.1109/CEC.2013.6557796. — The **iCMAESILS** algorithm. Loose coupling of IPOP-CMA-ES with an iterated local search built on **Mtsls1** (Tseng's Multi-Trajectory Local Search 1, coordinate-wise derivative-free). Initial competition phase, then the winner runs to exhaustion, with one-way solution exchange from IPOP-CMA-ES to ILS. **Ranking note:** secondary sources disagree about whether iCMAESILS placed 1st or 2nd at the CEC 2013 Real-Parameter Single Objective competition (a recent longitudinal study by Novák et al., arXiv:2603.24140, lists ICMAES-ILS as the winner with NBIPOP-aCMA as runner-up; some other reviews give the reverse). Either way, iCMAESILS shared the top tier. Note: inner local search is *not* gradient-based despite "ILS" in the name.

**B3. Lou, Yuen, Chen & Zhang (2019).** "On-line Search History-assisted Restart Strategy for Covariance Matrix Adaptation Evolution Strategy." arXiv:1903.09085. — HR-CMA-ES uses the continuous non-revisiting GA (cNrGA) as the outer global explorer feeding "Regions of Interest" to CMA-ES as the inner local exploiter. Inverts the user's pattern but is an explicit memetic CMA-ES scheme of independent interest.

**B4. Atia, Picheny et al. (2017).** "A CMA-ES-based 2-Stage Memetic Framework for Solving Constrained Optimization Problems." (University of Trento repository.) — The cleanest published **CMA-ES → L-BFGS-B** chain. A single CMA-ES run delivers x*; that point warm-starts L-BFGS-B (also tested with BOBYQA and Nelder-Mead Simplex). Quote: *"We build upon it a 2-stage memetic framework, coupling the CMA-ES scheme with a local optimizer, so that the best solution found by CMA-ES is used as starting point for the local search."* Evaluated on CEC constrained benchmark suites. Lamarckian; budget allocation handled by giving the local optimizer its own termination tolerances rather than a fixed budget.

**B5. Loshchilov (2017).** "LM-CMA: An alternative to L-BFGS for large-scale black box optimization." *Evolutionary Computation* 25(1): 143–171; arXiv:1404.5520. — *Not a hybrid* but worth flagging: explicitly "inspired by the limited memory BFGS method of Liu and Nocedal (1989)" and represents convergence of CMA-ES and L-BFGS thinking inside a single algorithm. Useful when choosing between chaining and switching to LM-CMA-ES at high n.

**B6. Varelas (2019).** "Benchmarking Large Scale Variants of CMA-ES and L-BFGS-B on the bbob-largescale Testbed." *GECCO 2019 Companion*. DOI: 10.1145/3319619.3326893; HAL: hal-02160106. — Single-authored benchmark comparing separable-CMA-ES, VD-CMA, VkD-CMA, two LM-CMA-ES implementations, RmES, and L-BFGS-B on bbob-largescale (dimensions 20–640; CMA-ES baseline up to n = 320). Indispensable reference for understanding where pure L-BFGS-B beats every CMA-ES variant (unimodal, well-conditioned, smooth) and where it collapses (multimodal, non-smooth, noisy).

### C. Niching / multi-local CMA-ES (outer scheme across basins, CMA-ES per basin)

**C1. Preuss (2010).** "Niching the CMA-ES via nearest-better clustering." *GECCO 2010 Companion*. — Outer scheme: nearest-better-clustering meta-algorithm; inner local search: CMA-ES on each identified basin. Established as **NEA2** (Niching Evolutionary Algorithm 2).

**C2. Faliszewski, Sawicki, Łoś, Smołka & Schaefer (2019).** "Approximation of the objective insensitivity regions using Hierarchic Memetic Strategy coupled with Covariance Matrix Adaptation Evolutionary Strategy." arXiv:1905.07288. — **HMS-CMA-ES**: hierarchic outer scheme dispatches CMA-ES instances to multiple branches, each performing a local refinement, with insensitivity-region approximation as the application target.

### D. Application-domain papers (comparative or chained)

**D1. Bledsoe, Favorite & Aldemir (2011).** "A comparison of the Covariance Matrix Adaptation Evolution Strategy and the Levenberg–Marquardt method for solving multidimensional inverse transport problems." *Annals of Nuclear Energy* 38(4): 897–904. DOI: 10.1016/j.anucene.2010.11.001. — Head-to-head comparison on neutron/photon inverse transport. Verbatim conclusion: *"Numerical results indicate that the Levenberg–Marquardt method is more adept at problems with few unknowns (i.e., ≤ 3), but as the number of unknowns increases, CMA-ES becomes the superior strategy."* This is the empirical foundation for the "CMA-ES finds the basin, LM polishes" intuition.

**D2. Grbić et al. (2016).** "A Study on Performance of Levenberg-Marquardt and CMA-ES Optimization Methods for Atlas-based 2D/3D Reconstruction." *IFAC-PapersOnLine* 49(25): 372–377. DOI: 10.1016/j.ifacol.2016.12.075. — Medical-imaging application: fitting a deformable bone atlas to 2D X-rays. LM is several times faster than CMA/CMSA-ES at comparable accuracy (median error 1.12 mm, 7.2 s reconstruction time) — a result that motivates initialising with CMA-ES restarts and finishing with LM.

**D3. Jędrzejewski-Szmek et al. (2018).** "Parameter Optimization Using Covariance Matrix Adaptation—Evolutionary Strategy (CMA-ES), an Approach to Investigate Differences in Channel Properties Between Neuron Subtypes." *Frontiers in Neuroinformatics* 12: 47. DOI: 10.3389/fninf.2018.00047. — Biophysical neuron-model calibration on a discontinuous high-dimensional spike-feature fitness; converges in 1,600–4,000 model evaluations. Pure CMA-ES (not chained), but representative of the class of model-calibration problems where users routinely follow with an LM polish on the residual.

**D4. Tomasoni et al. (2022).** "Integration of Heterogeneous Biological Data in Multiscale Mechanistic Model Calibration: Application to Lung Adenocarcinoma." *Acta Biotheoretica* 71(1): 4. DOI: 10.1007/s10441-022-09445-3. — Staged CMA-ES calibration of a multiscale lung-cancer pathophysiology model; CMA-ES is iteratively applied to parameter subsets in a cascade, an interesting variant of the chained scheme where each stage's optimum seeds the next.

**D5. Bhattacharjee, Ranjan, Mandal & Tollner (2020).** "Efficient Calibration of a Conceptual Hydrological Model Based on the Enhanced Gauss–Levenberg–Marquardt Procedure." — Hydrological model calibration; CMA-ES and an enhanced Gauss–Levenberg–Marquardt are compared on the same SWAT-style rainfall–runoff target and reach comparable Nash–Sutcliffe efficiency, again motivating a chained pipeline.

### E. Software packages

**E1. pagmo2 / pygmo (ESA).** https://esa.github.io/pagmo2/. The **`mbh` (Monotonic Basin Hopping, generalised) meta-algorithm** wraps any UDA — including the `pygmo.cmaes` UDA — coupled with any local optimizer UDA (NLopt L-BFGS, BOBYQA, SLSQP; SciPy local optimizers). Quote from the pagmo2 docs: *"we provide an original generalization of this concept resulting in a meta-algorithm that operates on any pagmo::population using any suitable user-defined algorithm (UDA)."* The original PaGMO MBH design is documented in Izzo et al., "PyGMO and PyKEP: Open Source Tools for Massively Parallel Optimization in Astrodynamics" (ICATT 2012). MBH + SQP was historically the workhorse for ESA interplanetary trajectory optimization (Yam, Di Lorenzo & Izzo 2011, *Proc. IMechE Part G*); replacing SQP with CMA-ES or wrapping CMA-ES inside MBH is a straightforward configuration.

**E2. pycma (Hansen's Python CMA-ES).** https://github.com/CMA-ES/pycma. No built-in `local_refiner=` keyword; the chaining pattern is `x_best, es = cma.fmin2(f, x0, sigma0)` then `scipy.optimize.minimize(f, x0=x_best, method='L-BFGS-B')` — or `scipy.optimize.least_squares(method='lm')` if the residual vector is available. Lamarckian feedback into a running CMA-ES uses `es.inject([x_refined])` per Hansen 2011 (arXiv:1110.4181). pycma's `cma.fmin` option dictionary follows `scipy.optimize` conventions (`maxiter`, `maxfun`, `ftol`, etc.); GitHub issue #12 (opened by Hansen, 2017) is the roadmap item for returning a fully `scipy.optimize.OptimizeResult`-compatible object.

**E3. Rmalschains (CRAN).** Direct implementation of MA-LSCh-CMA (Molina et al. 2010). Note: CMA-ES is the *local* method inside a steady-state GA.

**E4. FOQUS / ParAMS / Optuna / Nevergrad / Pints.** All expose CMA-ES alongside SciPy/NLopt local optimizers; chaining is up to the user via solver-portfolio scripts. ParAMS (SCM, https://www.scm.com/doc/params/) and Pints (cardiac/electrophysiology modelling, https://pints.readthedocs.io/) are the most idiomatic for parameter-estimation workflows.

## Recommendations

For a user implementing CMA-ES (outer) → local-refiner (inner) on a complex non-linear landscape:

1. **Default recipe (small/medium problems, n ≤ 50):** Run pycma's `cma.fmin2` with `bipop=True` and a wide initial sigma; once it stagnates (≥ 1000 evals without 1e-6 improvement in best-so-far), warm-start `scipy.optimize.minimize(method='L-BFGS-B')` from the returned mean. If your objective is a sum-of-squared residuals, prefer `scipy.optimize.least_squares(method='lm')` for the polish — this is the closest available stand-in for a true CMA-ES → Levenberg–Marquardt chain. Threshold to change: if L-BFGS-B / LM consistently fails to improve over CMA-ES's best, the landscape is likely non-smooth or finite-difference gradients are unreliable; switch to `method='Nelder-Mead'` or `BOBYQA` (NLopt).

2. **Larger or constrained problems (50 ≤ n ≤ 500):** Use **pagmo2/pygmo's `mbh` wrapping `cmaes` + `nlopt('lbfgs')` or `nlopt('slsqp')`** — the only mature open-source implementation of the chained scheme. If n > 500, switch the outer optimizer to LM-CMA-ES (Loshchilov 2017, arXiv:1404.5520).

3. **Per-candidate refinement (true memetic / Lamarckian) instead of one-shot polish:** Implement Hansen 2011's `es.inject()` protocol — within each generation, pick the best 1–2 sampled points, run L-BFGS-B for a small budget (≤ 5n evaluations) with finite-difference Jacobian, and inject the refined solutions back. The most aggressive memetic option but uses far more function evaluations per generation; reserve it for cases where each evaluation is cheap and the landscape has many narrow local basins.

4. **If your problem is least-squares with an analytic Jacobian:** Quasi-Newton with gradient information is *much* faster than CMA-ES on a quadratic objective. Per Hansen's own characterisation on cma-es.github.io: *"on purely convex-quadratic functions, f(x)=xᵀHx, BFGS (Matlab's `fminunc`) is typically faster by a factor of about ten… On the most simple quadratic function f(x)=‖x‖² BFGS is faster by a factor of about 30."* The reason to chain rather than restart LM repeatedly is that LM cannot escape local basins; CMA-ES finds the right basin.

5. **Benchmarks to consult before choosing:** Loshchilov, Schoenauer & Sebag (GECCO BBOB 2013) for HCMA on BBOB, Liao & Stützle (CEC 2013) for iCMAESILS, and Varelas (GECCO 2019, bbob-largescale, comparing CMA-ES variants with L-BFGS-B at n up to 320) for an honest picture of where pure L-BFGS-B dominates and where it collapses.

## Caveats

- **The term "memetic CMA-ES" most often refers to GA-outer + CMA-ES-inner** (the MA-LS-Chains tradition), which is the *inverse* of the user's intended pattern. Read carefully: Molina, Lozano et al.'s work and the `Rmalschains` package put CMA-ES on the inside.
- **No prominent published algorithm chains CMA-ES specifically with Levenberg–Marquardt.** The hybrid literature has preferred derivative-free polish (NEWUOA, BOBYQA, Nelder-Mead, STEP, Mtsls1) or L-BFGS-B over true LM. The few CMA-ES vs. LM works (Bledsoe et al. 2011; Grbić et al. 2016; Bhattacharjee et al. 2020) are head-to-head comparisons rather than chained schemes — the user should view the CMA-ES → LM pipeline as an engineering composition rather than a named algorithm from the literature.
- **HCMA's "line searches" terminology is slightly misleading:** NEWUOA is a derivative-free trust-region method with a quadratic interpolation model, and STEP is a coordinate-wise division strategy. Neither is gradient-based in the BFGS sense.
- **Function-evaluation budget allocation between the outer CMA-ES and the inner refiner is under-studied.** HCMA uses 10·n NEWUOA evaluations as an initial probe; Atia et al. let the local optimizer run to its own tolerance. There is no theoretical guidance on the optimal split; the user must tune empirically.
- **Lamarckian vs. Baldwinian:** nearly all CMA-ES memetic schemes in the literature are Lamarckian (the refined solution replaces or augments the parent). Baldwinian variants are rare in continuous-domain CMA-ES.
- **iCMAESILS CEC 2013 ranking:** secondary sources disagree about 1st vs. 2nd place; either way it shared the top tier with NBIPOP-aCMA-ES. The primary CEC 2013 competition results page was not accessible during this research.

### Completion table

| Spec item | Covered in |
|---|---|
| Full citations | Sections A–D |
| Hybrid structure descriptions | Each entry in A–D |
| Local optimizer named per entry | Each entry |
| Lamarckian vs. Baldwinian | A1, A4, Caveats |
| Budget allocation | B1 (10·n NEWUOA), B2 (competition phase), B4 (own tolerance), Recommendations |
| Key findings per paper | Each entry + Key Findings section |
| Links/DOIs/arXiv IDs | All entries |
| Software packages | Section E |
| Grouping by approach type | Sections A (foundational), B (modern hybrids), C (niching), D (applications), E (software) |
| Comparative context (BBOB / CEC benchmarks) | B1, B2, B6, Recommendations item 5 |
| Theoretical justification | Key Findings 1–4, Caveats |
| Injection mechanism | A4, Recommendations item 3 |
| Recommendations | Recommendations section |