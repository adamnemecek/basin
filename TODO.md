# TODO

Ordered by recommended sequence --- each item is easier or better-informed once
the previous lands.

## Next up

This section tracks the immediate next session's discrete items.
The long-arc plan toward LM-with-bounds and CMA-ES is done as of
S13 — see git history for the per-session prose if needed.

- [x] **S2b: sparse `Jacobian::Output` + sparse `LinearSolve`.**
      *(done)* CSC sparse Jacobian + sparse Cholesky/QR landed for
      both backends; sparse Gauss-Newton works on the new
      `SparseLeastSquares` fixture.

- [x] **S7: Wasm-safe RNG abstraction + simple stochastic solver.**
      *(done)* `RandomSearch` (elitist 1+λ) lands on top of new
      `BasicPopulationState<V>` / `PopulationState` and a new vector-
      tier `SampleUniformBox` trait. RNG is `rand 0.9` +
      `rand_chacha 0.9` (0.10 wants edition2024, blocks MSRV);
      seeded `ChaCha8Rng` works on `wasm32-unknown-unknown` with no
      JS shim.

- [x] **S8: CMA-ES (vanilla).**
      Second LA-heavy solver: covariance + eigendecomposition. Read
      Hansen, *The CMA Evolution Strategy: A Tutorial* first; pin
      `(μ/μ_w, λ)`-CMA-ES with rank-μ + rank-1 updates and tutorial-
      default constants. The `SampleStandardNormal` trait designed
      against this caller's needs.

## Deferred (not now)

- [ ] **README and rustdoc.** Wait until the public API stops churning ---
      premature docs rot fast.
- [x] **L-BFGS-B.** *(done)* Faithful port of Nocedal's Fortran v3.0
      with iteration-wise parity verified. Landed:
      `references/lbfgsb-v3.0/` (BSD-3 vendored source + `NOTES.md`);
      `MoreThuente` line search (`line_search/more_thuente.rs`, port
      of `dcsrch` + `dcstep`); `LbfgsState` with `(s, y)` history +
      compact-form Gram blocks (`core/state/lbfgs.rs`); generalized
      Cauchy point + compact-form helpers
      (`solver/lbfgsb/{cauchy,compact}.rs` — `formt`, `bmv`, pure-Rust
      Cholesky / triangular solves, `hpsolb` min-heap, full `cauchy`
      port); subspace minimization (`solver/lbfgsb/subsm.rs` — `WᵀZd`,
      `L·E·Lᵀ` middle-system solve via paired triangular solves with
      sign-flip, projected Newton + uniform-α bound-backtracking);
      `formk` (`solver/lbfgsb/formk.rs` — incremental `wn1` Gram cache,
      block-Cholesky `L·E·Lᵀ` factor of indefinite `K`); top-level
      `LBFGSB` solver (`solver/lbfgsb.rs`) wired through cauchy +
      freev + formk + cmprlb + subsm + Moré–Thuente + matupd/formt
      with the Fortran `goto 222` history-clear restart path.
      Validation: convergence on Rosenbrock 2D, BoothBoxed (slack +
      tight corner), and a 5-D quadratic across Vec / nalgebra / faer
      backends; iteration-wise parity to ≤ 1e-10 against the Fortran
      reference (`tests/lbfgsb_iter_parity.rs` reading a fixture
      dumped from gfortran-built `references/lbfgsb-v3.0/`).
- [ ] **Constraints (tenet 4).** Trait design deferred until the first
      constrained solver is being written (likely projected gradient on box
      bounds).
- [ ] **Generalize over scalar (`f64` → `F: Float`).** Per the
      provisional-choices section in `AGENTS.md`. The first stochastic
      solver (S7 `RandomSearch`) landed without forcing this — the
      bound-boilerplate cost of preemptive generality still outweighs
      the refactor. Trigger now reads "a real f32 use case appears" or
      "the second stochastic solver needs it" (CMA-ES in S8 will tell us).
- [ ] **Real bench tool (divan / Criterion) when MSRV pressure lifts.**
      Hand-rolled bench works for now; revisit when CRAN's Rust pin moves past
      1.85.

## Test problem corpus

Mirrored from `argmin-testfunctions`
(<https://github.com/argmin-rs/argmin/tree/main/crates/argmin-testfunctions/src>).
Each adds raw `fn` + a `Problem<P>` wrapper under `basin::problems`. Annotated
with dimensionality and character so we can pick what's worth adding next.
"Local-only solvers" (GD, BFGS, Nelder-Mead) get little value from highly
multimodal functions until SA / CMA-ES / a global solver lands — defer those.

- [x] **Rosenbrock** — N-D, smooth, narrow curved valley. Standard hard test
      for first/second-order methods. *(done)*
- [x] **Sphere** — N-D, `Σ xᵢ²`, convex. Trivial canary; dedups the inline
      `Sphere` in `tests/nelder_mead.rs`. *(done)*
- [x] **Beale** — 2D, smooth, flat region near optimum at `(3, 0.5)`. Good
      second smooth test for BFGS distinct from Rosenbrock. *(done)*
- [x] **Booth** — 2D, smooth, convex quadratic-like. Easy gradient test. *(done)*
- [x] **Matyas** — 2D, smooth, plate-like. Very easy; mostly a sanity check. *(done)*
- [x] **McCormick** — 2D, smooth, single minimum. *(done)*
- [x] **Goldstein-Price** — 2D, smooth polynomial, single minimum, large
      dynamic range. Stresses step-size control. *(done)*
- [x] **Powell singular** — 4D, smooth, sum-of-squares with rank-deficient
      Jacobian at the optimum. Classic LM benchmark. *(done in S1)*
- [x] **Rosenbrock-as-residuals** — 2D residual factoring of the existing
      Rosenbrock cost; `Σ rᵢ² = rosenbrock(x)`. Fixture for the LM
      track. *(done in S1; lives in `rosenbrock.rs`)*
- [ ] **Three-hump camel** — 2D, smooth, three local minima. Local-solver
      basin-of-attraction test.
- [ ] **Picheny** — 2D, log-rescaled Rosenbrock variant. Same shape, different
      conditioning — useful for line-search behavior.
- [ ] **Zero** — `f(x) = 0` everywhere. Sanity / termination edge case
      (gradient is identically zero).
- [ ] **Himmelblau** — 2D, four equal minima. Defer until a global solver
      makes "which minimum?" interesting.
- [ ] **Ackley** — N-D, multimodal (exp + cos). Defer (global).
- [x] **Rastrigin** — N-D, highly multimodal (cosine ripple). *(done)*
- [ ] **Levy** — N-D, multimodal. Defer (global).
- [ ] **Styblinski-Tang** — N-D, multimodal. Defer (global).
- [ ] **Schaffer (N.2 / N.4)** — 2D, multimodal. Defer (global).
- [ ] **Bukin N.6** — 2D, sharp non-differentiable ridge. Defer (needs
      derivative-free + global; pathological for first-order).
- [ ] **Cross-in-tray** — 2D, multimodal. Defer (global).
- [ ] **Easom** — 2D, single sharp minimum, mostly flat. Defer (needs global
      / good initialization to be meaningful).
- [ ] **Eggholder** — 2D, highly multimodal. Defer (global).
- [ ] **Holder table** — 2D, multimodal with four equal minima. Defer
      (global).

## Cleanup / design debt (review notes)

Surfaced while implementing the termination layer. Not blocking, but each
gets harder to fix as more code piles on.

- [x] **Rustdoc the load-bearing invariants on public traits.** Done in
      S0. `# Contract` heading + `**Caller must:**` /
      `**Implementor must:**` bullets are the established convention;
      `#![warn(missing_docs)]` and `#![warn(rustdoc::broken_intra_doc_links)]`
      are on at the crate root. Filling in docs on items that hold no
      contract (struct fields, trivial constructors) is the open
      follow-up — those are the ~100 `missing_docs` warnings still
      surfaced by the lint.
- [ ] **`ParamVec<F>` marker for solvers doing linear algebra on params.**
      Nelder-Mead needs `V: Clone + ScaledAdd<f64>`; gradient descent needs
      `V: ScaledAdd<f64>`; future solvers will repeat the bound pair. Add a
      blanket-impl marker like
      `trait ParamVec<F>: Clone + ScaledAdd<F> + NormSquared {}` once the
      third solver wants it — premature with only two users.
- [ ] **Unified `Composed<Outer, Inner>` abstraction (or honest "no").**
      Two concrete memetic shapes now exist: `CmaInject` / `BoundedCmaInject`
      (per-generation top-k polish via `MemeticInner`, S11 + S13) and
      `MaLsChCma` (per-individual persistent LS chains, S12). The
      `MemeticInner` trait covers CMA-injection-style composition but
      doesn't model MA-LSCh's persistent-state shape. Question: is there
      a shared `Composed` abstraction (probably *not* `MemeticInner` —
      something coarser like a "composed solver" marker), or do these
      two memetic shapes genuinely have nothing in common worth
      extracting? Resolve by either writing the trait or writing the
      honest "no, these two don't share more than the AGENTS.md
      composition contracts" comment in `core/inner.rs`.

See `AGENTS.md` for the design tenets and constraints that shape these
decisions.
