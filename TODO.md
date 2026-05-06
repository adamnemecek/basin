# TODO

Ordered by recommended sequence --- each item is easier or better-informed once
the previous lands.

## Next up

See `ROADMAP.md` for the long-arc plan toward LM-with-bounds and CMA-ES.
This section tracks the immediate next session's discrete items.

- [x] **S2b: sparse `Jacobian::Output` + sparse `LinearSolve`.**
      *(done)* CSC sparse Jacobian + sparse Cholesky/QR landed for
      both backends; sparse Gauss-Newton works on the new
      `SparseLeastSquares` fixture. See `ROADMAP.md` Phase 1 for
      decisions and deferred follow-ups.

- [ ] **S4: Levenberg-Marquardt (unconstrained).**
      Gauss-Newton + Marquardt damping with the Nielsen 1999 λ-update.
      Inherits dense + sparse + both backends from S2b. Read
      Madsen/Nielsen/Tingleff (2004) + Nielsen (1999) + skim MINPACK
      `lmder` first.

## Deferred (not now)

- [ ] **README and rustdoc.** Wait until the public API stops churning ---
      premature docs rot fast.
- [ ] **L-BFGS or Adam.** Too big a chunk for now; needs the termination layer +
      state-with-history first.
- [ ] **Constraints (tenet 4).** Trait design deferred until the first
      constrained solver is being written (likely projected gradient on box
      bounds).
- [ ] **Generalize over scalar (`f64` → `F: Float`).** Per the
      provisional-choices section in `AGENTS.md`. Triggered by the first
      stochastic solver or a real f32 use case.
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
- [ ] **Rastrigin** — N-D, highly multimodal (cosine ripple). Defer (global).
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
      S0 — see `ROADMAP.md`. `# Contract` heading + `**Caller must:**` /
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
See `AGENTS.md` for the design tenets and constraints that shape these
decisions.
