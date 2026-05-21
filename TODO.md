# TODO

Ordered by recommended sequence --- each item is easier or better-informed once
the previous lands.

## General design

- [ ] **Constraints (tenet 4).** Trait design deferred until the first
      constrained solver is being written (likely projected gradient on box
      bounds).
- [ ] **Generalize over scalar (`f64` → `F: Float`).** Per the
      provisional-choices section in `AGENTS.md`. The first stochastic solver
      (S7 `RandomSearch`) landed without forcing this --- the bound-boilerplate
      cost of preemptive generality still outweighs the refactor. Trigger now
      reads "a real f32 use case appears" or "the second stochastic solver needs
      it" (CMA-ES in S8 will tell us).

## Benchmarks

- [ ] **Real bench tool (divan / Criterion).** Hand-rolled bench works for now.
      CRAN moved to Rust 1.91.1 so the `edition2024` blocker is gone --- revisit
      when someone wants to do the bench-rewrite.

## Test problem corpus

Mirrored from `argmin-testfunctions`
(<https://github.com/argmin-rs/argmin/tree/main/crates/argmin-testfunctions/src>).
Each adds raw `fn` + a `Problem<P>` wrapper under `basin::problems`. Annotated
with dimensionality and character so we can pick what's worth adding next.
"Local-only solvers" (GD, BFGS, Nelder-Mead) get little value from highly
multimodal functions until SA / CMA-ES / a global solver lands --- defer those.

- [x] **Rosenbrock** --- N-D, smooth, narrow curved valley. Standard hard test
      for first/second-order methods. *(done)*
- [x] **Sphere** --- N-D, `Σ xᵢ²`, convex. Trivial canary; dedups the inline
      `Sphere` in `tests/nelder_mead.rs`. *(done)*
- [x] **Beale** --- 2D, smooth, flat region near optimum at `(3, 0.5)`. Good
      second smooth test for BFGS distinct from Rosenbrock. *(done)*
- [x] **Booth** --- 2D, smooth, convex quadratic-like. Easy gradient test.
      *(done)*
- [x] **Matyas** --- 2D, smooth, plate-like. Very easy; mostly a sanity check.
      *(done)*
- [x] **McCormick** --- 2D, smooth, single minimum. *(done)*
- [x] **Goldstein-Price** --- 2D, smooth polynomial, single minimum, large
      dynamic range. Stresses step-size control. *(done)*
- [x] **Powell singular** --- 4D, smooth, sum-of-squares with rank-deficient
      Jacobian at the optimum. Classic LM benchmark. *(done in S1)*
- [x] **Rosenbrock-as-residuals** --- 2D residual factoring of the existing
      Rosenbrock cost; `Σ rᵢ² = rosenbrock(x)`. Fixture for the LM track. *(done
      in S1; lives in `rosenbrock.rs`)*
- [ ] **Three-hump camel** --- 2D, smooth, three local minima. Local-solver
      basin-of-attraction test.
- [ ] **Picheny** --- 2D, log-rescaled Rosenbrock variant. Same shape, different
      conditioning --- useful for line-search behavior.
- [ ] **Zero** --- `f(x) = 0` everywhere. Sanity / termination edge case
      (gradient is identically zero).
- [ ] **Himmelblau** --- 2D, four equal minima. Defer until a global solver
      makes "which minimum?" interesting.
- [ ] **Ackley** --- N-D, multimodal (exp + cos). Defer (global).
- [x] **Rastrigin** --- N-D, highly multimodal (cosine ripple). *(done)*
- [ ] **Levy** --- N-D, multimodal. Defer (global).
- [ ] **Styblinski-Tang** --- N-D, multimodal. Defer (global).
- [ ] **Schaffer (N.2 / N.4)** --- 2D, multimodal. Defer (global).
- [ ] **Bukin N.6** --- 2D, sharp non-differentiable ridge. Defer (needs
      derivative-free + global; pathological for first-order).
- [ ] **Cross-in-tray** --- 2D, multimodal. Defer (global).
- [ ] **Easom** --- 2D, single sharp minimum, mostly flat. Defer (needs global /
      good initialization to be meaningful).
- [ ] **Eggholder** --- 2D, highly multimodal. Defer (global).
- [ ] **Holder table** --- 2D, multimodal with four equal minima. Defer
      (global).

## Cleanup / design debt (review notes)

Surfaced while implementing the termination layer. Not blocking, but each gets
harder to fix as more code piles on.

- [x] **Rustdoc the load-bearing invariants on public traits.** Done in S0.
      `# Contract` heading + `**Caller must:**` / `**Implementor must:**`
      bullets are the established convention; `#![warn(missing_docs)]` and
      `#![warn(rustdoc::broken_intra_doc_links)]` are on at the crate root.
      Filling in docs on items that hold no contract (struct fields, trivial
      constructors) is the open follow-up --- those are the \~100 `missing_docs`
      warnings still surfaced by the lint.
- [ ] **`ParamVec<F>`marker for solvers doing linear algebra on params.**
      Nelder-Mead needs `V: Clone + ScaledAdd<f64>`; gradient descent needs
      `V: ScaledAdd<f64>`; future solvers will repeat the bound pair. Add a
      blanket-impl marker like
      `trait ParamVec<F>: Clone + ScaledAdd<F> + NormSquared {}` once the third
      solver wants it --- premature with only two users.
- [ ] **Unified `Composed<Outer, Inner>` abstraction (or honest "no").** Two
      concrete memetic shapes now exist: `CmaInject` / `BoundedCmaInject`
      (per-generation top-k polish via `MemeticInner`, S11 + S13) and
      `MaLsChCma` (per-individual persistent LS chains, S12). The `MemeticInner`
      trait covers CMA-injection-style composition but doesn't model MA-LSCh's
      persistent-state shape. Question: is there a shared `Composed` abstraction
      (probably *not* `MemeticInner` --- something coarser like a "composed
      solver" marker), or do these two memetic shapes genuinely have nothing in
      common worth extracting? Resolve by either writing the trait or writing
      the honest "no, these two don't share more than the AGENTS.md composition
      contracts" comment in `core/inner.rs`.

See `AGENTS.md` for the design tenets and constraints that shape these
decisions.
