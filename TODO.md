# TODO

Ordered by recommended sequence --- each item is easier or better-informed once
the previous lands.

## Next up

(empty)

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

- [ ] **Rustdoc the load-bearing invariants on public traits.** Things like
      "`Solver::init` must populate cost/gradient before `next_iter`",
      "criteria are checked before iter 0", "first criterion to fire wins",
      "`gradient()` must match `param()` at end of `next_iter`". These are
      contract, not narrative — easy to forget and not derivable from the
      type signatures. Do this *before* the larger rustdoc pass since these
      pin down current decisions.
- [ ] **`ParamVec<F>` marker for solvers doing linear algebra on params.**
      Nelder-Mead needs `V: Clone + ScaledAdd<f64>`; gradient descent needs
      `V: ScaledAdd<f64>`; future solvers will repeat the bound pair. Add a
      blanket-impl marker like
      `trait ParamVec<F>: Clone + ScaledAdd<F> + NormSquared {}` once the
      third solver wants it — premature with only two users.
See `AGENTS.md` for the design tenets and constraints that shape these
decisions.
