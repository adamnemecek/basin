# TODO

Ordered by recommended sequence --- each item is easier or better-informed once
the previous lands.

## General design

- [ ] **Constraints (tenet 4).** Box bounds shipped (`BoxConstraints`,
      consumed by `ProjectedGradientDescent` / `LBFGSB` / `Trf` /
      `BoundedCmaEs`). Linear inequalities `A x ≤ b` shipped
      (`LinearInequalityConstraints` + the `LogBarrier` adapter + the
      log-barrier `BarrierMethod`, a `constrOptim`-style layer over an inner
      `GradientDescent`; nalgebra/faer via `MatVec`/`MatTransposeVec`). Linear
      equalities `A x = b` shipped (`LinearEqualityConstraints` + the
      `AugmentedLagrangian` adapter + `AugmentedLagrangianMethod`, a
      penalty-plus-multiplier outer loop over an inner `GradientDescent`;
      tolerates infeasible starts, nalgebra/faer via `MatVec`/`MatTransposeVec`).
      Remaining: phase-1 feasibility (the barrier needs a strictly feasible
      start today; the augmented Lagrangian does not); broadening the inner
      solver beyond `BasicState`/`GradientDescent` (see the dedicated
      inner-solver-agnostic item below); `MatVec`/`MatTransposeVec` impls for
      `Vec<f64>`/`ndarray` to lift the backend gate; a framework-level
      `FeasibilityTolerance` once a 2nd equality-constrained solver justifies
      it (tenet 3); nonlinear equality and nonlinear (in)equality constraints.
      Keep deferring a `Constraint` supertrait — box (projection),
      linear-inequality (barrier), and linear-equality (penalty+multipliers)
      still share no feasibility op beyond accessors (tenet 4).
- [ ] **Make `BarrierMethod` / `AugmentedLagrangianMethod` inner-solver-agnostic.**
      Both currently hard-code the inner solver as
      `So: for<'a> Solver<Adapter<'a, P>, BasicState<V>>` (`Adapter` =
      `LogBarrier` resp. `AugmentedLagrangian`), so in practice the *only*
      usable inner is `GradientDescent` — you can swap its line search, but
      that's the whole flexibility. The original intent was "wrap a barrier/AL
      around L-BFGS, LM, Nelder-Mead, …"; that is **not** what ships. Two
      orthogonal blockers, learned while shipping the AL method:

  1. **State-type lock-in (fixable).** The `BasicState<V>` slot rejects, at
     compile time, every solver that carries its own state: L-BFGS / L-BFGS-B
     (`LbfgsState<V>`), BFGS (`QuasiNewtonState`), Nelder-Mead
     (`BasicSimplexState<V>`). The state-seeding abstraction the `BarrierMethod`
     doc gestures at *already exists*: `MemeticInner<V>` in
     `src/solver/cma_inject.rs` —
     `type State: State<Param=V>; fn seed(&self, x: &V, sigma: f64) -> State;
     fn work_units(&self, &State) -> u64` — already impl'd for `NelderMead`,
     `LevenbergMarquardt`, and `LBFGSB`. The lift: change the bound to
     `So: MemeticInner<V> + for<'a> Solver<Adapter<'a, P>, So::State>`, swap
     `BasicState::new(param)` for `inner.seed(&param, …)`, read the iterate back
     from `So::State`, and bound `So::State: GradientState<Param=V>` so eval
     aggregation (`cost_evals` / `gradient_evals`) still works.

  2. **Least-squares inners are a category error (do *not* "fix").** LM /
     Gauss-Newton / `Trf` already use `BasicState<V>` (state isn't what excludes
     them) but require `P: Residual + Jacobian`, which the adapters deliberately
     don't expose (`CostFunction + Gradient` only). A barrier `f − μ·Σ log sᵢ`
     or Lagrangian `f + λᵀc + (ρ/2)‖c‖²` is not a sum of squares, so there's no
     residual vector to hand LM — wrapping a barrier destroys the structure LM
     exists to exploit. Constrained least-squares is a separate design (cf.
     `Trf`, which bakes box bounds into the LM trust region).

  **Trigger is met.** The `barrier_method.rs` doc deferred this "until there's a
  second consumer to validate the shape against." `AugmentedLagrangianMethod` is
  now that second consumer — both share the identical
  `Solver<Adapter, BasicState<V>>` shape — so the project's ≥2-consumers rule
  (tenet 4 / the "Solver composition" section in `AGENTS.md`) now permits the
  generalization.

  **Open design wrinkle to resolve first.** `MemeticInner::seed(&self, x, sigma)`
  is CMA-flavored: `sigma` is the initial step / simplex scale — meaningful for
  Nelder-Mead, meaningless for a barrier/AL subproblem (the `LevenbergMarquardt`
  and `LBFGSB` impls already ignore it as `_sigma`). Passing a dummy `sigma`
  silently mis-sizes a Nelder-Mead inner's simplex. So decide: (a) reuse
  `MemeticInner` with a documented dummy and accept the Nelder-Mead caveat, or
  (b) carve a slimmer sibling trait (e.g. `WarmStart<V> { type State; fn
  seed(&self, x: &V) -> State; }`) that `MemeticInner` can supertrait. This is
  exactly the "validate the abstraction against a 2nd consumer" call, and it
  overlaps the `Composed<Outer, Inner>` question under *Cleanup / design debt*
  below — resolve them together.

  Files: `src/solver/barrier_method.rs`, `src/solver/augmented_lagrangian_method.rs`,
  `src/solver/cma_inject.rs` (the `MemeticInner` trait + impls),
  `src/core/state.rs` (`GradientState`). Integration-test templates:
  `tests/barrier_method_*.rs`, `tests/augmented_lagrangian_*.rs`,
  `tests/cma_inject_*.rs` (for the `MemeticInner` usage pattern).
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
