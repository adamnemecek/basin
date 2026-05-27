---
description: Contracts for solvers that run another solver as a sub-step (memetic CMA, basin hopping, barrier/augmented-Lagrangian, multi-start polish) — eval aggregation, inner-criteria statefulness, failure routing, and the WarmStart/MemeticInner seeding split.
paths:
  - "crates/basin/src/solver/**/*.rs"
  - "crates/basin/src/core/inner.rs"
  - "crates/basin/src/core/executor.rs"
---

# Solver composition

Some solvers run another solver as a sub-step (memetic CMA-ES + LM, basin
hopping, barrier / augmented-Lagrangian, multi-start polish, …). The
composition primitive is `run_loop(&problem, state, &mut solver, &mut
criteria, max_iter)` in `src/core/executor.rs`; the builder-style adapter
`InnerExecutor<S, So>` in `src/core/inner.rs` wraps it for the common case
where an outer solver stores a pre-configured inner and reuses it across
outer iters.

## Three contracts every outer solver must follow

1. **Eval aggregation.** After each `inner.run(&problem, inner_state)`, roll
   the inner's `cost_evals()` into the outer state via
   `increment_cost_evals(...)` (and `gradient_evals()` via
   `increment_gradient_evals(...)` when both inner and outer are
   `GradientState`). Skipping this silently corrupts `MaxCostEvals` budgets
   and the public `result.cost_evals()` read. The contract is spelled out on
   `Solver::next_iter`'s rustdoc; `crates/basin/tests/inner_executor.rs`
   asserts it.

2. **Inner termination criteria must be stateless across calls.** An
   `InnerExecutor` keeps its `Vec<Box<dyn TerminationCriterion<S>>>` for its
   whole lifetime and reuses it on every `run()`. Fine for `MaxIter`,
   `*Tolerance`, and `MaxCostEvals` (no internal state, or state that resets
   meaningfully each call). **Not** fine for `MaxTime`, whose internal
   `start: Option<Instant>` is set on the first `check` and persists — on the
   second `run()` it fires prematurely. Outer solvers needing per-run criteria
   should call `run_loop` directly with a fresh `Vec` per call.

3. **Failure routing.** `run()` returns a full `OptimizationResult` carrying a
   `TerminationReason`. Use `reason.is_failure()` (true only for
   `SolverFailed`) to decide whether to bubble the failure via the outer's
   mid-iter `Option<TerminationReason>` return. Everything else — `MaxIter`,
   the tolerance reasons, `SolverConverged` — is a clean stop: the outer
   consumes the inner's final iterate and continues. The common bug is
   forgetting to propagate `SolverFailed` and treating an aborted inner run as
   a successful one.

## `InnerExecutor` vs `run_loop`

Reach for `InnerExecutor` when the outer wants to expose `inner_max_iter` /
`inner.terminate_on(...)` to its users via builder methods that mirror the
framework. Reach for raw `run_loop` when the outer needs to reconstruct
criteria each call (statefulness escape hatch) or wants per-call criteria
passed through a different surface.

## Seeding an inner's state: `WarmStart` (+ `MemeticInner`)

An outer that re-solves a subproblem from the current iterate must *build* the
inner's state, not just drive it — and inners carry different state shapes
(`BasicState`, `LbfgsState`, `QuasiNewtonState`, `BasicSimplexState`).

- `WarmStart<V>` (`src/core/inner.rs`) is the minimal primitive: `type State:
  State<Param=V>` + `seed(&self, x) -> State` (σ-free, the solver's natural
  default scale).
- `MemeticInner<V>: WarmStart<V>` (`src/solver/cma_inject.rs`) extends it with
  `seed_scaled(x, σ)` (defaults to `seed`; only Nelder-Mead's σ-scaled simplex
  overrides it) and `work_units` for CMA-injection eval aggregation.

Two consumer families validate the split: the barrier / AL methods bound `So:
WarmStart<V>` with `So::State: GradientState` (gradient inners only — they read
`cost_evals`/`gradient_evals` directly, so they don't need `work_units`, and
the `GradientState` bound excludes the only σ-sensitive inner, Nelder-Mead, so
the σ-free `seed` is exactly right); CMA-injection (`CmaInject` /
`BoundedCmaInject`) bound `I: MemeticInner<V>` and call `seed_scaled`. This
split resolved the "dummy-σ wrinkle" — barrier / AL never pass a meaningless σ.

## Don't grow a `Composed<Outer, Inner>` type until ≥2 concrete consumers want it

Same spirit as the "no `Constraint` supertrait until two consumers" rule.
`WarmStart` / `MemeticInner` cover *state seeding* for both memetic CMA and the
barrier/AL family, but they are not a `Composed` abstraction: they say nothing
about the outer loop, eval-aggregation routing, or failure bubbling (those stay
the three contracts above). A coarser `Composed` marker is still unmotivated —
the shipped composed solvers share the three contracts and (some) `WarmStart`,
and nothing more.
