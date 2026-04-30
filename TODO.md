# TODO

Ordered by recommended sequence — each item is easier or better-informed once the previous lands.

## Next up

- [ ] **`StepSize` abstraction + backtracking impl.** Refactor `GradientDescent` to take a `StepSize` strategy; ship `Constant(α)` (current behavior) and `Backtracking { … }` as the first two impls. Backtracking pressures `next_iter` calling `problem.cost()` multiple times per iteration, surfacing what to cache vs recompute. Naming: `StepSize` over `LineSearch` — the latter implies 1D search along a direction, which is only one family (constant, BB, decay schedules don't fit). Gives us strategy variety for the termination layer without pretending step-size choice is a separate algorithm.
- [ ] **Termination criteria layer (tenet 3).** Pluggable framework-level termination: `max_iter`, `gradient_tolerance`, `param_tolerance`, `cost_tolerance`, `max_time`. Designed against `GradientDescent` with both step-size strategies so the abstraction is validated by variety in per-iter cost-eval patterns. Watch out: derivative-free solvers (Nelder-Mead, SA) have no gradient — design must handle "criterion N/A for this solver" cleanly.
- [ ] **WASM build verification.** Run `cargo build --target wasm32-unknown-unknown` and confirm it works. Cheap; we've committed to WASM as a hard constraint without ever verifying. Pair with a minimal GitHub Actions CI workflow that runs build/test/clippy/fmt + the wasm build, so regressions can't slip in.

## Deferred (not now)

- [ ] **README and rustdoc.** Wait until the public API stops churning — premature docs rot fast.
- [ ] **L-BFGS or Adam.** Too big a chunk for now; needs the termination layer + state-with-history first.
- [ ] **Constraints (tenet 4).** Trait design deferred until the first constrained solver is being written (likely projected gradient on box bounds).
- [ ] **Generalize over scalar (`f64` → `F: Float`).** Per the provisional-choices section in `AGENTS.md`. Triggered by the first stochastic solver or a real f32 use case.
- [ ] **Linear-algebra backend impls (nalgebra, ndarray) behind features.** When the first user/example wants something other than `Vec<f64>`, or when we add a solver that benefits from BLAS-shaped operations.
- [ ] **Real bench tool (divan / Criterion) when MSRV pressure lifts.** Hand-rolled bench works for now; revisit when CRAN's Rust pin moves past 1.85.

See `AGENTS.md` for the design tenets and constraints that shape these decisions.
