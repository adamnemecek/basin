# TODO

Ordered by recommended sequence — each item is easier or better-informed once the previous lands.

## Next up

- [ ] **Second solver: GD with backtracking line search.** Same problem traits as plain GD; pressures `next_iter` calling `problem.cost()` multiple times per iteration, surfaces what to cache vs recompute, separates step-direction from step-size logic. Reuses Rosenbrock; ~one file in `src/solver/`. Gives us a second data point for the termination layer.
- [ ] **Termination criteria layer (tenet 3).** Pluggable framework-level termination: `max_iter`, `gradient_tolerance`, `param_tolerance`, `cost_tolerance`, `max_time`. Designed against the two solvers from the step above so the abstraction is validated by variety. Watch out: derivative-free solvers (Nelder-Mead, SA) have no gradient — design must handle "criterion N/A for this solver" cleanly.
- [ ] **WASM build verification.** Run `cargo build --target wasm32-unknown-unknown` and confirm it works. Cheap; we've committed to WASM as a hard constraint without ever verifying. Pair with a minimal GitHub Actions CI workflow that runs build/test/clippy/fmt + the wasm build, so regressions can't slip in.

## Deferred (not now)

- [ ] **README and rustdoc.** Wait until the public API stops churning — premature docs rot fast.
- [ ] **L-BFGS or Adam.** Too big a chunk for now; needs the termination layer + state-with-history first.
- [ ] **Constraints (tenet 4).** Trait design deferred until the first constrained solver is being written (likely projected gradient on box bounds).
- [ ] **Generalize over scalar (`f64` → `F: Float`).** Per the provisional-choices section in `AGENTS.md`. Triggered by the first stochastic solver or a real f32 use case.
- [ ] **Linear-algebra backend impls (nalgebra, ndarray) behind features.** When the first user/example wants something other than `Vec<f64>`, or when we add a solver that benefits from BLAS-shaped operations.
- [ ] **Real bench tool (divan / Criterion) when MSRV pressure lifts.** Hand-rolled bench works for now; revisit when CRAN's Rust pin moves past 1.85.

See `AGENTS.md` for the design tenets and constraints that shape these decisions.
