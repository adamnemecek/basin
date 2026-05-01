# TODO

Ordered by recommended sequence — each item is easier or better-informed once the previous lands.

## Next up

- [ ] **Termination criteria layer (tenet 3).** Pluggable framework-level termination: `max_iter`, `gradient_tolerance`, `param_tolerance`, `cost_tolerance`, `max_time`. Designed against `GradientDescent` with both step-size strategies so the abstraction is validated by variety in per-iter cost-eval patterns. Watch out: derivative-free solvers (Nelder-Mead, SA) have no gradient — design must handle "criterion N/A for this solver" cleanly.
- [ ] **CI workflow.** Minimal GitHub Actions running build/test/clippy/fmt on `Vec<f64>` and `--features nalgebra`, plus `cargo build --target wasm32-unknown-unknown` (default + `--features nalgebra`), so backend/wasm regressions can't slip in.

## Deferred (not now)

- [ ] **README and rustdoc.** Wait until the public API stops churning — premature docs rot fast.
- [ ] **L-BFGS or Adam.** Too big a chunk for now; needs the termination layer + state-with-history first.
- [ ] **Constraints (tenet 4).** Trait design deferred until the first constrained solver is being written (likely projected gradient on box bounds).
- [ ] **Generalize over scalar (`f64` → `F: Float`).** Per the provisional-choices section in `AGENTS.md`. Triggered by the first stochastic solver or a real f32 use case.
- [ ] **`ndarray` backend.** `nalgebra` is in; add `ndarray` behind a `ndarray` feature when the first user/example wants it.
- [ ] **Real bench tool (divan / Criterion) when MSRV pressure lifts.** Hand-rolled bench works for now; revisit when CRAN's Rust pin moves past 1.85.

See `AGENTS.md` for the design tenets and constraints that shape these decisions.
