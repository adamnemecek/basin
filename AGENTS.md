# AGENTS.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository.

## State

This is very early-stage alpha code. The public API is not yet defined and feel
free to iterate on it and make breaking changes as needed.

## What this is

`basin` is a Rust library crate for numerical optimization, inspired by
`argmin`. The framework (problem traits, state, solver loop, termination
layer, math abstraction) is in place along with two concrete solvers
(gradient descent, Nelder-Mead) and two param backends (`Vec<f64>`,
`nalgebra`). The public API is still iterating; see the "State" section
above.

## Commands

- `cargo build`: build the library.
- `cargo test`: run tests.
- `cargo test <name>`: run a single test by name.
- `cargo clippy --all-features`: lint (matches the pre-commit hook; always pass
  `--all-features`).
- `cargo fmt`: format (also enforced by pre-commit).

The dev environment is provided by `devenv.nix` (loaded automatically via
`direnv` from `.envrc`). It pins Rust 1.84.1 (matches `rust-version` in
`Cargo.toml`) and adds the `wasm32-unknown-unknown` target plus tooling:
`cargo-llvm-cov`, `cargo-flamegraph`, `cargo-audit`, `cargo-deny`, `cargo-msrv`,
`samply`, `wasm-pack`, `go-task`. Pre-commit hooks run `clippy` (with
`allFeatures = true`) and `rustfmt`.

## Architecture

The framework follows argmin's overall shape: a generic driver loop (`Executor`)
iterates a `Solver` over a `State`, calling into user-provided `Problem` traits.

- `src/lib.rs`: public re-exports only.
- `src/core.rs` (module file) + `src/core/`: the framework:
  - `problem.rs`: traits the *user* implements: `CostFunction`, `Gradient` (more
    to come: `Hessian`, `Jacobian`, `Operator`).
  - `state.rs`: the `State` trait plus concrete states. `BasicState<P>` for
    single-iterate solvers (param, cost, gradient, iter); `SimplexState<V>`
    for simplex-based solvers (n+1 vertices, parallel costs, iter).
    `GradientState` extends `State` for solvers that produce gradients.
    Fields are `pub(crate)`; external access goes through trait methods or
    accessors like `SimplexState::vertices()` / `costs()`.
  - `solver.rs`: `Solver` trait: `init(&problem, state) -> state` (one-time
    setup, e.g. seeding cost/gradient at iter 0) and
    `next_iter(&problem, state) -> state`, plus a `terminate` hook.
  - `executor.rs`: `Executor` owns problem + state + solver and drives the loop
    until termination. `run()` returns an `OptimizationResult<S>` carrying
    the final state and `TerminationReason`.
  - `termination.rs`: `TerminationCriterion<S>` trait plus framework-level
    criteria (`MaxIter`, `GradientTolerance`, `ParamTolerance`,
    `CostTolerance`, `MaxTime`). Per tenet 3, criteria are bound on the
    minimum state shape they need (e.g. `GradientTolerance` requires
    `S: GradientState`), so derivative-free solvers can't be paired with
    them by mistake.
  - `math.rs` + `math/`: the math layer the solvers depend on. Traits
    (`ScaledAdd<S>`, `NormSquared`) plus per-backend impls
    (`math/vec.rs` for `Vec<f64>`, `math/nalgebra_backend.rs` behind the
    `nalgebra` feature).
- `src/solver.rs` + `src/solver/`: concrete solvers. Currently
  `gradient_descent.rs` (with pluggable `step_size.rs`: `Constant`,
  `Backtracking`) and `nelder_mead.rs`.

Module convention: **newer style, no `mod.rs`**: use `src/foo.rs` for the module
file and `src/foo/bar.rs` for submodules. Do not introduce `mod.rs` files.

## Design tenets (deliberate departures from argmin)

These shape API decisions and are non-obvious from the code alone:

1. **Keep argmin's vocabulary and overall shape** (`Executor`, `Solver`,
   `Problem` traits, `IterState`-style state). Diverge only when one of the
   tenets below demands it.
2. **No per-version backend feature gates.** Linear-algebra backends (nalgebra,
   ndarray, plain `Vec`, ...) gate behind a *single* feature each (`nalgebra`,
   `ndarray`), pinning one version. A backend major bump becomes a basin major
   bump. Do **not** introduce features like `nalgebra-v0_33` /
   `nalgebra-v0_34`---argmin does this for compatibility, basin deliberately
   doesn't.
3. **Termination criteria are framework-level, not per-solver.** Generic
   stopping conditions (`max_iter`, `gradient_tolerance`, `param_tolerance`,
   `cost_tolerance`, `max_time`, and once constraints land
   `feasibility_tolerance` / `kkt_tolerance`) belong on the `Executor` / a
   shared termination layer, configured uniformly across solvers.
   Solver-specific knobs stay on the solver. Subtlety: derivative-free solvers
   (Nelder-Mead, SA) have no gradient, so termination must be pluggable / opt-in
   based on what state and problem expose---not a fixed set of fields.
4. **First-class constraints (planned).** argmin has no general constraint
   interface; basin will. Box bounds and linear (in)equalities at minimum, with
   a generic hook for nonlinear constraints later. Constraints describe the
   *problem*, so they live on the problem side, not as executor config. Solvers
   declare support via marker traits / associated types; constrained problems
   handed to unconstrained solvers should be a compile-time error, with an
   opt-in adapter (projection / penalty / barrier) to wrap unconstrained solvers
   when needed. Concrete trait design is deferred until the first constrained
   solver (likely projected gradient on box constraints) --- designing on paper
   without a solver to validate against tends to need redoing.

## WASM as a hard constraint

basin must build for `wasm32-unknown-unknown` out of the box. This is a
constraint on dependencies, not a feature:

- Every default dep must be wasm-compatible. Anything that isn't (file I/O,
  threads, BLAS/LAPACK-linked math, system clocks pre-recent-Rust) must sit
  behind a non-default feature.
- For `max_time` termination, use `web-time` or feature-gate the time-based
  criterion --- `std::time::Instant` is only reliable on recent wasm targets.
- No rayon / parallelism in default features. Gate parallel evaluation behind an
  opt-in feature.
- nalgebra and ndarray are wasm-fine in their pure-Rust configurations;
  BLAS-backed configurations are not --- pick the pure-Rust paths when both
  exist.
- Some solvers (e.g. L-BFGS-B) traditionally lean on BLAS/LAPACK. Prefer
  pure-Rust implementations; if a solver can't realistically run on wasm,
  document that in a per-solver compat note rather than weakening the wasm
  guarantee.
- CI must include `cargo build --target wasm32-unknown-unknown` so wasm
  regressions can't slip in. The `wasm32-unknown-unknown` target is already
  installed via `devenv.nix`.

## MSRV is externally constrained — do not bump casually

basin's MSRV (currently **Rust 1.84.1**, pinned in `Cargo.toml` and
`devenv.nix`) is set by downstream consumers, not by basin's own preferences:

- **Primary constraint: CRAN.** A planned R-package wrapper around basin must
  build under whatever Rust toolchain CRAN ships. CRAN's pin moves slowly and
  lags stable Rust significantly. Bumping basin's MSRV above CRAN's pin makes
  the R bindings unshippable.
- **Secondary (currently non-binding): Python bindings.** Planned eventually.
  PyO3 / maturin track recent stable Rust, so this is unlikely to bind tighter
  than CRAN --- noting it for completeness.

Practical consequences:

- Do **not** bump `rust-version` or the `devenv.nix` Rust pin without verifying
  the current CRAN Rust toolchain version first.
- Every new dependency (and dev-dependency, which `cargo publish --dry-run` and
  CI exercise) must compile under the MSRV. Transitive deps that pull in
  newer-edition crates (e.g. `clap 4.6+` requiring `edition2024`, only stable in
  Rust 1.85) need to be pinned, replaced, or worked around.
- Prefer deps with small, stable transitive trees over feature-rich ones with
  sprawling dep graphs. Each transitive dep is another chance to silently lose
  MSRV compatibility.
- When MSRV pain forces a workaround (e.g. pinning a transitive dep to an older
  major), document the *reason* in `Cargo.toml` next to the pin so future-you
  doesn't lift it without re-checking CRAN.

## Provisional choices (deferred, not tenets)

These are working decisions to revisit, not permanent design.

- **Scalar type is hardcoded to `f64`.** Solvers, `BasicState`, and tolerance
  defaults all assume `f64`. Trade: simpler trait bounds and clearer
  algorithm-constant defaults today, at the cost of a future mechanical refactor
  when scalar-genericity is wanted.
  - **Why deferred:** project scope explicitly includes ensmallen-style
    stochastic optimization (SGD, Adam, RMSProp) eventually, where f32 is the
    natural scalar. So scalar-genericity *is* coming --- just not now.
  - **Trigger to generalize:** the first stochastic solver lands, OR a real f32
    use case appears, OR the bound-boilerplate cost of doing it preemptively
    starts feeling cheaper than the refactor cost. Plan: switch to
    `F: num_traits::Float` on `BasicState<P, F>`, `GradientDescent<F>`, etc. The
    `ScaledAdd<S>` trait is already generic, so the math layer is ready.
  - **Not to do:** add a "fake" scalar generic where defaults still only work in
    `f64`. Either commit to scalar-genericity properly (per-scalar algorithm
    constants, validated f32 paths) or stay f64-only honestly.

## Repo structure: single crate, not a workspace

basin is intentionally one crate, not an argmin-style multi-crate workspace.
argmin's splits exist for reasons that mostly don't apply here:

- `argmin-math` isolates its per-backend-version feature explosion. **Tenet 2
  deletes this reason**---math abstractions live in `basin` behind one feature
  per backend (`nalgebra`, `ndarray`, ...).
- Observer / checkpointing crates exist to isolate heavy optional deps (slog,
  serde, TUI). basin has none of these yet.
- `spectator` is a viz binary, `argmin-py` is Python bindings --- both would be
  separate from a Rust library regardless.

Use Cargo features (e.g. `nalgebra`, `ndarray`, `serde`) for optional
integrations. Split into a workspace only when there's a concrete trigger:

- An observer with heavy deps → `basin-observer-foo`.
- Test problems that other crates want to depend on independently →
  `basin-testfunctions`.
- Python bindings → `basin-py`.
- A viz/CLI tool → separate binary crate.
- Core stabilizes while extras churn and need independent versioning.

Converting single-crate → workspace later is cheap (`git mv` the crate into
`crates/basin/`, add a workspace `Cargo.toml`). Don't pre-pay the complexity.
