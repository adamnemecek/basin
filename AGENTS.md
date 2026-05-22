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
(gradient descent, Nelder-Mead) and four param backends (`Vec<f64>`,
`nalgebra`, `ndarray`, `faer`). The public API is still iterating; see the
"State" section above.

## Commands

- `cargo build`: build the library.
- `cargo test`: run tests.
- `cargo test <name>`: run a single test by name.
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`: lint
- `cargo fmt`: format (also enforced by pre-commit).

The dev environment is provided by `devenv.nix` (loaded automatically via
`direnv` from `.envrc`). It pins Rust 1.91.1 (matches `rust-version` in
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
    single-iterate solvers (param, cost, gradient, iter);
    `BasicSimplexState<V>` for simplex-based solvers (n+1 vertices, parallel
    costs, iter). `GradientState` extends `State` for solvers that produce
    gradients; `SimplexState` extends it for solvers that carry a simplex,
    exposing `vertices()` / `costs()` so termination criteria can bound on
    them. Fields are `pub(crate)`; external access goes through trait methods.
  - `solver.rs`: `Solver` trait: `init(&problem, state) -> state` (one-time
    setup, e.g. seeding cost/gradient at iter 0) and
    `next_iter(&problem, state) -> state`, plus a `terminate` hook.
  - `executor.rs`: `Executor` owns problem + state + solver and drives the loop
    until termination. `run()` returns an `OptimizationResult<S>` carrying
    the final state and `TerminationReason`.
  - `termination.rs`: `TerminationCriterion<S>` trait plus framework-level
    criteria (`MaxIter`, `GradientTolerance`, `ParamTolerance`,
    `CostTolerance`, `SimplexTolerance`, `MaxTime`). Per tenet 3, criteria
    are bound on the minimum state shape they need (e.g. `GradientTolerance`
    requires `S: GradientState`, `SimplexTolerance` requires
    `S: SimplexState`), so derivative-free solvers can't be paired with
    gradient-based criteria by mistake, and single-iterate solvers can't be
    paired with simplex-based ones.
  - `math.rs` + `math/`: the math layer the solvers depend on. Traits
    (`ScaledAdd<S>`, `NormSquared`, `NormInfinity`) plus per-backend impls
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
4. **First-class constraints.** argmin has no general constraint interface;
   basin does. Box bounds and linear (in)equalities at minimum, with a generic
   hook for nonlinear constraints later. Constraints describe the *problem*, so
   they live on the problem side, not as executor config. Solvers declare
   support via marker traits / associated types; constrained problems handed to
   unconstrained solvers are a compile-time error, with an opt-in adapter
   (projection / penalty / barrier) to wrap unconstrained solvers when needed.

   **Status.** Three constraint kinds ship, all in `src/core/constraint.rs`:
   - `BoxConstraints` (interval bounds), used by `Brent` (1D),
     `ProjectedGradientDescent`, `LBFGSB`, `Trf`, `BoundedCmaEs` --- all via
     *projection* / clamping.
   - `LinearInequalityConstraints` (`A x ≤ b`, exposing `a()` / `b()`), used
     by the log-barrier `BarrierMethod` (`src/solver/barrier_method.rs`) via
     the `LogBarrier` adapter (`src/core/barrier.rs`) --- via a *barrier*, no
     projection. `BarrierMethod` is a `constrOptim`-style continuation loop
     over an inner unconstrained solver; v1 requires a strictly feasible
     start (phase 1 deferred) and an Armijo-backtracking inner line search
     (the barrier's `+∞` wall is the only feasibility guard). Backend-gated
     to nalgebra/faer because it needs `MatVec` + `MatTransposeVec` (never a
     solve) --- `Vec`/`ndarray` can join later by implementing those two ops.
   - `LinearEqualityConstraints` (`A x = b`, exposing `a()` / `b()` --- same
     *shape* as the inequality trait but a distinct *type*, so `≤` and `=`
     problems can't be confused by the type system), used by the
     `AugmentedLagrangianMethod` (`src/solver/augmented_lagrangian_method.rs`)
     via the `AugmentedLagrangian` adapter (`src/core/augmented_lagrangian.rs`)
     --- via a *quadratic penalty plus multiplier updates*
     (`L_ρ = f + λᵀc + (ρ/2)‖c‖²`, `c = A x − b`), no projection and no
     barrier. The outer loop minimizes `L_ρ` with an inner unconstrained
     solver, then updates `λ ← λ + ρ c` (or increases `ρ` when feasibility
     stalls). Unlike the barrier, `L_ρ` is finite everywhere, so it tolerates
     an **infeasible start** and any inner line search (no `+∞` wall, no
     phase 1). Convergence (`‖A x − b‖ ≤ tol`) lives in the solver's
     `terminate` hook, mirroring the barrier's gap test (tenet 3: a
     framework-level `FeasibilityTolerance` waits for a 2nd equality solver).
     Backend-gated to nalgebra/faer for the same `MatVec` + `MatTransposeVec`
     reason as the barrier.
   Nonlinear equality and nonlinear (in)equality constraint kinds are not yet
   designed.

   **Adapters must not re-implement the constraint trait they consumed.** A
   wrapper that converts a constrained problem into an unconstrained one (log
   barrier, quadratic penalty) exposes `CostFunction + Gradient` *only*. Two
   shipped adapters do exactly this: `LogBarrier<'a, P: LinearInequalityConstraints>`
   and `AugmentedLagrangian<'a, P: LinearEqualityConstraints>` both impl
   `CostFunction + Gradient` and pointedly do **not** impl the constraint
   trait they consumed --- that asymmetry is what flows the wrapped problem to
   unconstrained solvers. If a wrapper also implemented the constraint trait,
   it would route back into constrained solvers and the whole adapter model
   collapses. (Contrast `FiniteDiff`, which *adds* a capability and therefore
   *forwards* `BoxConstraints`.) Load-bearing and non-obvious; preserve it
   deliberately.

   **Do not design a `Constraint` supertrait or hierarchy until ≥2
   fundamentally different constrained solvers exist that share more than
   `lower()` / `upper()` accessors.** Three constraint kinds have now landed
   and *keep confirming* the wait rather than ending it. Each keeps feasibility
   by a different mechanism: the box family by *projection* (`ClampInPlace`),
   the linear-inequality family by a *barrier* (`MatVec`/`MatTransposeVec`),
   and the linear-equality family by a *penalty plus multipliers* (also
   `MatVec`/`MatTransposeVec`, but used to assemble `∇L_ρ`, not a barrier). The
   two linear families happen to share the same *carrier ops* (`MatVec` +
   `MatTransposeVec`), but they share no *feasibility* operation, and their
   data accessors (`a()`/`b()`) are the only common surface --- so
   `BoxConstraints`, `LinearInequalityConstraints`, and
   `LinearEqualityConstraints` stay sibling traits with no supertrait. A shared
   abstraction still waits for a constraint kind (nonlinear) that genuinely
   shares a feasibility-check or projection op. One-member (or no-shared-op
   multi-member) hierarchies are overhead with no value; designing on paper
   without a solver to validate against tends to need redoing.

   **Constraints live on the problem, never on state.** Don't put `lower` /
   `upper` on `BasicState` "for convenience". State carries iteration history;
   constraints define the problem. Bounds on state would silently un-constrain
   a problem if a different state were swapped in, and decouple constraint
   semantics from where the solver type-system enforces them. Termination
   criteria that need bounds (e.g. `ProjectedGradientTolerance`) clone them at
   construction --- that's the deliberate pattern, not a workaround.
5. **Backends are tiered; not every solver works on every backend.** The shared
   math layer (`ScaledAdd`, `NormSquared`, `NormInfinity`, ...) stays small and
   honest: only ops that every backend can implement well belong there.
   First-order and derivative-free solvers (gradient descent, Nelder-Mead, SA,
   ...) stay backend-generic via that layer. LA-heavy solvers (Newton,
   trust-region, L-BFGS, anything needing Cholesky / QR / eigensolves) may
   require a specific backend (most likely faer or nalgebra) and bound their
   param type on a richer trait that only that backend implements --- so a
   `Vec<f64>` user gets a *compile-time* error, not a runtime surprise. Same
   spirit as tenet 3: bound on the minimum capability the solver actually needs.
   Do **not** add LA ops (`Cholesky`, `Eigen`, `LinearSolve`, ...) to the shared
   math traits just to preserve symmetry across backends; if only one backend
   can implement an op well, it shouldn't be in the shared layer. Per-solver
   doc comments must include a "Backends" note listing supported param types,
   mirroring the wasm compat-note pattern below.

## Solver composition

Some solvers run another solver as a sub-step (memetic CMA-ES + LM, basin
hopping, multi-start polish, …). The composition primitive is
`run_loop(&problem, state, &mut solver, &mut criteria, max_iter)` in
`src/core/executor.rs`; the builder-style adapter `InnerExecutor<S, So>` in
`src/core/inner.rs` wraps it for the common case where an outer solver
stores a pre-configured inner and reuses it across outer iters. Three
contracts every outer solver must follow:

1. **Eval aggregation.** After each `inner.run(&problem, inner_state)`,
   the outer must roll the inner's `cost_evals()` into the outer state via
   `increment_cost_evals(...)` (and `gradient_evals()` via
   `increment_gradient_evals(...)` when both inner and outer are
   `GradientState`). Skipping this silently corrupts `MaxCostEvals`
   budgets and the public `result.cost_evals()` read. The contract is
   spelled out on `Solver::next_iter`'s rustdoc; the POC integration test
   in `crates/basin/tests/inner_executor.rs` asserts it.

2. **Inner termination criteria must be stateless across calls.** An
   `InnerExecutor` keeps its `Vec<Box<dyn TerminationCriterion<S>>>` for
   its whole lifetime and reuses it on every `run()`. This is fine for
   `MaxIter`, `*Tolerance`, and `MaxCostEvals` (no internal state, or
   only state that resets meaningfully on each call). It is **not** fine
   for `MaxTime`, whose internal `start: Option<Instant>` is set on the
   first `check` call and persists — on the second `run()` it would fire
   prematurely. Outer solvers that need per-run criteria should call
   `run_loop` directly with a fresh `Vec` per call rather than reaching
   for `InnerExecutor`.

3. **Failure routing.** `run()` returns a full `OptimizationResult`
   carrying a `TerminationReason`. Use `reason.is_failure()` (true only
   for `SolverFailed`) to decide whether to bubble the failure via the
   outer's mid-iter `Option<TerminationReason>` return. Everything else
   — `MaxIter`, the tolerance reasons, `SolverConverged` — is a "clean
   stop, outer consumes the inner's final iterate and continues". Most
   composed-solver bugs in this category are forgetting to propagate
   `SolverFailed` and silently treating an aborted inner run as a
   successful one.

**`InnerExecutor` vs `run_loop`.** Reach for `InnerExecutor` when the
outer solver wants to expose `inner_max_iter` / `inner.terminate_on(...)`
to its own users via builder methods that mirror the framework. Reach
for raw `run_loop` when the outer needs to reconstruct criteria each
call (statefulness escape hatch) or when it wants per-call criteria the
user passes through a different surface.

**Don't grow a `Composed<Outer, Inner>` type or a composition trait
hierarchy until ≥2 concrete composed solvers exist.** Same spirit as
tenet 4's "no `Constraint` supertrait until two consumers": one example
(S11's CMA + LM) doesn't reveal which abstraction wants to be shared.

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

basin's MSRV (currently **Rust 1.91.1**, pinned in `Cargo.toml` and
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
  CI exercise) must compile under the MSRV. Transitive deps that demand a newer
  rustc than CRAN ships need to be pinned, replaced, or worked around. (Prior
  pain points: `edition2024` crates back when CRAN was on 1.84 — long since
  resolved by the 1.91 bump; logged here as a reminder of the failure mode.)
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

## Repo structure: workspace (basin + basin-wasm)

basin lives at `crates/basin/`. A second member, `crates/basin-wasm/`,
provides `wasm-bindgen` JS bindings consumed by the Svelte/Tailwind
visualizer in `web/` (deployed to GitHub Pages). The workspace manifest
lives at the repo root; the lockfile is shared. `web/` is its own
node project and is **not** a Cargo workspace member.

The workspace conversion happened once a real trigger landed (web
visualizer needing wasm bindings). Until then basin was a single
crate. The historical reasoning is preserved below — argmin's splits
mostly don't apply, and we still avoid speculative crate proliferation.

- `argmin-math` isolates its per-backend-version feature explosion. **Tenet 2
  deletes this reason**---math abstractions live in `basin` behind one feature
  per backend (`nalgebra`, `ndarray`, ...).
- Observer / checkpointing crates exist to isolate heavy optional deps (slog,
  serde, TUI). basin has none of these yet.
- `spectator` is a viz binary, `argmin-py` is Python bindings --- both would be
  separate from a Rust library regardless.

Use Cargo features (e.g. `nalgebra`, `ndarray`, `serde`) for optional
integrations on `basin` itself. Add new workspace members only when there's
a concrete trigger:

- Web/wasm bindings → `basin-wasm` (already added; consumed by `web/`).
- An observer with heavy deps → `basin-observer-foo`.
- Test problems that other crates want to depend on independently →
  `basin-testfunctions`.
- Python bindings → `basin-py`.
- Core stabilizes while extras churn and need independent versioning.

A new member should pull in heavy or platform-specific deps (`wasm-bindgen`,
`pyo3`, an observer's TUI deps) that have no business in the core crate.
If the only reason is "feels tidy", keep it in `basin` behind a feature.
