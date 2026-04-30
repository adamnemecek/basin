# AGENTS.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`basin` is a Rust library crate for numerical optimization, inspired by `argmin`. It is in the earliest scaffolding stage — the public surface is a handful of empty traits and an `Executor` skeleton.

## Commands

- `cargo build` — build the library.
- `cargo test` — run tests.
- `cargo test <name>` — run a single test by name.
- `cargo clippy --all-features` — lint (matches the pre-commit hook; always pass `--all-features`).
- `cargo fmt` — format (also enforced by pre-commit).

The dev environment is provided by `devenv.nix` (loaded automatically via `direnv` from `.envrc`). It pins Rust 1.84.1 (matches `rust-version` in `Cargo.toml`) and adds the `wasm32-unknown-unknown` target plus tooling: `cargo-llvm-cov`, `cargo-flamegraph`, `cargo-audit`, `cargo-deny`, `cargo-msrv`, `samply`, `wasm-pack`, `go-task`. Pre-commit hooks run `clippy` (with `allFeatures = true`) and `rustfmt`.

## Architecture

The framework follows argmin's overall shape: a generic driver loop (`Executor`) iterates a `Solver` over a `State`, calling into user-provided `Problem` traits.

- `src/lib.rs` — public re-exports only.
- `src/core.rs` (module file) + `src/core/` — the framework:
  - `problem.rs` — traits the *user* implements: `CostFunction`, `Gradient` (more to come: `Hessian`, `Jacobian`, `Operator`).
  - `state.rs` — `State` trait carrying per-iteration data (param, iter count, gradient, etc.).
  - `solver.rs` — `Solver` trait: `next_iter(&problem, state) -> state` plus a `terminate` hook.
  - `executor.rs` — `Executor` owns problem + state + solver and drives the loop until termination.
- `src/solver.rs` — placeholder for concrete solver implementations (gradient descent, L-BFGS, Nelder-Mead, etc.).

Module convention: **newer style, no `mod.rs`** — use `src/foo.rs` for the module file and `src/foo/bar.rs` for submodules. Do not introduce `mod.rs` files.

## Design tenets (deliberate departures from argmin)

These shape API decisions and are non-obvious from the code alone:

1. **Keep argmin's vocabulary and overall shape** (`Executor`, `Solver`, `Problem` traits, `IterState`-style state). Diverge only when one of the tenets below demands it.
2. **No per-version backend feature gates.** Linear-algebra backends (nalgebra, ndarray, plain `Vec`, …) gate behind a *single* feature each (`nalgebra`, `ndarray`), pinning one version. A backend major bump becomes a basin major bump. Do **not** introduce features like `nalgebra-v0_33` / `nalgebra-v0_34` — argmin does this for compatibility, basin deliberately doesn't.
3. **Termination criteria are framework-level, not per-solver.** Generic stopping conditions (`max_iter`, `gradient_tolerance`, `param_tolerance`, `cost_tolerance`, `max_time`) belong on the `Executor` / a shared termination layer, configured uniformly across solvers. Solver-specific knobs stay on the solver. Subtlety: derivative-free solvers (Nelder-Mead, SA) have no gradient, so termination must be pluggable / opt-in based on what state and problem expose — not a fixed set of fields.

## Repo structure: single crate, not a workspace

basin is intentionally one crate, not an argmin-style multi-crate workspace. argmin's splits exist for reasons that mostly don't apply here:

- `argmin-math` isolates its per-backend-version feature explosion. **Tenet 2 deletes this reason** — math abstractions live in `basin` behind one feature per backend (`nalgebra`, `ndarray`, …).
- Observer / checkpointing crates exist to isolate heavy optional deps (slog, serde, TUI). basin has none of these yet.
- `spectator` is a viz binary, `argmin-py` is Python bindings — both would be separate from a Rust library regardless.

Use Cargo features (e.g. `nalgebra`, `ndarray`, `serde`) for optional integrations. Split into a workspace only when there's a concrete trigger:

- An observer with heavy deps → `basin-observer-foo`.
- Test problems that other crates want to depend on independently → `basin-testfunctions`.
- Python bindings → `basin-py`.
- A viz/CLI tool → separate binary crate.
- Core stabilizes while extras churn and need independent versioning.

Converting single-crate → workspace later is cheap (`git mv` the crate into `crates/basin/`, add a workspace `Cargo.toml`). Don't pre-pay the complexity.
