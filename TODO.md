# TODO

Ordered by recommended sequence --- each item is easier or better-informed once
the previous lands.

## Next up

- [ ] **CI workflow.** Minimal GitHub Actions running build/test/clippy/fmt on
      `Vec<f64>` and `--features nalgebra`, plus
      `cargo build --target wasm32-unknown-unknown` (default +
      `--features nalgebra`), so backend/wasm regressions can't slip in.

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
- [ ] **`ndarray`backend.** `nalgebra` is in; add `ndarray` behind a `ndarray`
      feature when the first user/example wants it.
- [ ] **Real bench tool (divan / Criterion) when MSRV pressure lifts.**
      Hand-rolled bench works for now; revisit when CRAN's Rust pin moves past
      1.85.

See `AGENTS.md` for the design tenets and constraints that shape these
decisions.
