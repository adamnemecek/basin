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
