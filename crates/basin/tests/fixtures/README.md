# L-BFGS-B parity fixtures

`lbfgsb_rosenbrock_5d.tsv` is the iteration-wise trajectory of Nocedal's
L-BFGS-B v3.0 Fortran source on Rosenbrock 5D, used by
`tests/lbfgsb_iter_parity.rs` to verify basin's port reproduces it
within `~1e-10`.

## Format

One line per iterate. Whitespace-separated (Fortran TSV-ish):

```
iter f x(0) x(1) x(2) x(3) x(4) g(0) g(1) g(2) g(3) g(4)
```

- `iter == 0` is the post-init state (x has been projected onto the
  feasible box; cost and gradient have been evaluated). No L-BFGS-B
  step has been taken yet.
- `iter == k > 0` is the state at the end of iteration `k`, i.e.
  after `k` accepted line searches.

Numbers are printed with `es24.16` for full f64 round-trip.

## Problem setup (locked)

- `n = 5`, `m = 5`
- Bounds `[0, 5]^5` (Fortran `nbd(i) = 2`)
- Start `(-1, 2, -1, 2, -1)` (infeasible; `active` projects it to
  `(0, 2, 0, 2, 0)`)
- `factr = 0`, `pgtol = 0`, `max_iter = 30`
- Rosenbrock in basin's standard coefficient form
  (`Σ 100 (xᵢ₊₁ − xᵢ²)² + (1 − xᵢ)²`, not the rescaled `driver1.f`
  variant).

## Regenerating

The committed `.tsv` is the artifact the test reads; you only need
to follow the steps below if you've changed the fixture parameters
(start point, bounds, `max_iter`, etc.) and need a fresh trajectory.

The L-BFGS-B v3.0 source is **not vendored** in this repo —
`references/` is gitignored, by project convention (papers and
reference implementations live there locally). Fetch the BSD-3
v3.0 tarball from Nocedal's group:

```bash
mkdir -p ../../../../references/lbfgsb-v3.0
curl -L https://users.iems.northwestern.edu/~nocedal/Software/Lbfgsb.3.0.tar.gz \
  | tar -xz --strip-components=1 -C ../../../../references/lbfgsb-v3.0
```

Then build the driver and dump a fresh fixture:

```bash
gfortran -O0 -std=legacy -o lbfgsb_driver lbfgsb_driver.f \
  ../../../../references/lbfgsb-v3.0/lbfgsb.f \
  ../../../../references/lbfgsb-v3.0/linpack.f \
  ../../../../references/lbfgsb-v3.0/blas.f \
  ../../../../references/lbfgsb-v3.0/timer.f
./lbfgsb_driver > lbfgsb_rosenbrock_5d.tsv
rm lbfgsb_driver
```

This is a manual on-demand step. CI does not rebuild the fixture.
