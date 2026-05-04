# ROADMAP

Long-arc plan toward the two anchor solvers:

1. **Levenberg-Marquardt with box constraints** (TRF-style).
2. **CMA-ES**, eventually with LM as an inner refinement step (memetic).

`TODO.md` tracks the immediate next session's discrete items. This file
holds the phased plan, design decisions, and paper-ingestion order.
Sessions are checked off as they land so the historical reasoning stays
intact.

## Anchor design decisions (locked)

These shape every following session — change them only with a deliberate
re-discussion.

- **Sparsity is in scope from S1.** `Jacobian` (and future `Hessian`)
  carry an associated `Output` matrix type; the math layer is generic
  over it. Solvers bound on the LA ops they need, not on a specific
  matrix type.
- **Both nalgebra and faer stay on the table as LA-heavy backends.**
  Math layer splits into a vector tier (every backend) and a `linalg`
  tier (LA-heavy backends only). Per tenet 5, no backend is forced to
  implement ops it can't do well.
- **Sparse support varies by backend.** faer has first-class sparse
  (CSC, supernodal Cholesky, sparse QR/LU); nalgebra-sparse is thinner
  (especially for sparse factorizations). Some sparse problems will
  end up faer-only in practice — that's documented in per-solver
  Backends notes, not papered over.
- **`Vec<f64>` does not implement `Jacobian`.** No honest matrix type
  there. Compile-time error is the right outcome.
- **Matrix-free (`Operator` / JVP / VJP) is deferred** to whenever a
  Krylov inner solve actually wants it. Probably post-S6.

## Phase 0 — Prep

### S0. Pin termination + solver contracts in rustdoc — **done**

All eleven load-bearing invariants from the contract map are now
rustdoc-anchored. Conventions established (and inherited by S1+):

- `# Contract` heading with `**Caller must:** ...` /
  `**Implementor must:** ...` sub-bullets — single heading, two voices.
- Reserved heading vocabulary: `# Contract`, `# Panics`, `# Backends`,
  `# Examples`. No others.
- `#![warn(missing_docs)]` and
  `#![warn(rustdoc::broken_intra_doc_links)]` at crate root. `warn`
  not `deny`; promote in a later cleanup session once the missing-doc
  surface (struct fields, trivial constructors) is filled in.
- Canonical iteration ordering lives on the `executor` module; every
  other contract cross-links there.
- `# Backends` notes added to `GradientDescent` and `NelderMead` so
  S3+ has prior art per tenet 5.

## Phase 1 — Track A: toward LM with box constraints

### S1. `Residual` + `Jacobian` problem traits

- Define `Residual` (returns `R`, the residual vector type) and
  `Jacobian` with associated `Output` matrix type.
- No backend impls yet — just the trait shape, plus a couple of
  least-squares test problems on paper:
  - **Powell's singular function** (4D, classic LM benchmark).
  - **Rosenbrock-as-residuals** (`r = [10(x₂−x₁²), 1−x₁]`).
- Output: trait module + test problem stubs that compile but have no
  Jacobian impls until S2a.

### S2a. Math::linalg trait design + dual-backend dense prototype

The load-bearing session. Get this right and S3–S6 are mechanical.

- Define the minimum set of LA-ops a Gauss-Newton / LM solver bounds
  on: probably `Transpose`, `MatVec`, `MatTMatVec`, `LinearSolve<M>`.
- Prototype against **both** nalgebra dense (`DMatrix<f64>`) and faer
  dense (`Mat<f64>`). The trait must abstract over both without baking
  in either's idiom (nalgebra's method-call style vs faer's
  factorization-object style).
- **[ingest]** faer paper (Sarah Oudjedi, 2024) for design philosophy.
- **[ingest]** nalgebra-sparse user guide / examples.
- Output: `basin::math::linalg` module + `LinearSolve` impls for both
  dense backends.

**Escape hatch.** If the prototype shows the two backends genuinely
can't share a clean `LinearSolve` trait, split into
`LinearSolveNalgebra<M>` / `LinearSolveFaer<M>` and let solvers pick.
Default is unified; split only if forced.

### S3. Gauss-Newton solver

- Solves `(JᵀJ) δ = −Jᵀr`. Cholesky on `JᵀJ` is the simple path; QR on
  `J` has better conditioning — note the tradeoff in the doc, pick one.
- First LA-heavy solver: compile-time backend bound per tenet 5.
- Generic over `M: LinearSolve + ...` so sparse comes for free in S2b.
- **[ingest]** Madsen, Nielsen, Tingleff, *Methods for Non-Linear
  Least Squares Problems* (2004, IMM-DTU). Short, free, exactly the
  scope we need.
- Wire `Jacobian` impls for the test problems from S1 against both
  backends.

### S2b. Sparse `Jacobian::Output` + sparse `LinearSolve`

Slotted after S3 (so dense GN is green first), before S4 (so LM gets
sparse for free).

- Add sparse matrix types as valid `Jacobian::Output`:
  `nalgebra_sparse::CscMatrix<f64>`, `faer::sparse::SparseColMat<usize, f64>`.
- Implement `LinearSolve` for both — sparse Cholesky/QR.
- Add a sparse least-squares test problem (small linear regression as
  residuals, where `J` is sparse by construction).
- Sparse Gauss-Newton works automatically; verify with the new test
  problem on at least one backend (faer almost certainly; nalgebra
  if its sparse factorizations cooperate).

### S4. Levenberg-Marquardt (unconstrained)

- Gauss-Newton + Marquardt damping. Use **Nielsen 1999** λ-update (the
  variant MINPACK and most modern LM use; better λ recovery than
  Marquardt's original multiply/divide).
- **[ingest]** Nielsen, *Damping Parameter in Marquardt's Method*
  (1999, IMM-DTU). Short, readable, directly implementable.
- **[ingest]** MINPACK's `lmder` source/docs as the reference impl
  (public domain Fortran, well-commented).
- Inherits dense + sparse + both backends from the layer below.

### S5. Box constraints + projected gradient descent

- First constrained solver per tenet 4. Smallest possible vehicle for
  the `Constraint` trait.
- Decision to make: constraint-as-marker-trait on the problem, or as
  data on the problem? Lean: data, with a marker trait
  `BoxConstrained` that solvers can require.
- Add a constrained test problem.

### S6. LM with box bounds (TRF — Trust Region Reflective)

- The modern reference for bounded LM (SciPy's `least_squares` with
  `method='trf'`).
- **[ingest]** Branch, Coleman, Li (1999), *A Subspace, Interior, and
  Conjugate Gradient Method for Large-Scale Bound-Constrained
  Minimization Problems*.
- Cross-reference SciPy's `least_squares` source (BSD) for the
  details (initial step, scaling) that aren't in the paper.

## Phase 2 — Track B: toward CMA-ES

### S7. Wasm-safe RNG abstraction + simple stochastic solver

- Pick `rand` + a wasm-compatible seedable PRNG (probably
  `rand_chacha`). Document the seed-control story — reproducibility
  matters for stochastic tests.
- Vehicle solver: random search or a (1+1) evolution strategy. Tiny,
  but exercises stochastic state + new termination considerations
  (no monotone cost).
- New `PopulationState` (n candidates, n costs) — analogous to
  `BasicSimplexState`.

### S8. CMA-ES (vanilla)

- Second LA-heavy solver: needs eigendecomposition of the covariance.
- **[ingest]** Hansen, *The CMA Evolution Strategy: A Tutorial*
  (latest revision). Canonical reference; pseudocode is
  implementation-ready.
- Sanity-check constants against `pycma` source.
- Default to `(μ/μ_w, λ)`-CMA-ES with rank-μ + rank-1 updates,
  popsize `4 + ⌊3 ln n⌋`. Stick to tutorial defaults.

### S9. CMA-ES with bounds

- Multiple options in literature: resampling, reflection, penalty,
  BIPOP. Pick one, document the rest.
- **[ingest]** Reference for whichever bound-handling we pick — likely
  what `pycma` does, or Hansen's combustion-control paper appendix.

## Phase 3 — Convergence

### S10. Solver composition design

- Now that CMA-ES exists and LM-with-bounds exists, design how an
  outer solver invokes an inner `Executor` on a sub-problem.
- Open questions: does the outer solver own an `Executor<InnerSolver>`?
  How are inner termination criteria configured? Is the inner result
  observable from the outer state?
- Output: a short design note (probably appended to `AGENTS.md`) plus
  a minimal proof-of-concept (e.g. warm-restart GD from each
  Nelder-Mead simplex vertex — silly but tests the pattern).

### S11. CMA-ES + LM hybrid (memetic)

- Outer CMA-ES proposes candidates; inner LM refines a subset
  (often the best-k per generation).
- **[ingest]** A memetic-CMA-ES paper for the literature anchor —
  candidates: Auger et al. on LM-CMA hybrids, or whichever has the
  cleanest pseudocode rather than the highest citation count.

## Cross-cutting (slot in opportunistically)

- **Per-solver "Backends" doc note** (tenet 5) — start with S3,
  retroactive for older solvers in same session.
- **Test-problem corpus** — Picheny, three-hump camel, Powell singular,
  Brown badly-scaled. Add as needed in solver tests rather than upfront.
- **`ParamVec<F>` marker** (TODO cleanup) — fold into the session that
  introduces the third user of the bound pair, probably S3 or S6.

## Ingestion order (read papers just-in-time)

1. Before **S2a**: faer paper + nalgebra-sparse user guide.
2. Before **S3**: Madsen/Nielsen/Tingleff (2004).
3. Before **S4**: Nielsen 1999 + skim MINPACK `lmder`.
4. Before **S6**: Branch/Coleman/Li 1999 (TRF).
5. Before **S8**: Hansen CMA-ES tutorial.
6. Before **S9**: pycma bound-handling reference.
7. Before **S11**: memetic-CMA-ES paper TBD.

Use the `ingest-paper` skill before each session to pull the PDF into
`references/<name>/`.
