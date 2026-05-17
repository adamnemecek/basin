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

### S1. `Residual` + `Jacobian` problem traits — **done**

`Residual` and `Jacobian` traits landed in `core::problem` with the
established `# Contract` rustdoc style. Both use `type Output` for the
produced value (matches `CostFunction::Output`); `Jacobian::Output` is
the first associated *matrix* type in the codebase. The `# Backends`
note on `Jacobian` calls out that `Vec<f64>` deliberately doesn't
implement it — no honest matrix type, compile-time error is correct.

Test-problem stubs:
- **Powell singular** in `problems/powell_singular.rs`. Raw fns + spec
  + `PowellSingular<P>` wrapper + per-backend `Residual` /
  `CostFunction` impls (Vec, nalgebra, ndarray, faer). Cost uses the
  LM ½‖r‖² convention. Tests cover the rank-deficient-at-origin
  property explicitly so it stays load-bearing for S4.
- **Rosenbrock-as-residuals** appended to `problems/rosenbrock.rs` as
  `RosenbrockResiduals<P>`, sharing `ROSENBROCK_SPEC` (one Rosenbrock
  entry in the catalog). 2D-only; `Σ rᵢ² == rosenbrock(x)` exactly,
  matching the published unscaled form rather than the LM ½ form —
  documented on the `Residual` trait contract.

`Jacobian` trait impls (per backend) deferred to S2a where the matrix
`Output` type and `linalg` ops are pinned down. Raw `_jacobian`
functions ship now with row-major layout documented, so S2a's per-
backend impls can plug them in verbatim.

### S2a. Math::linalg trait design + dual-backend dense prototype — **done**

`basin::math::linalg` lands with four traits, exactly the GN inner-step
op set:

- `MatVec<V>`: `y = A x`.
- `MatTransposeVec<V>`: `y = Aᵀ x` (forms `Jᵀ r` without
  materializing `Jᵀ`).
- `GramMatrix`: `G = Aᵀ A` (returns `Self` for both supported dense
  backends — promote to an associated `type Gram` if/when sparse
  needs a different shape in S2b).
- `LinearSolveSpd<V>`: SPD solve via Cholesky, returning
  `Result<V, LinearSolveError>`.

Impls for `nalgebra::DMatrix<f64>` (with `V = DVector<f64>`) and
`faer::Mat<f64>` (with `V = faer::Col<f64>`). Six tests per backend
covering matvec / transpose-matvec / gram identities, an SPD happy
path, and rank-deficient → `NotPositiveDefinite` failure. The escape
hatch (`LinearSolveNalgebra` / `LinearSolveFaer`) stays unused — both
backends fit the unified owned-return shape cleanly.

**Decisions made (deltas from the original brief).**

- `MatTMatVec` (composed `(AᵀA) x`) is *not* in the surface. It only
  matters for matrix-free CG-on-normal-equations, which is post-S6
  per the locked anchor decisions. Adding it now would have been
  speculative.
- A standalone `Transpose` trait was dropped. nalgebra's `.transpose()`
  allocates and faer's returns a view — the asymmetry is real, and
  `MatTransposeVec` covers every spot a GN solver actually wants
  `Aᵀ`. Promote when a third user appears.
- `LinearSolveSpd` returns owned rather than in-place. Faer prefers
  in-place idioms but the unified owned-return shape is honest on
  both backends, and an `O(n²)` allocation per Cholesky factorization
  is negligible. An `*Into` variant can land later if a hot loop
  actually wants it.
- LU / QR variants are deferred to S3, where the QR-on-`J` vs
  Cholesky-on-`JᵀJ` tradeoff for Gauss-Newton lives.
- Trait names match the existing `math/` style (`ScaledAdd`, `Dot`):
  short imperative verbs, with the SPD assumption baked into the
  trait name so future `LinearSolveLstsq` / `LinearSolveLu` can sit
  alongside.

**S1 deferred work, now wired.**

- `Jacobian for PowellSingular<DVector<f64>>` → `DMatrix<f64>`.
- `Jacobian for PowellSingular<Col<f64>>` → `Mat<f64>`.
- `Jacobian for RosenbrockResiduals<DVector<f64>>` → `DMatrix<f64>`.
- `Jacobian for RosenbrockResiduals<Col<f64>>` → `Mat<f64>`.

All four route through the existing row-major `_jacobian` raw fns —
single source of truth, no per-backend reimplementation. Tests
include a real GN-step computation on Rosenbrock at the classical
start `(-1.2, 1.0)` (verifies `MatTransposeVec` + `GramMatrix` +
`LinearSolveSpd` end-to-end against an independently solved 2×2
system: `δ ≈ [-2.2, 4.84]`) and a `Jᵀ J` rank-deficiency check on
Powell singular at the origin via Cholesky failure.

**Backend tiering, made explicit.** The `Jacobian` trait's
`# Backends` rustdoc now spells out: nalgebra and faer wired; ndarray
deliberately not (no honest `LinearSolveSpd` for `Array2<f64>` —
`ndarray-linalg` requires system BLAS/LAPACK and breaks the wasm-
default tenet); `Vec<f64>` excluded as before. Per tenet 5, missing
coverage is a compile-time error, not a runtime surprise.

**Paper ingestion.** Skipped for S2a — the API survey of pinned
faer 0.24 + nalgebra 0.33 sources gave enough signal to make the
trait-shape decision. The faer paper (Sarah Oudjedi, 2024) and the
nalgebra-sparse user guide are still queued for S2b/S3 where their
sparse-factorization details become load-bearing.

### S3. Gauss-Newton solver — **done**

`GaussNewton` lands in `solver/gauss_newton.rs` as the first solver to
exercise the S2a linalg surface end-to-end. Generic over `V` and
`M: GramMatrix + MatTransposeVec<V> + LinearSolveSpd<V>` — sparse
backends in S2b will satisfy the same bound set with no solver-side
change. The bound on `V` is `ScaledAdd<f64> + NormSquared +
NormInfinity + NegInPlace + Clone`; that's the third user of the
ScaledAdd/Clone bound pair, so the `ParamVec<F>` cleanup (TODO) is now
unblocked but stayed deferred this session.

**Cholesky-on-`JᵀJ` vs QR-on-`J`.** Picked Cholesky — the only path
S2a's `LinearSolveSpd` exposes today. Documented the tradeoff in the
solver's rustdoc: Cholesky squares the condition number and fails
noisily on rank-deficient `J`; QR is more robust but adds a second
factorization to the linalg surface. The pure-GN trust regime is
already weak enough that when QR matters, you wanted LM (S4) anyway.

**Failure path is correct.** Cholesky breakdown returns
`TerminationReason::SolverFailed` rather than a panic or a silently
ill-conditioned step. Tested directly via Powell singular at
`(1, 2, 1, 1)` — both quadratic-residual rows of `J` vanish there
(`x₁ − 2x₂ = 0` and `x₀ − x₃ = 0` simultaneously), so `J` has rank 2,
`JᵀJ` is exactly singular, and pure GN cannot recover. This is the
load-bearing "why LM" test for S4.

**Convergence test on Rosenbrock-as-residuals from `(-1.2, 1.0)`** —
GN converges in two iterations (the residual is linear in `y` at
fixed `x`, so the linear model is exact along that axis). The single-
step test asserts the post-step iterate matches the hand-computed
normal-equation solution from S2a's own end-to-end check —
`x_new = (1.0, −3.84)` — guarding against transpose / sign mistakes
that the convergence test alone would mask. Powell singular from the
classical start `(3, −1, 0, 1)` *also* converges (in 12 iterations to
cost ≈ 3·10⁻¹³), contrary to a common assumption — the rank
deficiency only bites at the optimum, by which point the iterate is
already nearly converged. The truly load-bearing failure is the
rank-deficient *non-optimal* point above.

**Solver-internal termination.** Emits
`TerminationReason::SolverConverged` when `‖Jᵀr‖_∞ ≤ tol_grad`
(Madsen/Nielsen/Tingleff eq. 3.3a, default `1e-8`). This is the
canonical NLLS first-order optimality measure; a generic
`OptimalityTolerance` criterion stayed deferred because it would need
problem access in the criterion hook (a termination-layer redesign).
The framework's `MaxIter`, `CostTolerance`, `ParamTolerance`, and
`MaxTime` work on `BasicState<V>` for free.

**State.** `BasicState<V>` reused unchanged — GN recomputes `r` and
`J` every iteration anyway (both depend on `x`), and caching the Gram
or factorization in state buys nothing without trust-region
machinery. `state.cost = ½‖r‖²` (LM convention) is derived inside the
solver from the residual it already evaluates; the bound on `P` is
`Residual + Jacobian` only, no `CostFunction`. Problems whose user-
facing `cost()` uses an unscaled `Σ rᵢ²` form (e.g. Rosenbrock-as-
residuals) will see `state.cost()` differ from
`problem.cost(state.param())` by a factor of two — both go to zero at
the optimum, so cost-based termination criteria are unaffected.

**Backends.** Wired and tested for both nalgebra
(`DVector<f64>` / `DMatrix<f64>`) and faer (`Col<f64>` / `Mat<f64>`).
`Vec<f64>` and `ndarray::Array1<f64>` produce a compile-time error
per tenet 5. Two integration test files under
`crates/basin/tests/gauss_newton_{nalgebra,faer}.rs`, four cases each
(convergence, single-step correctness, SolverConverged, rank-
deficient failure).

**Paper ingestion.** Skipped this session — the algorithm is short
enough (eq. 3.7 in Madsen/Nielsen/Tingleff is a five-line pseudocode
loop) that the relevant equations were carried directly from the
existing `linalg` module and `Jacobian` rustdoc. Madsen/Nielsen/
Tingleff stays queued for S4 where the LM λ-update needs the full
algorithmic context.

### S2b. Sparse `Jacobian::Output` + sparse `LinearSolve` — **done**

Sparse CSC types land for both backends; sparse Gauss-Newton works
end-to-end on a new `SparseLeastSquares` fixture without any solver-
side change (the S2a `M: GramMatrix + MatTransposeVec<V> +
LinearSolveSpd<V>` bound set is satisfied by the sparse impls
exactly).

**Sparse `Jacobian::Output` types wired.**

- nalgebra path: `nalgebra_sparse::CscMatrix<f64>` over
  `DVector<f64>`. Folded into the existing `nalgebra` Cargo feature
  rather than getting its own — `nalgebra-sparse 0.10` is small and
  pure-Rust, so the manifest stays one feature per backend per tenet
  2. The 0.10 pin is required by MSRV: 0.11 wants edition2024
  (Rust 1.85+).
- faer path: `faer::sparse::SparseColMat<usize, f64>` over
  `Col<f64>`. Faer's sparse module is bundled into the `faer 0.24`
  dep with no extra feature needed.

**Five linalg traits, with one honest asymmetry.**

- `MatVec`, `MatTransposeVec`, `GramMatrix`, `LinearSolveSpd` —
  implemented for both sparse types. nalgebra-sparse uses
  `spmm_csc_dense` with `Op::Transpose` for the transposed SpMV (no
  materialized `Aᵀ`); faer reuses the `SparseRowMatRef` view returned
  by `transpose()` against the same `sparse_dense_matmul` entry
  point. `GramMatrix::gram(&self) -> Self` survives sparse
  unchanged: CSC^T · CSC → CSC for both backends, so the dense-
  prototype shape from S2a didn't need to grow an associated
  `Output` type.
- `LinearSolveLstsq<V>` — new this session, mirrors
  `LinearSolveSpd<V>` (owned-return, single-method, same `# Contract`
  shape). Implemented for `SparseColMat` only — nalgebra-sparse
  doesn't ship sparse QR. The asymmetry is documented on the trait's
  `# Backends` note per tenet 5; missing coverage stays a compile-
  time error rather than a runtime surprise.

**`LinearSolveError::Singular` finally has an implementor.** S2a
introduced the variant as "reserved for future LU/QR paths"; S2b's
sparse-QR `solve_lstsq` is the first user. The QR path returns
`Singular` on faer factorization-stage errors only — sparse QR
succeeds on numerically rank-deficient inputs and produces a
solution whose null-space components are meaningless. The
`LinearSolveLstsq` rustdoc spells out this caveat (callers who need
rank-deficiency detection check the residual norm themselves).

**Test fixture: `SparseLeastSquares<M, V>`.** Linear regression
`r(x) = A · x − b` with stored design matrix and target. Unlike the
existing analytic problems in the corpus where the residual is a
closed-form function of `x`, this one carries data — the struct is
generic on `(M, V)` and per-backend `Residual` + `Jacobian` impls
pick the concrete pair. `Jacobian::jacobian` returns `self.a.clone()`
(constant `J` for linear residuals). The integration tests use a
6×3 design (`I₃` stacked on three pairwise-sum rows) with `b = A·x*`
where `x* = [1, 2, 3]`, so the closed-form least-squares minimum
has zero residual. Sparse GN converges in two iterations on both
backends (one full Newton step lands on `x*`, the next finds
`‖Jᵀr‖_∞ = 0`).

**Backends note on the test-problem corpus.** `SparseLeastSquares`
is the first problem in `problems/` whose `Vec<f64>` and `ndarray`
columns are deliberately empty rather than just deferred — those
backends have no sparse matrix type to pair with, so per tenet 5 the
absence is permanent rather than a follow-up TODO.

**QR was scoped in mid-session.** The original brief left QR
deferred (S2a and S3 both deferred QR for honest reasons). The user
chose to include sparse QR this session — faer-sparse only — to
exercise the second linalg-tier `Result`-returning solver and to
unblock future TRF / rank-deficient-LM work without re-touching the
linalg module. Dense QR stays deferred; no current solver needs it,
and adding it alongside sparse QR would have expanded surface beyond
what's load-bearing.

**Paper ingestion.** Skipped this session — the API survey of pinned
faer 0.24 sparse and nalgebra-sparse 0.10 sources gave enough signal
to make the trait-shape decisions. The faer paper (Sarah Oudjedi,
2024) and the nalgebra-sparse user guide stay queued for whenever a
future session needs supernodal-vs-simplicial Cholesky tradeoffs or
sparse QR rank-deficiency handling.

### S4. Levenberg-Marquardt (unconstrained) — **done**

`LevenbergMarquardt` lands in `solver/levenberg_marquardt.rs` as the
first damped-Newton solver in the codebase. Same `Solver<P,
BasicState<V>>` shape as Gauss-Newton; the only solver-side change is
two extra trait bounds on `M` (`AddDiagonalInPlace + MaxDiagonal`) and
one extra on `V` (`Dot`).

**Algorithm: Nielsen 1999 smooth update.** Each outer iteration solves
the damped normal equations `(JᵀJ + μI) h = −Jᵀr` via Cholesky-on-Gram,
then adapts μ from the gain ratio
`ρ = (F(x) − F(x+h)) / (L(0) − L(h))` (Nielsen eq. 2.2). On a
successful step (ρ > 0): `μ ← μ · max(1/3, 1 − (2ρ−1)³); ν ← 2`. On a
failed step (ρ ≤ 0): `μ ← μ·ν; ν ← 2ν` with `ν` initialized to 2 — the
ν-doubling lets consecutive failures escalate damping quickly. The
paper's parameter choice (β = 2, γ = 3, p = 3) is the canonical one
used in MNT 2004, Ceres, and every modern LM. Initial damping
`μ₀ = τ · max diag(J(x₀)ᵀ J(x₀))` (eq. 1.10) with default τ = 10⁻³.

**Linalg additions.** Two new traits in `linalg.rs`:

- `AddDiagonalInPlace` — `A ← A + scalar · I` in place. Implemented for
  all four backends (nalgebra dense/sparse, faer dense/sparse). The
  sparse impls require the diagonal to be in the CSC pattern; LM only
  ever calls it on a freshly computed `JᵀJ` whose diagonal is positive
  by construction (full column rank), so this is safe. Documented as a
  precondition.
- `MaxDiagonal` — `maxᵢ Aᵢᵢ` returning `f64`. Implemented for the same
  four backends. Used only at `init`-time to size μ₀; sparse backends
  treat missing-pattern diagonal entries as the implicit zero, matching
  CSC semantics.

Eight unit tests cover the new traits — two per backend (basic
diagonal-augmentation correctness + the load-bearing
"damping regularizes a singular Gram" property that motivates LM
existing).

**State-consistency choice.** LM mirrors GN's choice of leaving
`state.gradient = None`. The framework's `GradientTolerance` (`‖∇f‖₂²
≤ tol²`) is the wrong metric for NLLS — the canonical first-order
test is the ∞-norm of `Jᵀr`, which the solver checks internally and
emits `SolverConverged` on. Documented in the rustdoc `# Termination`
section. Carrying μ and ν across iterations works through `&mut self`
on `Solver::next_iter`, so no new state type was needed; `BasicState<V>`
remains the state for both GN and LM.

**Tests: 14 new (4 + 4 + 3 + 3).** Per backend (nalgebra, faer):

1. Convergence on Rosenbrock-as-residuals from `(-1.2, 1.0)` (matches
   GN's path; LM converges in ~10 iterations vs GN's 2 because the
   damping is non-zero, but reaches the same optimum cleanly).
2. **The "why LM" test.** Powell singular from `(1, 2, 1, 1)` where
   GN's Cholesky fails on the rank-deficient `JᵀJ` — LM's damping
   regularizes the system and converges to the origin in ~50 iterations.
   This is the canonical demonstration that LM strictly subsumes GN.
3. Powell singular from the classical start `(3, −1, 0, 1)` — both GN
   and LM converge here (the rank deficiency only bites at the
   optimum); LM matches GN's iteration count to within a small constant.
4. `SolverConverged` via `‖Jᵀr‖_∞ ≤ tol_grad` (parallel to GN's test).

Per backend (nalgebra-sparse, faer-sparse), reusing the
`SparseLeastSquares<M, V>` fixture from S2b:

1. Convergence on the 6×3 sparse linear-regression fixture; LM lands at
   `[1, 2, 3]` to ‖·‖_∞ < 1e-7.
2. Sparse `add_diagonal_in_place` round-trip verification (tighter
   `tol_grad`).
3. `SolverConverged` exit reason.

**Failure path.** No `SolverFailed` test for LM. The damping is
designed to prevent Cholesky failure; the only path that can hit
`SolverFailed` is the inner attempts cap (default 50 bumps, μ growing
by 2⁵⁰ ≈ 10¹⁵), which requires a pathologically constructed problem.
Out of scope for this session.

**Paper ingestion.** Nielsen 1999 was ingested via the `ingest-paper`
skill. The PDF was rotated 90° two-column landscape and pymupdf4llm
mangled it; the user manually rotated and split to 29 single-column
pages before re-running, then a stage-2 marker pass on pages 0–13
(~50 min CPU on 8 cores) recovered the literal constants in equations
(1.10), (2.2), (2.3), (2.5) that pymupdf4llm had stripped. See
`references/nielsen-1999/NOTES.md` for the algorithm map and parser
quirks.

**lmder.f** was read for understanding only (public domain Fortran).
It uses QR + Marquardt-style λ-update; basin uses Cholesky + Nielsen
update. The high-level loop shape (outer iterate → inner damping →
gain-ratio adapt) is shared.

**Existing Rust crate `levenberg-marquardt` v0.15.0** is also a port
of MINPACK lmder.f (MIT, nalgebra-only) — different algorithm from
ours, useful only as an independent reproduction target on shared
problems. Not a structural reference. Notes in `NOTES.md`.

**Out of scope.** Box constraints (S5). Scaling matrix `D` in
`(JᵀJ + μDᵀD)` (lmder uses it; Nielsen drops it). QR-on-stacked-system
LM (lands in S6 / TRF where rank-deficient `J` and box constraints make
QR materially better). Geodesic acceleration / second-order corrections
(post-S6).

### S5. Box constraints + projected gradient descent — **done**

`ProjectedGradientDescent` lands in
`solver/projected_gradient_descent.rs` as the first n-D constrained
solver in basin. The `BoxConstrained` trait already lived at
`core/constraint.rs` (introduced when 1D Brent landed) — S5 doesn't
*introduce* it, it makes the first n-D solver actually require it,
exercising tenet 4 end-to-end. Handing this solver an unconstrained
problem is now a compile error.

**Algorithm: naive PG.** Each iteration computes
`d ← −∇f(x); α ← line_search.next(...); x ← π_C(x + α d)`. The line
search runs against the *unconstrained* trial step `f(x + α d)`, not
`f(π_C(x + α d))` — Armijo guarantees on the unconstrained step do
not transfer to the projected post-step iterate. Documented as the
known limitation; SPG (Birgin–Martínez–Raydan, projected line
search) is a follow-up if the failure mode bites. At `init` the
iterate is projected onto the feasible box once, so an infeasible
starting point is silently corrected and downstream termination
checks at iter 0 see a feasible iterate.

**Vector-tier projection primitive.** `ClampInPlace` lands in
`core/math/clamp.rs` with one method:
`fn clamp_in_place(&mut self, lower: &Self, upper: &Self)`. Lives in
the vector tier (every backend can implement it well, per tenet 5),
not the linalg tier. Implemented for all four vector backends: Vec,
nalgebra (`Matrix<f64, R, C, S: StorageMut>` — broad enough to cover
DVector, DMatrix, etc.), ndarray (`ArrayBase<S: DataMut, D>`), and
faer (`Col<f64>` via `faer::zip!` triple-zip). Naming follows the
existing `NegInPlace` / `ScaledAdd` verb-form convention; the alternative
`ProjectBoxInPlace` was rejected as speculative (the only n-D
projection that's load-bearing today is component-wise).

**`ProjectedGradientTolerance` criterion.** New framework-level
criterion at `core/termination.rs`. Convergence test is the
canonical KKT-residual metric `‖x − π_C(x − ∇f(x))‖_∞ ≤ tol`, which
collapses to `‖∇f‖_∞` when no constraint is active and vanishes
exactly at constrained KKT points. A regular `GradientTolerance`
*does not* trigger at a face-active optimum (the gradient points
into the active face), which is exactly why this criterion has to
exist. Bounds are captured at construction time
(`new(lower, upper, tol)` or `from_problem(&problem, tol)`) so the
existing `TerminationCriterion::check(&mut self, &state)` signature
stays unchanged — no problem-access plumbing in the executor.
Mirrors the pattern `ParamTolerance` already established (criterion
holds its own state). New `TerminationReason::ProjectedGradientTolerance`
variant; the basin-wasm boundary string-mapping was extended to match.

**Test problem.** `BoothBoxed<P>` lifted into
`problems/booth.rs` next to the existing `Booth<P>` wrapper, sharing
the same raw `_cost` / `_gradient` free fns (single source of
truth). Carries `lower: P` / `upper: P` as data and impls
`BoxConstrained` per backend. Booth's global min `(1, 3)` lies
*outside* the `[-1, 1]²` box, so the constrained optimum is the box
corner `(1, 1)` — a load-bearing edge-active test case where the
unprojected ‖∇f‖_∞ ≈ 20 (so `GradientTolerance` would not trigger)
but the projected metric vanishes exactly.

**State-consistency choice.** PG mirrors GD's reuse of `BasicState<V>`
unchanged — projected GD recomputes cost and gradient every iteration
anyway, and there's no per-iteration scratch state that needs
caching. The framework's `MaxIter`, `ParamTolerance`, `CostTolerance`,
and `MaxTime` work on `BasicState<V>` for free; the new
`ProjectedGradientTolerance` is the only criterion that knows about
the bounds.

**Tests: 20 new (16 integration + 4 unit).** Per backend (Vec,
nalgebra, ndarray, faer), four integration cases under
`tests/projected_gradient_descent_<backend>.rs`:

1. **Slack bounds, interior min.** `BoothBoxed` with `[-5, 5]²` from
   `(0, 0)`. Converges to `(1, 3)` to `‖·‖_∞ < 1e-4`. Verifies the
   solver doesn't break the unconstrained case.
2. **Tight bounds, edge-active min.** `BoothBoxed` with `[-1, 1]²`
   from `(0, 0)`. Converges to `(1, 1)` (the corner of the box
   closest to the unconstrained `(1, 3)`).
3. **Infeasible start, init projection.** `BoothBoxed` with
   `[-1, 1]²` from `(10, 10)`. After `init`, `state.param() == (1, 1)`
   exactly. Asserted via `MaxIter(0)` so only `init` runs. Load-bearing
   test for the `init`-time projection contract — confirmed by
   reading `executor::run_loop` lines 266–268: `solver.init` runs
   *before* the loop, so even with `max_iter = 0` the projected
   iterate is the one returned.
4. **`ProjectedGradientTolerance` triggers at `(1, 1)`.** Same setup
   as case 2 with `tol = 1e-7`; asserts
   `result.reason == TerminationReason::ProjectedGradientTolerance`.

Plus four `Vec<f64>`-side unit tests for `ClampInPlace` (inside-the-box
identity, partial-clip preserves untouched components, entire-box-clip
pins to faces, equal bounds pin to the value). The other three backends
exercise the trait implicitly through the integration tests.

**Backends.** All four vector backends wired and tested. The
`# Backends` rustdoc on `ProjectedGradientDescent` mirrors
`GradientDescent` literally and adds `ClampInPlace` to the bound
list plus the `BoxConstrained` problem requirement.

**Out of scope.** SPG / projected line search (deferred to whenever
the naive-PG Armijo-after-projection failure mode actually bites in
practice). General `Constraint` supertrait or hierarchy (one
constraint kind ≠ a hierarchy worth designing). Linear (in)equality
constraints (S5 is box-only by ROADMAP scope). Active-set
bookkeeping (TRF in S6 will introduce its own).

**Paper ingestion.** Skipped this session — projected gradient
descent is short enough (steepest-descent + a clamp) that the
relevant equations come straight out of any constrained-optimization
text. Branch/Coleman/Li 1999 stays queued for S6, where the TRF
algorithm needs the full algorithmic context.

### S6. LM with box bounds (TRF — Trust Region Reflective) — **done**

`Trf` lands in `solver/trf.rs` as the first n-D box-constrained NLLS
solver in basin and the natural extension of S4's `LevenbergMarquardt`
to bounded problems. Mirrors LM's overall shape (`Solver<P,
BasicState<V>>`, Nielsen smooth μ-update, `mu`/`nu` runtime fields,
inner Cholesky-failure retry) with three additions: the Coleman-Li
affine scaling diagonals `D` and `C` from BCL eqs (i)–(iv), a
strict-interior step-back per BCL eq 2.7, and a scaled first-order
optimality termination metric.

**Algorithm: simplified BCL.** Each iteration solves
`(JᵀJ + diag(c) + μ · diag(d²)) h = −g` via Cholesky on the
SPD-by-construction damped Gram, then steps back to keep the iterate
strictly inside `(lower, upper)`:

```text
α = min(1, θ · τ_max)        # τ_max from BoxAffineScaling::max_feasible_step
x_trial = x + α · h
ρ = (Δf − ½(αh)ᵀC(αh)) / -ψ_k(α·h)   # BCL gain ratio with C correction
```

ρ > 0 accepts and shrinks μ via Nielsen smooth cubic; ρ ≤ 0 rejects and
bumps μ·ν, ν·=2 (same μ machinery as S4). Initial
`μ₀ = τ · max diag(JᵀJ + diag(c))` (BCL-aware seeding).

**Reduction to LM.** When `lower = -∞, upper = +∞` element-wise, the
BCL scaling reduces to `D = I`, `C = 0`, the step-back is a no-op, and
the algorithm becomes exactly LM with Nielsen's μ-update — same
iterates. `Trf` strictly subsumes `LevenbergMarquardt` at the
trait-bound level.

**One linalg-tier addition.** `AddDiagonalVectorInPlace<V>`: adds a
vector to the diagonal in place. The vector counterpart of the existing
scalar `AddDiagonalInPlace`, used to add `c + μ·d²` (precomputed from
`c.clone()` + `damping_vec.scaled_add(mu, &d_sq)`) in one in-place
pass. Implemented for all four matrix backends (nalgebra dense/sparse,
faer dense/sparse) — same coverage as the scalar trait. Sparse impls
require the diagonal to be in the CSC pattern (always satisfied by a
fresh Gram).

**One vector-tier trait, five methods.** `BoxAffineScaling` in
`core/math/cl_scaling.rs`:

- `compute_cl_scaling(g, lower, upper, &mut d_sq, &mut c_diag)` — fills
  `d_sq[i] = 1/|v_i|` and `c_diag[i] = |g_i|/|v_i|` (or 0 for infinite
  bounds) per BCL eqs (i)–(iv).
- `max_feasible_step(step, lower, upper) -> f64` — strict-interior
  step-back's `τ_max`.
- `cl_kkt_inf_norm(d_sq) -> f64` — the BCL first-order optimality
  metric `‖v ⊙ g‖_∞ = max_i |g_i|/d_sq_i`. **Load-bearing distinction**:
  not `‖D·g‖_∞ = max |g_i|/√|v_i|` (which blows up at face-active
  points where `|v_i| → 0`) but `‖v·g‖_∞` (which goes to 0 at any KKT
  point — interior or face-active). Matches SciPy's
  `least_squares(method='trf')` optimality measure; the wrong choice
  was caught by the tight-bounds test failing on the corner case
  before being fixed.
- `weighted_norm_squared(weights) -> f64` — `Σ self[i]² · weights[i]`,
  for the BCL-scaled predicted reduction `½(μ‖Dh‖² − h^T g)`.
- `project_strictly_inside(lower, upper, rstep)` — used at `init` to
  bring an arbitrary starting point into the *open* box (`D` is
  undefined where `v_i = 0`). Mirrors SciPy's `make_strictly_feasible`.

Per-backend impls for all four vector backends (Vec, nalgebra,
ndarray, faer) — pure element-wise, no LA dependency.

**Test fixtures.**

- `BoothResiduals<P>` and `BoothBoxedResiduals<P>` appended to
  `problems/booth.rs`. Booth's `f = (x+2y−7)² + (2x+y−5)²` factors as
  `Σ rᵢ²` with constant Jacobian `J = [[1, 2], [2, 1]]` — the linear-
  residual sibling of `RosenbrockResiduals`. With box `[-1, 1]²`, the
  unconstrained min `(1, 3)` is outside and the constrained min sits
  at the corner `(1, 1)` — load-bearing edge-active test case.
- `SparseLeastSquaresBoxed<M, V>` appended to
  `problems/sparse_least_squares.rs`: same data shape as
  `SparseLeastSquares` plus `lower`/`upper`. `BoxConstrained` impls for
  both sparse backends.

**Tests: 14 new (4 + 4 + 3 + 3).** Per dense backend (nalgebra, faer):

1. **Slack bounds, interior min.** `BoothBoxedResiduals` with
   `[-5, 5]²` from `(0, 0)` → unconstrained `(1, 3)` to ‖·‖ < 1e-5.
2. **Tight bounds, edge-active min.** `BoothBoxedResiduals` with
   `[-1, 1]²` from `(0, 0)` → corner `(1, 1)` to ‖·‖ < 1e-3 (the
   strict-interior θ < 1 keeps the iterate just inside).
3. **Infeasible-start strict-interior projection.** `BoothBoxedResiduals`
   with `[-1, 1]²` from `(10, 10)`. After `init` (asserted via
   `MaxIter(0)`), `state.param()` is *strictly* inside the box —
   tighter than PG's `≤` because `D` is undefined on the face.
4. **`SolverConverged` via `‖v ⊙ Jᵀr‖_∞ ≤ tol_grad`.**

Per sparse backend (nalgebra-sparse, faer-sparse):

1. **Slack bounds, interior min.** `SparseLeastSquaresBoxed` 6×3
   regression with `[-10, 10]³` → `[1, 2, 3]`.
2. **Bound-binding sparse case.** Upper bound on `x[2]` set to 1.5
   (below the unconstrained 3); face is active at the optimum.
3. **`SolverConverged` via the scaled-gradient metric.**

**The BoxAffineScaling termination metric was the one place the
implementation got it wrong on first cut and the tests caught it.**
Initial `scaled_inf_norm` computed `max |g_i| · √(1/|v_i|) = max
|g_i|/√|v_i|`, which blows up at face-active points. The tight-bounds
and `SolverConverged` tests failed with `MaxIter` instead of converging.
Fix: swap to `cl_kkt_inf_norm` computing `max |g_i| · |v_i| = max
|g_i|/d_sq_i`, which goes to zero at any KKT point. Matches SciPy's
`least_squares` `g_norm = max |g · v|`. Documented at length in the
trait rustdoc and the NOTES.md.

**`BoxConstrained: CostFunction` supertrait note.** `BoothBoxedResiduals`
is also the first NLLS test fixture in basin where `BoxConstrained` is
layered on `Residual + Jacobian`. The `BoxConstrained: CostFunction`
supertrait forces NLLS-with-bounds problems to also implement
`CostFunction` — fine in practice since `½‖r‖²` is trivial to compute,
but worth noting that the LM bound on `Residual + Jacobian` only is
*narrower* than the TRF bound on `Residual + Jacobian + BoxConstrained`.
Handing TRF an unconstrained-only problem is a compile error per
tenet 4.

**Failure path.** No `SolverFailed` test for TRF. The damped, scaled
Gram is SPD by construction for `μ > 0`, so Cholesky succeeds on the
first attempt. The retry path (capped at 50 bumps, `μ` growing by
2⁵⁰ ≈ 10¹⁵) is reachable only on pathologically ill-conditioned
problems.

**Backends.** All four LA-heavy backends wired and tested: nalgebra
dense (`DVector`/`DMatrix`), faer dense (`Col`/`Mat`), nalgebra-sparse
(`DVector`/`CscMatrix`), faer-sparse (`Col`/`SparseColMat`). `Vec<f64>`
and `ndarray::Array1<f64>` produce a compile error per tenet 5
(`Jacobian` isn't implemented on those). Vector-tier `BoxAffineScaling`
covers all four vector backends including Vec and ndarray (pure
element-wise, no LA story).

**Out of scope.** STIR 2D subspace (BCL FIG.5) — for large-scale where
dense Cholesky becomes expensive. Reflection technique (BCL FIG.2) —
2-3× iteration-count reduction on many-bind problems but non-trivial
implementation; deferred until a test case demands it. Explicit Δ
trust-region radius with Moré-Sorensen-style λ-adaptation (BCL FIG.6) —
LM-style μ-update reuses S4 machinery and matches SciPy's `trf_linear`.
Negative-curvature termination clause (BCL §6) — needs eigendecomposition
or Lanczos, not load-bearing before STIR.

**Paper ingestion.** BCL 1999 ingested via the `ingest-paper` skill.
Stage-2 marker pass on PDF pages 1, 3-4, 12, 15 (CPU, no `--use_llm`);
notes at `references/branch-coleman-li-1999/NOTES.md`. The marker pass
recovered FIG.1 (TIR pseudocode rasterized in the PDF and dropped by
pymupdf4llm), eqs 2.1–2.7 defining `D`/`v`/`C`, FIG.5 (STIR pseudocode),
and FIG.6 (trust-region update with literal constants). SciPy's
`least_squares` source was *not* consulted directly — the BCL paper
alone was enough since we picked LM-style μ-adaptation rather than the
SciPy 2D-subspace + reflection path. The `g_norm = max |g · v|`
optimality-metric reference came from secondary knowledge of SciPy's
TRF, documented in the `cl_kkt_inf_norm` rustdoc.

## Phase 2 — Track B: toward CMA-ES

### S7. Wasm-safe RNG abstraction + simple stochastic solver — **done**

`RandomSearch` lands in `solver/random_search.rs` as the first
stochastic, derivative-free, population-based solver in basin and the
vehicle for the new `BasicPopulationState` / `PopulationState` story.
Same `Solver<P, S>` shape as every other solver, with the RNG carried
on the solver itself (`&mut self` on `init` / `next_iter`) — same seed
in, same iterate trajectory out, on every platform basin builds for.

**Algorithm: elitist (1+λ) random search.** At `init` the solver fills
the population with λ candidates drawn component-wise uniformly from
the problem's box `[lower, upper]`, evaluates each, and sorts by
ascending cost. Each `next_iter` snapshots the elite
`(candidates[0], costs[0])`, resamples λ fresh candidates, evaluates
them, sorts the combined `λ + 1` set, and truncates back to λ. The
elite carry-over keeps `state.cost()` non-increasing across
generations, so the framework's `CostTolerance` and `ParamTolerance`
work honestly under stochastic dynamics without any termination-layer
redesign. (CMA-ES is genuinely non-monotone and the "no monotone
cost" termination story will be designed alongside it in S8 / S9.)

**RNG: `rand 0.9` + `rand_chacha 0.9`, ChaCha8Rng only.** Both pinned
to the 0.9 line: `rand 0.10` and `rand_chacha 0.10` require
edition2024 (Rust 1.85+), above basin's MSRV. `default-features =
false` on `rand` drops `std_rng` / `thread_rng` and the implicit
`getrandom` JS-feature pull-in — a `ChaCha8Rng::seed_from_u64(seed)`
works on `wasm32-unknown-unknown` with no JS shim. Verified with
`cargo build --target wasm32-unknown-unknown --all-features` after
landing the deps. Unconditional (not feature-gated): RNG is core
infrastructure, not optional; the manifest stays at one feature per
backend per tenet 2. New `core/rng.rs` module is a tiny re-export
layer (`pub use rand::{Rng, RngCore, SeedableRng}; pub use
rand_chacha::ChaCha8Rng;`).

**One vector-tier trait, one method.** `SampleUniformBox` in
`core/math/sample.rs`:

```rust
pub trait SampleUniformBox: Sized {
    fn sample_uniform_box<R: Rng + ?Sized>(
        lower: &Self, upper: &Self, rng: &mut R,
    ) -> Self;
}
```

Per-component uniform sample via `rng.random_range(lower[i]..=upper[i])`.
Inclusive-bound semantics (`Uniform::new_inclusive`) so equal bounds
deterministically pin the coordinate. Implemented for all four vector
backends (Vec, nalgebra `DVector`, ndarray `Array1`, faer `Col`) —
sampling allocates a fresh vector per call, so the trait is concrete
on each backend's specific 1D type rather than generic over
`Matrix<f64, R, C, S>` (the existing `ClampInPlace` shape doesn't
extend honestly here — there's no generic constructor across the four
backends). Standard-normal sampling (the natural next step for S8) is
*not* introduced this session — adding a `SampleStandardNormal` trait
without a caller would be speculative; CMA-ES is the right time to
design its shape.

**New state: `BasicPopulationState<V>` + `PopulationState` trait.** Same
shape as `BasicSimplexState<V>` + `SimplexState` — `candidates: Vec<V>`,
`costs: Vec<f64>`, sorted ascending by cost so `param() = &candidates[0]`
and `cost() = costs[0]`. NaN-last sort comparator lifted from
Nelder-Mead. Two constructors: `with_size(lambda)` (empty container,
solver fills it in `init` — the common case) and `from_population(Vec<V>)`
(advanced users with custom initial distributions). Capability trait
`PopulationState` exposes `candidates()` / `costs()` slices for any
future population-aware termination criterion (a `PopulationStalled`
"no improvement in N generations" check is the natural follow-up but
unjustified without a real test case).

**Tests: 18 new + 2 unit (4 + 4 + 4 + 6 + 2).** Per backend (Vec,
nalgebra, ndarray, faer):

1. **Reproducibility.** Same seed → identical trajectory across two
   independent runs. Load-bearing for the stochastic-solver contract.
2. **Convergence on `BoothBoxed`** with `[-1, 1]²` from λ = 64,
   200 iterations: lands within `0.05` of the constrained optimum
   `(1, 1)` (the unconstrained min `(1, 3)` is outside the box). Loose
   tolerance because random search converges polynomially in the
   sample budget.
3. **Elitism / monotonicity.** `state.cost()` is non-increasing across
   `next_iter`, asserted iteration-by-iteration via `Stepper::step`.
4. **Sort + length invariants** survive iteration: `candidates.len() ==
   costs.len() == λ` and costs are sorted ascending.

Vec backend additionally covers:
5. **Different seeds → different trajectories** (constant-RNG bug
   regression).
6. **`MaxIter(0)` returns the init-time elite**, with every component
   inside the box (init-time projection contract).

Plus two `core/math/vec.rs` unit tests for `SampleUniformBox`:
in-bounds correctness with a pinned coordinate (lower == upper), and
seed-reproducibility on a single sample.

**State-consistency choice.** `RandomSearch` always *clears* the
population at `init` and fills it from its own RNG, even when the
caller built the state via `BasicPopulationState::from_population(...)`.
Reason: making the trajectory depend on the constructor would silently
break reproducibility for callers using the common `with_size` path.
Callers who genuinely want a custom initial population should drive
the solver step-by-step rather than letting `init` overwrite it. This
is documented on the solver's `# Contract` rustdoc.

**Backends.** All four vector backends wired and tested. The
`# Backends` rustdoc on `RandomSearch` mirrors `GradientDescent`'s
literally — backend-generic, four concrete `V` choices behind the
respective feature flags, plus `BoxConstrained` on the problem.

**Out of scope.** `(1+1)`-ES with Gaussian perturbation (would need
`rand_distr::StandardNormal`, deferred to S8). Non-elitist random
search and the new termination criteria it would force (`PopulationStalled`,
diversity-based checks) — bigger redesign than warranted before
CMA-ES actually needs them. `serde` integration for checkpointing
seeded RNGs (no caller yet). Generalizing `BasicState<P, F>` over `F:
Float` (the trigger TODO.md called out for the first stochastic
solver) — `f64` everywhere is still honest, no f32 user has appeared,
and the bound-boilerplate cost of preemptive generality outweighs
deferring per the *Provisional choices* section of `AGENTS.md`.

**Paper ingestion.** Skipped — random search is a half-page algorithm
with no canonical paper. `rand` and `rand_chacha` API surveys (via
docs.rs) gave enough signal to pin versions and trait shape. The
Hansen CMA-ES tutorial stays queued for S8 where the algorithm and
its constants need a literal reference.

### S8. CMA-ES (vanilla) — **done**

`CmaEs` lands in `solver/cma_es.rs` as the second LA-heavy solver in
basin and the first stochastic LA-heavy solver. Implements
`(µ/µ_W, λ)`-CMA-ES with negative weights (aCMA-ES) per Hansen 2016
Figure 6 / Table 1 — same `Solver<P, BasicPopulationState<V>>` shape
as `RandomSearch`, with five new math ops underneath.

**Algorithm: Hansen 2016 with the 2016 negative-weights setting.**
Each iteration: rebuild `y_{i:λ} = (x_{i:λ} − m)/σ` from the previous
generation's sorted candidates, compute the recombination mean
`⟨y⟩_w = Σ w_i y_{i:λ}`, advance the mean (eq. 42), update the
conjugate path `p_σ` (eq. 43) and step-size `σ` (eq. 44), test the
`h_σ` heuristic (`(1−c_σ)^{2(g+1)}` denominator, matching pycma),
update the cumulation path `p_c` (eq. 45), apply the rank-1 +
rank-µ covariance update with negative-weight rescaling (eqs. 46–47),
re-eigendecompose `C` into `(B, d²)`, then sample λ fresh candidates
`x_k = m + σ B (d ⊙ z_k)` for the next generation. Init samples the
first generation with `C = I` (so `B D = I`, no eigendecomposition
needed at iter 0).

**Five new math primitives.** Vector tier: `SampleStandardNormal`
(per-component `N(0,1)` via `rand_distr::StandardNormal`),
`ComponentMulAssign` (`y[i] *= x[i]`), `ScaleInPlace` (`y *= s`,
borrow-checker-honest counterpart of `ScaledAdd`), and `VectorLen`
(get `n` from a template — used to derive constants from
`initial_mean.vec_len()` so the user doesn't pass `n` separately).
Linalg tier: `SymmetricEigen<V>::try_eigh` returning `(B, λ)` with
`A ≈ B diag(λ) Bᵀ`, `MatrixIdentity::identity(n)` for the `C = I`
init, and `RankOneUpdate<V>` for the `C += α v vᵀ` accumulator.
Method named `try_eigh` rather than `symmetric_eigen` to avoid
colliding with nalgebra's inherent `Matrix::symmetric_eigen` (which
consumes `self` and returns a `SymmetricEigen` struct, not a
`Result`).

**Default parameters per Table 1.** Resolved at init time from the
problem dimension `n = initial_mean.vec_len()` and the optional
`λ_override` (default `4 + ⌊3 ln n⌋`, eq. 48). Recombination weights
follow rows (49)–(53): preliminary `w_i' = ln((λ+1)/2) − ln i`,
positive weights normalized to sum to 1, negative weights bounded by
`min(α_µ, α_µeff, α_pos_def) / Σ|w_j'|−`. The "apparent circular
dependency" between `c_1`, `c_µ`, `α_µ`, `µ_eff`, and the negative
weights is broken (per Hansen Appendix A) by computing `µ_eff` once
from raw `w_i'` (it's invariant under positive-weight rescaling),
deriving `c_1`, `c_µ` from it, then computing the final weights.

**State shape: `BasicPopulationState<V>` reused unchanged from S7.**
The user-visible iterate is the population's best (sorted ascending).
Solver-internal mutable state — `m`, `σ`, `C`, `B`, `d`, `d_inv`,
`p_σ`, `p_c`, `weights`, all the constants, RNG, generation
counter — lives on the solver via `&mut self`. The S7 contract about
`init` resampling regardless of caller-provided population carries
over: passing a custom `BasicPopulationState::from_population(...)`
is silently overwritten by the solver's seeded sampler at `init`,
keeping reproducibility honest.

**Termination.** Solver-internal **TolX** via `Solver::terminate`:
`σ · max_i d_i < tol_x` emits `SolverConverged`. Default
`tol_x = 1e−12 · initial_sigma` per Hansen Appendix B.3. Other
CMA-ES termination heuristics (NoEffectAxis, NoEffectCoord,
ConditionCov, EqualFunValues, Stagnation, TolXUp, TolFun) are out
of scope — most need state introspection that would force a richer
state shape, or are restart-machinery (S11) concerns.

**Tests: 11 integration + 5 unit (6 + 5 + 2 × {nalgebra, faer}).**
Per LA-heavy backend (nalgebra `(DVector, DMatrix)`, faer
`(Col, Mat)`):

1. **Reproducibility.** Same seed → identical trajectory.
2. **Convergence on Sphere 5-D from `(1, 1, 1, 1, 1)`.** 80 iters at
   default λ ≈ 8 → cost < `1e−6`.
3. **Convergence on Rosenbrock 2-D from `(−1, 1)`.** 800 iters at
   default λ = 6 → iterate within `1e−3` of `(1, 1)`. The canonical
   non-convex banana-valley test where CMA-ES is supposed to shine.
4. **`SolverConverged` via TolX.** 2000-iter cap on Sphere 3-D with
   default `tol_x = 1e−12 · σ₀`; the run terminates by TolX, not by
   `MaxIter`.
5. **PopulationState invariants.** `candidates` and `costs` stay
   length-λ and sorted ascending across iterations.

Plus, on nalgebra only:
6. **Different seeds → different trajectories** (constant-RNG bug
   regression).

Plus per backend, two new linalg unit tests for the matrix tier:
matrix-identity correctness, rank-1 outer-product update, and
symmetric eigendecomposition recomposition `B diag(λ) Bᵀ ≈ A`
to `1e−10`.

**Backends.** All four LA-heavy combinations would in principle work
— but only nalgebra and faer carry the full
`SymmetricEigen + MatrixIdentity + RankOneUpdate + ScaleInPlace +
MatVec + MatTransposeVec` set, so those are the two wired and tested.
`Vec<f64>` and `ndarray::Array1<f64>` produce a compile-time error
per tenet 5 (no honest matrix type / no pure-Rust eigendecomposition
on `Array2<f64>`). The vector-tier additions (`SampleStandardNormal`,
`ComponentMulAssign`, `ScaleInPlace`, `VectorLen`) cover all four
vector backends including Vec and ndarray, anticipating future
stochastic solvers that don't need a covariance.

**Out of scope.** Bounded variant (S9). Eigendecomposition refresh
every `max(1, ⌊1/(10n(c_1+c_µ))⌋)` iterations (Hansen Appendix B.2)
— we refresh every iteration for simplicity. BIPOP / IPOP restarts
(S11). Constraint handling (S9). Active-CMA decay tightening when
C is ill-conditioned (pycma-specific, not in the tutorial). Adaptive
σ-stalling termination (TolXUp, EqualFunValues, Stagnation).

**Paper ingestion.** Hansen 2016 ingested via `ingest-paper`. The
fast `pymupdf4llm` pass mangled equations badly (eq. 1–60 came out
as `aVar[ˆτyx[z][2][]]`-style bracket noise), but **direct
`pymupdf.get_text()`** on the algorithm-summary pages (28–31) and
the step-size pages (16–21) extracted the math cleanly — no marker
pass needed. The marker recipe was attempted but blocked by an
unrelated `pycma` dep in `tools/pyproject.toml` that uv refuses to
resolve; the pymupdf path was sufficient. Notes at
`references/hansen-2016/NOTES.md`.

**Reference impls.** pycma (BSD-3-Clause) was consulted for the
`h_σ` denominator convention (`(2*countiter + 2)`) and the
weights-bounding flow ordering, but no code was ported — basin's
implementation is from Hansen 2016 directly. The MATLAB source in
Appendix C of the tutorial is part of the paper and was likewise
not copied.

### S9. CMA-ES with bounds — **done**

`BoundedCmaEs` lands in `solver/bounded_cma_es.rs` as the box-
constrained sibling of S8's `CmaEs`. Same `Solver<P,
BasicPopulationState<V>>` shape, same nalgebra+faer backend coverage,
same TolX termination — the only addition is Hansen's adaptive
quadratic boundary penalty (pycma's `BoundPenalty`, the long-time
default) woven through sample evaluation and the top of `next_iter`.

**Algorithm: adaptive quadratic penalty (Hansen / pycma).** Per
generation: sample `x_k ~ N(m, σ² C)` exactly as in `CmaEs`, repair
`x_k_rep = clamp(x_k, l, u)`, evaluate `f_raw = f(x_k_rep)` at the
repaired point, add penalty `pen = (1/n) Σ_i γ_i (x_k[i] − x_k_rep[i])²`,
sort the population by `f_raw + pen`. The **un-repaired** sample
enters recombination so the covariance learns "don't go that way";
the **penalized** cost is what ranks the population. After each
m / σ / C update (and before the next sample loop), γ is adapted from
(a) the IQR of recent raw fitness values, normalized by the average
per-axis variance, and (b) the per-coordinate violation of the new
mean, measured in σ-units. Both pieces are scale-invariant under σ,
so the penalty self-tunes without user knobs. The initial mean is
projected onto `[lower, upper]` once at iter 0 (mirrors PGD's iter-0
projection — pycma doesn't do this, basin does for ergonomics).

**Why this strategy and not the others.** Resampling distorts the
implicit sampling distribution and rejection rates explode when the
optimum is on a face. Reflection / clipping is unprincipled — clipping
puts a delta on the distribution that fights covariance adaptation;
reflection aliases multimodally near corners. Smooth-boundary
transformations (pycma's `BoundTransform`) distort the optimization
landscape near active bounds. Adaptive penalty is the only one with
a serious self-adapting reference implementation (pycma) and matches
CMA-ES's "no extra knobs" ethos. **BIPOP** is sometimes lumped with
these but is a population-restart scheme, orthogonal to bound
handling — reserved for S11.

**Tenet 4: constraints on the problem, not state.** The user impls
`CostFunction + BoxConstrained` on the same problem type, exactly as
with `ProjectedGradientDescent` / `Brent` / `Trf`. No new constraint
trait, no constraint state on `BasicPopulationState` — γ and the
fitness-IQR history live solver-internally on `Working<V, M>`, since
they're adaptation state, not constraint definition.

**State storage convention.** `BasicPopulationState.costs` carries
**penalized** fitness values (so the `PopulationState`
sorted-ascending invariant remains coherent for tenet-3 termination
criteria). Raw fitness is held in a sidecar `Vec<f64>` on `Working`
in *sample order* (not sorted) — γ-update reads it as a flat bag of
values for the IQR estimator, so order is irrelevant.

**One new math primitive.** `MatDiagonal<V>::diagonal(&self) -> V`
extracts `diag(C)` as a vector for the σ²·diag(C) per-axis
variances the γ-update needs. Wired for `nalgebra::DMatrix<f64>`
(over `DVector<f64>`) and `faer::Mat<f64>` (over `Col<f64>`). Same
backend coverage as `SymmetricEigen` — every dense matrix backend
that supports CMA-ES's eigendecomposition gets `MatDiagonal` for
free as a few-line impl. The other math traits (`SampleStandardNormal`,
`ScaleInPlace`, `ComponentMulAssign`, `RankOneUpdate`,
`SymmetricEigen`, `MatrixIdentity`, `MatVec`, `MatTransposeVec`,
`NormSquared`, `ScaledAdd`, `VectorLen`) are reused unchanged from
S8; `ClampInPlace` is reused from PGD.

**Termination.** Solver-internal **TolX** carries over from S8
unchanged: `σ · max_i d_i < tol_x`, default `tol_x = 1e−12 · σ_init`.
No new criteria — feasibility is enforced by construction at the
sample-evaluation site (every evaluated point is in
`[lower, upper]`), and the framework's `MaxIter` / `MaxCostEvals`
work against `BasicPopulationState` without modification.

**Code reuse from S8.** Three helpers in `solver/cma_es.rs` were
bumped to `pub(crate)` and shared verbatim:
`expected_norm_n01(n)`, `compute_weights(n, λ, c_1, c_µ)`, and
`sort_population_ascending(candidates, costs)`. The CMA core itself
(sampling, recombination, σ adaptation, rank-1 + rank-µ C update,
eigendecomposition, TolX termination) is duplicated in
`bounded_cma_es.rs` rather than refactored into a shared inner loop —
the bound-handling additions thread through enough of the iter
sequence (sample → repair → evaluate → penalize, plus γ-update at
the top of `next_iter`) that a shared inner loop would expose more
internals than the duplication costs. Same call as
`GradientDescent` vs. `ProjectedGradientDescent`.

**Tests: 12 integration (6 + 6 × {nalgebra, faer}).** Per LA-heavy
backend:

1. **Reproducibility.** Same seed → identical trajectory.
2. **Slack bounds recover unconstrained minimum.** Booth on
   `[−5, 5]²` → `(1, 3)` within `1e−2` after 400 iters. Verifies
   the penalty machinery doesn't get in the way when bounds are
   inactive.
3. **Tight bounds converge to box corner.** Booth on `[−1, 1]²`
   → `(1, 1)` within `1e−2` after 800 iters. Verifies the
   adaptive penalty steers the search distribution toward an
   active corner without the rank-µ update going sideways.
4. **Infeasible initial mean converges.** Start at `(10, 10)` with
   bounds `[−1, 1]²` → `(1, 1)` within `1e−2` after 800 iters.
   Tests the iter-0 mean projection plus penalty co-recovery.
5. **`SolverConverged` via TolX.** 2000-iter cap on slack-bounded
   Booth; the run terminates by TolX, not `MaxIter`.
6. **PopulationState invariants.** `candidates` and `costs` stay
   length-λ and sorted ascending across iterations (where `costs`
   holds penalized values).

All 12 tests pass on both backends.

**Backends.** Same coverage and reasoning as S8: nalgebra
(`DVector<f64>` / `DMatrix<f64>`) and faer (`Col<f64>` / `Mat<f64>`).
`Vec<f64>` and `ndarray` produce a compile-time error per tenet 5
(no honest matrix type / no pure-Rust `SymmetricEigen` on
`Array2<f64>`).

**Out of scope.** Linear-equality constraints (need a different
solver entirely — projection onto an affine set, not a box). General
nonlinear constraints (waiting for the second concrete solver kind
per AGENTS.md tenet 4 — pycma's `AugmentedLagrangian` is the
candidate reference but the *abstraction* it should bind to isn't
designed yet). The other CMA-ES termination heuristics
(NoEffectAxis, NoEffectCoord, ConditionCov, EqualFunValues,
Stagnation, TolXUp, TolFun) — same OOS as S8, those are restart-
machinery / state-introspection concerns. Pycma's `countiter == 2`
γ re-init path — minor empirical refinement, skipped for v1; will
revisit if a head-to-head pycma comparison shows divergence.
`BoundTransform` (smooth-boundary alternative) — explicitly chose
penalty over transform, see S9 doc-comment for rationale.

**Reference ingestion.** pycma r4.4.4 `cma/boundary_handler.py`
(BSD-3-Clause) vendored verbatim under
`references/pycma-bound-handling/`, alongside a NOTES.md mapping
the algorithm to basin's port (γ initialization, the IQR /
σ²-normalization, the `tanh(edist/3)/2` damping factor, the
`5·dfit` decay cap, the `20 + 3n/λ` history depth). The
`ingest-paper` skill's PDF-shaped Stage-1 dump didn't apply
(the reference is code, not a PDF) — manual NOTES.md is the
bridge. Hansen et al. 2009 *A Method for Handling Uncertainty…*
(IEEE TEC) is the paper anchor and is referenced in NOTES.md;
no PDF was needed because the pycma source is the operative
authority.

**Reference impls.** pycma `BoundPenalty.update` was studied
line-by-line and basin's `update_gamma` mirrors its active branch
(boundary_handler.py:731). The legacy elif/else branches are dead
code in pycma and were skipped. The penalty `/n` divisor (also
pycma-specific, not in Hansen 2009) is preserved because pycma's
γ-init `2·dfit` is calibrated against it. No code was ported; the
implementation is rewritten from the algorithm description in
NOTES.md.

## Phase 3 — Convergence

### S10. Solver composition design — **done**

The composition primitive `run_loop(&problem, state, &mut solver, &mut
criteria, max_iter)` was already in `core/executor.rs` (pub-exported,
zero in-tree callers); S10 is its first real exercise. Lands as a thin
builder-style adapter `InnerExecutor<S, So>` in `core/inner.rs` that
wraps `run_loop`, owning the inner solver, its termination criteria,
and `max_iter`. Outer solvers store one as a field and call
`inner.run(&problem, state)` once per outer iter.

**Open questions, resolved.**
- *Does the outer own an `Executor<InnerSolver>`?* No — `Executor` owns
  its problem; composed inners borrow the outer's `&P` from
  `next_iter`. `InnerExecutor` deliberately does not own the problem;
  it's supplied at `.run()` time.
- *How are inner termination criteria configured?* On the
  `InnerExecutor` builder, mirroring `Executor`'s `terminate_on`. Reused
  across calls — see contract 2 below.
- *Is the inner result observable from the outer state?* The
  `OptimizationResult` is returned to the outer's `next_iter`, which
  rolls cost evals (and gradient evals when applicable) into the outer
  state. Inner termination reasons aren't surfaced as state — they're
  classified at the call site via `TerminationReason::is_failure()`.

**Three contracts** documented in `AGENTS.md` "Solver composition":
1. **Eval aggregation.** Inner `cost_evals()` rolls into the outer state
   via `increment_cost_evals(...)`; same for gradient evals when both
   inner and outer are `GradientState`. Codified as a clause on
   `Solver::next_iter`'s rustdoc.
2. **Inner criteria must be stateless across calls.** `MaxTime` is the
   load-bearing exception (its internal `start: Option<Instant>` is set
   on first check and persists). Outer solvers needing per-run criteria
   call `run_loop` directly with a fresh `Vec`.
3. **Failure routing.** `TerminationReason::is_failure()` (true only
   for `SolverFailed`) classifies whether the outer should bubble via
   mid-iter `Option<TerminationReason>`. Everything else
   (`MaxIter`, `*Tolerance`, `SolverConverged`) is "clean stop, outer
   consumes and continues".

**POC.** `crates/basin/tests/inner_executor.rs` defines a private
`PerVertexRefine<G>` outer solver over a custom `MultiStartState`
(rather than `BasicSimplexState`, whose vertex/cost fields are
`pub(crate)` and unreachable from integration tests — using a custom
state proves composition works through the public `State` / `Solver`
traits alone). Three tests on Booth: convergence to `(1, 3)`, cost-eval
aggregation, and `SolverFailed` bubbling via a fake `AlwaysFails` inner
solver.

**Deliberately not done.** No `Composed<Outer, Inner>` adapter / trait
hierarchy — same spirit as tenet 4's "no `Constraint` supertrait until
≥2 consumers." S11's CMA-ES + injected local search is the only known
concrete consumer; one example doesn't reveal which abstraction wants
to be shared. If MA-LS-Chains lands later (separate S-number,
SSGA outer + CMA inner) that's a second consumer and the shared
abstraction question can be revisited honestly.

**Backends.** `InnerExecutor` is backend-agnostic (parameterised on the
inner state and solver types); the POC test uses `Vec<f64>`. No new
math primitives.

### S11. CMA-ES + local-search polish (memetic, via injection) — **done**

`CmaInject` lands in `solver/cma_inject.rs` wrapping `CmaEs` plus an
`InnerExecutor<BasicSimplexState<V>, NelderMead>` for the inner polish.
Each `next_iter` runs vanilla CMA-ES (sample λ candidates, evaluate,
update m/σ/C), refines the best `k` (default 1) via inner NM, clips
each refined point's normalised step `y_i = (x_i − m)/σ` in Mahalanobis
distance — `y_i ← min(1, c_y/‖C^{-1/2}y_i‖) × y_i` with default
`c_y = √n + 2n/(n+2)` (Hansen 2011 Table 1) — re-evaluates the cost at
the clipped point, and re-sorts. The CMA update on the next iter reads
the modified candidates through the standard equations unchanged
(Hansen 2011 §3: *"All update equations starting from (5) are
formulated relative to the original sample distribution. This means we
are, in principle, free to change the distribution before each
iteration step."*). Lamarckian; no Baldwinian mode.

**Implementation notes.**
- **First cut specialises to NelderMead inner.** The struct is
  `CmaInject<V, M>` with the inner solver hard-wired to
  `InnerExecutor<BasicSimplexState<V>, NelderMead>`. Genericity over
  the inner type (`CmaInject<I: Solver>`) is deferred to §S13 (LM and
  L-BFGS-B inners) — designing the "seed inner state from a
  candidate" abstraction against one consumer is premature, same
  spirit as tenet 4's "no `Constraint` supertrait until ≥2 consumers."
  The long-term anchor pair (CMA-ES + LM, per goal #2 at top of file)
  is what motivates keeping the inner-generic refactor in scope.
- **Re-evaluation after clipping is mandatory.** Both inner refinement
  and Mahalanobis clipping move the point in original-`x` space, so
  the cost field has to match the geometry — otherwise the next CMA
  update ranks the wrong point. The extra `+1` eval per refined
  candidate is unavoidable and is documented in
  `solver/cma_inject.rs` next to the call.
- **Inner simplex seed uses absolute σ-scaled edges.** Vertices are
  `x_i + scale·σ·e_j` for `j = 1..=n` (scale defaults to `1.0`), not
  the FMINSEARCH/SciPy relative 5% step. Absolute scaling tracks
  CMA's shrinking distribution so the inner stops exploring outside
  the current generation's spread as σ decays.
- **`Working` bumped to `pub(crate)` to expose `m`, `σ`, `B`,
  `D^{-1}`, `n` to the sibling solver module.** No public API
  change; mirrors the existing `pub(crate)` exposure of helpers
  like `sort_population_ascending` and `compute_weights` that
  `bounded_cma_es.rs` already consumes.

**Skipped in this PR.** Injection at iter 0 (Hansen's preliminary
experiments inject from iter 1 onward; cheap to add later). Mean-shift
injection mode (Hansen 2011 eq. 5 right-hand branch + the discussion
at §3) — not needed for per-point Lamarckian injection. The
`SolverFailed`-bubbling integration test is deferred to the
inner-generic follow-up; the inner-bubble path is short and visually
reviewable, and `tests/inner_executor.rs` exercises the contract
end-to-end against a fake `AlwaysFails` inner.

**Rationale carried over from the pre-implementation plan.**
- **Why injection instead of Melo & Iacca's sequential shape.** Both
  are paper-anchored. Per-generation refinement extracts more value
  from the inner solver and is the foundational primitive Hansen
  designed for memetic CMA-ES; the sequential pattern is just the
  `k = λ` / "polish once at the end" degenerate case. We get the
  general primitive for the same conceptual cost.
- **Inner choice is paper-anchored — Hansen 2011 anchors the
  injection *mechanism* for arbitrary inners.** From the paper's use
  cases (§1, line 232 of `RR-7748.tex`): "An improved solution from a
  local search started from a CMA-ES sample (Lamarckian) — explicitly
  named as the memetic-algorithm use case." Hansen does not advocate
  for a specific inner solver — it makes CMA-ES injection-tolerant so
  any local solver can plug in. Melo & Iacca 2014 supplies *empirical*
  justification for one specific choice (Nelder-Mead won 5/9 of their
  constrained problems against {BOBYQA, L-BFGS-B}); that's why we
  start there, but it does not constrain what *other* inners are
  legitimate under the same mechanism. Additional inners (LM,
  L-BFGS-B) land in §S13.

**[ingested]** `references/hansen-2011/` (Hansen 2011 INRIA RR-7748)
— the injection protocol + clipping; `references/melo-iacca-2014/`
(Melo & Iacca 2014, IEEE SSCI) — empirical justification for picking
Nelder-Mead as the first inner. Both have NOTES.md.

### S12. MA-LS-Chains (memetic with persistent local-search state) ✓

**Status.** Shipped as `Ssga` (`crates/basin/src/solver/ssga.rs`,
standalone — BLX-α + NAM + BGA + replace-worst, backend-generic incl.
`Vec<f64>`) and `MaLsChCma` (`crates/basin/src/solver/ma_ls_ch_cma.rs`,
composes SSGA outer + per-individual CMA-ES inner; nalgebra/faer
backends inheriting CMA-ES's matrix requirement). `CmaEs::init` /
`BoundedCmaEs::init` made idempotent so a paused CMA-ES re-entered via
`run_loop` keeps its evolution state. `MaLsChState<V, M>` is
solver-private (not promoted to `core::state`) per tenet 4 — one
consumer so far.

- A *different* memetic shape that's worth its own solver, not a
  variant of S11: **steady-state GA outer** (BLX-α crossover, negative
  assortative mating, replace-worst), with an inner local search that
  **stores its state per individual** so re-selecting an individual
  resumes its LS chain rather than restarting. The canonical inner is
  CMA-ES (giving MA-LSCh-CMA), with Solis-Wets / Subgrouping
  Solis-Wets as alternatives for high-dimensional problems.
- **Why a separate solver, not an S11 mode.** The chains mechanism
  *requires* persistent individual identity, which SSGA has (population
  carries across generations) and CMA-ES doesn't (the population is
  resampled each generation). Trying to graft chains onto S11 means
  inventing an elite archive — that's research. Doing it as its own
  outer (SSGA) is the honest path and matches the published algorithm.
- **Once S12 lands, revisit S10's "no `Composed` abstraction" note.**
  S11 + S12 are the two concrete composed solvers tenet 4-style logic
  needs to design the shared abstraction (if any) honestly. Open for
  the next session.
- **[ingested]** `references/bergmeir-2016/` (JSS paper on the
  Rmalschains R package) — cleanest pseudocode and parameter table.
  `references/molina-2010/` (the underlying EC-journal paper) —
  ingested as part of this session; provides BGA formula, exact S_LS
  fallback rule, and the full per-individual CMA-ES state list
  (m, σ, C, B, BD, D, p_c, p_σ).

### S13. CMA-ES injection — additional inners (LM, L-BFGS-B)

Direct follow-up on §S11. Same injection mechanism (Hansen 2011 eq. 4,
clip `y_i ← min(1, c_y/‖C^{-1/2}y_i‖) × y_i` and plug back into the
standard CMA update); only the inner solver and its state-seeding
change. Hansen 2011's framework is inner-agnostic, so this is
*mechanism reuse*, not new algorithm design.

**Refactor first: inner-generic `CmaInject`.** S11 specialised to
`CmaInject<V, M>` with NelderMead + `BasicSimplexState<V>` hard-wired
because designing the seed-inner-state abstraction against one
consumer is premature. With two more consumers in scope, the right
shape becomes visible. Plan: `CmaInject<I, S, V, M>` parametric on
inner solver `I` and inner state `S`, plus a seeder
`Box<dyn Fn(&V, f64) -> S>` builder method (the `f64` is the current
`σ`, so seeders that scale with the CMA distribution — like the
σ-scaled simplex from S11 — port directly). Move the NM-specific
seed-simplex constructor behind a free function /
`CmaInject::with_nelder_mead(...)` convenience builder so existing
S11 callers keep their ergonomics.

**S13a. CMA-ES + LM (the long-term anchor — top-of-file goal #2).**
LM is intrinsically a least-squares solver: it requires
`Residual + Jacobian` on the problem, and its inner cost is
`½‖r‖²` (S4 convention). So the *outer* `CmaInject<LM, …>` variant
requires the outer problem to be a residual problem too — the user
hands in a `P: CostFunction + Residual + Jacobian`, CMA-ES uses the
scalar cost, and the inner LM uses the residual + jacobian. Eval
aggregation extends to `gradient_evals` and (newly) `jacobian_evals`
if/when LM exposes that counter; per the existing composition
contracts in AGENTS.md, both inner and outer state must be
`GradientState`-ish for the gradient-eval roll-up to apply, so
either we extend the contract to a `JacobianState` tier or we just
roll jacobian evals into `cost_evals` honestly. Inner-state seeder
is trivial: `|x_i, _sigma| BasicState::new(x_i.clone())`.

Tests: a nonlinear-least-squares testbed where CMA's global stage
gets near a basin and LM polishes to high precision — e.g.
Rosenbrock-as-residuals (`r(x) = [10·(x₂−x₁²), 1−x₁]`, basin's
existing `RosenbrockResiduals`) or another residual problem from the
S3/S4/S6 test corpus. The point is to validate that the composition
delivers higher-precision basin convergence than vanilla CMA-ES at
the same outer-iter budget — LM's superlinear convergence inside the
basin is what justifies the pair.

**S13b. CMA-ES + L-BFGS-B.** Melo & Iacca 2014's other tested inner.
Blocked on L-BFGS-B itself landing (S14, see below). Once L-BFGS-B
lands and the inner-generic refactor from S13a is in, this is a
small follow-up: outer `P: CostFunction + Gradient`, inner state
`LbfgsState<V>`, seeder
`|x_i, _sigma| LbfgsState::new(x_i.clone(), m)` (analogous to LM).
Box bounds on the outer become interesting — L-BFGS-B is itself a
bounded solver, so the outer can be `BoxConstrained` and the bound
flows to both `BoundedCmaEs` *and* the inner L-BFGS-B. Probably want
a `BoundedCmaInject` variant to mirror the S8 → S9 pattern.

**Ingestion.** No new papers for S13a — Hansen 2011 already anchors
the mechanism; S4's MNT + Nielsen anchor LM. S13b's L-BFGS-B
reference is ingested in S14 (`references/lbfgsb-v3.0/`).

### S14. L-BFGS-B (faithful port of Nocedal v3.0)

Box-constrained quasi-Newton, anchored on Byrd–Lu–Nocedal 1995 and
Zhu–Byrd–Lu–Nocedal 1997 (the bundled `algorithm.pdf` and `code.pdf`
in `references/lbfgsb-v3.0/`). Targets iteration-wise parity with
the Fortran reference (≤ ~1e-11 ‖x_k − x_k_ref‖_∞ per iter on a
5D-Rosenbrock-with-bounds fixture). Backends: nalgebra + faer
(LA-tier per tenet 5; needs `LinearSolveSpd<V>` for the compact-form
middle-matrix solve). Plan checked in at
`~/.claude/plans/i-want-to-add-ticklish-breeze.md`.

Staged delivery; first five stages have landed:

- **S14.1 [done]** Ingest Fortran v3.0 source (`references/lbfgsb-v3.0/`)
  with NOTES.md mapping every subroutine we port to its upstream
  line range.
- **S14.2 [done]** Port Moré–Thuente line search
  (`dcsrch` + `dcstep`) as `MoreThuente` in
  `line_search/more_thuente.rs`. Defaults match Fortran `lnsrlb`
  (`ftol = 1e-3`, `gtol = 0.9`, `xtol = 0.1`) for parity. Available
  to `BFGS::with_line_search` for cross-validation.
- **S14.3 [done]** `LbfgsState<V>` in `core/state/lbfgs.rs`:
  chronological `(s, y)` history with capacity-bounded ring
  semantics; compact-form `SᵀY` / `SᵀS` Gram blocks; `theta`
  scaling; curvature-guarded `append_pair`.
- **S14.4 [done]** Port `cauchy.f` (generalized Cauchy point) in
  `solver/lbfgsb/cauchy.rs`, with compact-form helpers (`formt`,
  `bmv`, pure-Rust Cholesky / triangular solves, `hpsolb` min-heap)
  in `solver/lbfgsb/compact.rs`. Sidestepped the planned
  `AsFloatSlice` trait by giving the routines a slice-based API
  (`&[f64]` / `&mut [f64]`) and having the solver-level layer
  source slices per-backend at call time
  (nalgebra `as_slice`, faer `try_as_col_major().unwrap().as_slice`,
  `Vec` direct). Triangular solves on `wt` go straight to scalar
  Rust loops, preserving the Fortran two-step structured solve
  rather than collapsing to a generic SPD solve — matters for
  iteration-wise parity in S14.7.
- **S14.5 [done]** Port `subsm.f` (subspace minimization) in
  `solver/lbfgsb/subsm.rs`, including the uniform-α
  bound-backtracking path (`iword = 1` in Fortran) and the v3.0
  directional-derivative check at the original iterate. Consumes a
  precomputed `wn` (the `L·E·Lᵀ` factor of `K`), reusing the
  existing `solve_upper_tri{,_transposed}` helpers from
  `solver/lbfgsb/compact.rs` with the sign-flip on the top half
  between the two solves. `formk` (which builds `wn`) lands in
  S14.6 — until then `wn` is constructed by hand in the unit tests.
- **S14.6 [done]** `LBFGSB<S = MoreThuente>` solver wiring all of
  the above, with builder API mirroring `BFGS`. Lives in
  `solver/lbfgsb.rs`; the `formk` port (`solver/lbfgsb/formk.rs`)
  builds `wn` incrementally from the persistent `wn1` Gram cache.
  Per-backend slice extraction sits behind `backend::AsFloatSliceMut`
  with impls for `Vec<f64>`, `nalgebra::DVector<f64>`, and
  `faer::Col<f64>` (the planned external `AsFloatSlice` math trait
  was kept private to the L-BFGS-B module — the rest of the solver
  doesn't need it). Per-iter scratch lives in a lazy
  `LbfgsbWork` on `LbfgsState`, allocated once at `init`. The
  Fortran `goto 222` history-clear restart path is preserved as a
  single-shot restart in `next_iter`. Smoke convergence verified on
  a shifted-quadratic-in-box and unbounded Rosenbrock 2D; the
  full iteration-wise parity battery is S14.7.
- **S14.7 [done]** Validation tests. Convergence: across Vec /
  nalgebra / faer in `tests/lbfgsb_{vec,nalgebra,faer}.rs` —
  unbounded Rosenbrock 2D, BoothBoxed at the tight `[-1, 1]²`
  corner (optimum at `(1, 1)`, both bounds active), BoothBoxed with
  slack bounds (unconstrained recovery), and a strictly convex
  5-D quadratic with slack bounds. Iteration-wise parity:
  `tests/lbfgsb_iter_parity.rs` steps `LBFGSB` for 30 iters on
  Rosenbrock 5D with bounds `[0, 5]⁵` from infeasible start
  `(-1, 2, -1, 2, -1)` and asserts `≤ 1e-10` agreement with the
  Fortran reference (`m = 5`, `factr = 0`, `pgtol = 0`). The
  fixture is text-format `iter f x g` per line, dumped by
  `tests/fixtures/lbfgsb_driver.f` linked against
  `references/lbfgsb-v3.0/`; regeneration instructions in
  `tests/fixtures/README.md`. No `serde_json` dep — parse is
  whitespace `split_whitespace`. Plus a sanity comparator: BFGS
  with `MoreThuente` line search vs. `LBFGSB` on unbounded
  Rosenbrock — iteration counts comparable, confirming the
  limited-memory ≈ full-memory regime for `m ≥ 2` here.

## Cross-cutting (slot in opportunistically)

- **Per-solver "Backends" doc note** (tenet 5) — start with S3,
  retroactive for older solvers in same session.
- **Test-problem corpus** — Picheny, three-hump camel, Powell singular,
  Brown badly-scaled. Add as needed in solver tests rather than upfront.
- **`ParamVec<F>` marker** (TODO cleanup) — fold into the session that
  introduces the third user of the bound pair, probably S3 or S6.

## Ingestion order (read papers just-in-time)

1. Before **S2a**: faer paper + nalgebra-sparse user guide.
2. Before **S2b**: faer paper + nalgebra-sparse user guide (the
   sparse-factorization details get load-bearing here).
3. Before **S4**: Madsen/Nielsen/Tingleff (2004) + Nielsen 1999 +
   skim MINPACK `lmder`. (S3 deferred MNT — the GN pseudocode was
   short enough to derive without it; LM needs the full context.)
4. Before **S6**: Branch/Coleman/Li 1999 (TRF).
5. Before **S8**: Hansen CMA-ES tutorial.
6. Before **S9**: pycma bound-handling reference.
7. Before **S11**: Hansen 2011 (RR-7748 / arXiv:1110.4181) for the
   injection protocol + Melo & Iacca 2014 for empirical inner choice
   — both **done**, see `references/hansen-2011/NOTES.md` and
   `references/melo-iacca-2014/NOTES.md`.
8. Before **S12**: Bergmeir et al. 2016 (JSS, Rmalschains) — **done**,
   see `references/bergmeir-2016/source.md`. Promote the loose
   `references/Memetic_Algorithms_for_Continuous_Optimisation_Bas.pdf`
   into `references/molina-2010/` and run `ingest-paper` on it; the
   underlying methods paper has details the JSS software paper
   summarizes.

Use the `ingest-paper` skill before each session to pull the PDF into
`references/<name>/`.
