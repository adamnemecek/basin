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
2. Before **S2b**: faer paper + nalgebra-sparse user guide (the
   sparse-factorization details get load-bearing here).
3. Before **S4**: Madsen/Nielsen/Tingleff (2004) + Nielsen 1999 +
   skim MINPACK `lmder`. (S3 deferred MNT — the GN pseudocode was
   short enough to derive without it; LM needs the full context.)
4. Before **S6**: Branch/Coleman/Li 1999 (TRF).
5. Before **S8**: Hansen CMA-ES tutorial.
6. Before **S9**: pycma bound-handling reference.
7. Before **S11**: memetic-CMA-ES paper TBD.

Use the `ingest-paper` skill before each session to pull the PDF into
`references/<name>/`.
