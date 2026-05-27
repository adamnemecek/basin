//! Dense linear-algebra ops for LA-heavy solvers (Gauss-Newton,
//! Levenberg-Marquardt, TRF). This is the second math tier per
//! `AGENTS.md` tenet 5: most of these operations are carried only by
//! backends that can implement them honestly (currently nalgebra and
//! faer). The two matvec ops ([`MatVec`], [`MatTransposeVec`]) are the
//! exception — they are cheap and honest on every backend, so `Vec<f64>`
//! (via [`DenseMatrix`](super::DenseMatrix)) and `ndarray::Array2<f64>`
//! implement them too; that is what lets the linear-constraint solvers
//! run on every backend. The *factorization* ops do not generalize:
//! `Vec<f64>` and `ndarray` deliberately omit [`LinearSolveSpd`] /
//! [`GramMatrix`] / eigen — there's no honest dense-factorization story
//! for either (`ndarray-linalg` requires system BLAS/LAPACK and breaks
//! the wasm-default tenet).
//!
//! The op set covers what Gauss-Newton needs at minimum, plus a
//! least-squares solve for sparse backends that ship QR:
//!
//! - [`MatVec`]: `y = A x` (matrix-vector product).
//! - [`MatTransposeVec`]: `y = Aᵀ x` (transposed matrix-vector
//!   product — used to form `Jᵀ r` without materializing `Jᵀ`).
//! - [`GramMatrix`]: `G = Aᵀ A` (the SPD normal-equations matrix).
//! - [`LinearSolveSpd`]: `A x = b` for SPD `A` (Cholesky inside).
//! - [`LinearSolveLstsq`]: `min_x ‖A x − b‖₂` (QR inside). Implemented
//!   per-backend wherever a sparse QR exists; not all backends do.
//!
//! Dense LU and `(AᵀA) x` for matrix-free Krylov inner solves are
//! deliberately deferred — they land alongside the first solver that
//! actually wants them (post-S6).

/// Matrix-vector product `y = A x`.
///
/// # Contract
///
/// - **Caller must:** pass `x` whose length equals `self.ncols()`.
///   Backends panic on shape mismatch.
/// - **Implementor must:** return a freshly allocated `V` of length
///   `self.nrows()`, with `y[i] = Σⱼ A[i, j] · x[j]`. The op is a
///   pure function of `(self, x)`.
///
/// # Backends
///
/// Implemented on every backend's dense matrix type:
/// [`DenseMatrix`](super::DenseMatrix) (with `V = Vec<f64>`, always
/// available), `nalgebra::DMatrix<f64>` (`V = DVector<f64>`),
/// `faer::Mat<f64>` (`V = faer::Col<f64>`), and `ndarray::Array2<f64>`
/// (`V = Array1<f64>`) — each behind its backend feature where
/// applicable.
pub trait MatVec<V> {
    /// Compute `A x`, allocating a fresh `V` of length `self.nrows()`.
    fn matvec(&self, x: &V) -> V;
}

/// Transposed matrix-vector product `y = Aᵀ x`. Lets least-squares
/// solvers form `Jᵀ r` without materializing `Jᵀ`.
///
/// # Contract
///
/// - **Caller must:** pass `x` whose length equals `self.nrows()`.
///   Backends panic on shape mismatch.
/// - **Implementor must:** return a freshly allocated `V` of length
///   `self.ncols()`, with `y[j] = Σᵢ A[i, j] · x[i]`. The op is a
///   pure function of `(self, x)` and must agree with [`MatVec`] on
///   the implicit transpose.
///
/// # Backends
///
/// Same backend coverage as [`MatVec`].
pub trait MatTransposeVec<V> {
    /// Compute `Aᵀ x`, allocating a fresh `V` of length `self.ncols()`.
    fn mat_transpose_vec(&self, x: &V) -> V;
}

/// Gram matrix `G = Aᵀ A`. The SPD matrix at the heart of the
/// Gauss-Newton normal equations. Returns `Self` because for both
/// supported dense backends a Gram of a dense matrix is the same
/// type — when sparse backends land in S2b the `Output` may need to
/// become an associated type, but premature parameterization here
/// would buy nothing for the dense prototype.
///
/// # Contract
///
/// - **Implementor must:** return a square `n × n` matrix where
///   `n = self.ncols()`. The result is symmetric and positive
///   semi-definite by construction; it is positive definite iff
///   `self` has full column rank. (Rank-deficient `A` produces a
///   `G` for which [`LinearSolveSpd::solve_spd`] returns
///   [`LinearSolveError::NotPositiveDefinite`].)
///
/// # Backends
///
/// Same backend coverage as [`MatVec`].
pub trait GramMatrix {
    /// Compute `Aᵀ A` and return the freshly allocated SPD matrix.
    fn gram(&self) -> Self;
}

/// Solve the SPD linear system `A x = b` via Cholesky factorization.
/// `A` is `self`; `b` is the right-hand side.
///
/// Owned-return rather than in-place — both backends offer in-place
/// solve paths, but every Cholesky factorization is `O(n³)` versus an
/// `O(n²)` allocation, so the unified owned-return shape isn't a
/// meaningful perf cost at this layer. An in-place variant can be
/// added behind a separate trait if a hot inner loop ever wants it.
///
/// # Contract
///
/// - **Caller must:** pass an SPD `self` and a `b` of length
///   `self.nrows()`. `self` must be square; backends panic otherwise.
/// - **Implementor must:** return [`LinearSolveError::NotPositiveDefinite`]
///   when the Cholesky factorization fails (zero or negative pivot).
///   On success, `x` satisfies `‖A x − b‖` to within the backend's
///   factorization accuracy.
///
/// # Backends
///
/// Same backend coverage as [`MatVec`]. Both backends use a
/// dense Cholesky (`L Lᵀ`); pivoting variants live in the backend
/// crates if needed.
pub trait LinearSolveSpd<V> {
    /// Solve `self · x = b` assuming `self` is SPD. Returns
    /// [`LinearSolveError::NotPositiveDefinite`] when factorization
    /// fails.
    fn solve_spd(&self, b: &V) -> Result<V, LinearSolveError>;
}

/// Least-squares solve `min_x ‖A x − b‖₂` via QR factorization. `A`
/// is `self`; `b` is the right-hand side. Unlike [`LinearSolveSpd`],
/// `A` need not be square or full-rank.
///
/// Owned-return for the same reasons as [`LinearSolveSpd`]: the
/// `O(mn²)` factorization dominates the `O(n)` allocation, and the
/// unified return shape stays honest across backends.
///
/// # Contract
///
/// - **Caller must:** pass a `b` of length `self.nrows()`. `self` may
///   be any shape; backends do not require `self` to be square.
/// - **Implementor must:** return [`LinearSolveError::Singular`] when
///   the underlying QR factorization itself fails (allocation or
///   index-overflow errors counted as `Singular` for callers who only
///   need pass/fail). On success, the returned `V` has length
///   `self.ncols()` and minimizes `‖A x − b‖₂` to within the backend's
///   factorization accuracy.
/// - **Caller must (numerical caveat):** rank-deficient inputs are
///   *not* guaranteed to surface as [`LinearSolveError::Singular`] —
///   sparse QR backends (faer) succeed on rank-deficient systems and
///   produce a solution whose components in the null space are
///   numerically meaningless. Callers that need rank-deficiency
///   detection should check `‖A x − b‖₂` against expected residuals
///   themselves.
///
/// # Backends
///
/// Implemented for `faer::sparse::SparseColMat<usize, f64>` (with
/// `V = faer::Col<f64>`) when the `faer` feature is enabled.
/// `nalgebra-sparse` does not ship a sparse QR at the pinned version,
/// so `nalgebra_sparse::CscMatrix<f64>` deliberately does not
/// implement this trait — per tenet 5, missing coverage is a
/// compile-time error rather than a runtime surprise.
pub trait LinearSolveLstsq<V> {
    /// Solve the least-squares problem `min_x ‖self · x − b‖₂` via QR.
    /// Returns [`LinearSolveError::Singular`] on numerical rank
    /// deficiency.
    fn solve_lstsq(&self, b: &V) -> Result<V, LinearSolveError>;
}

/// `max_i Aᵢᵢ` — the maximum diagonal entry of a square matrix.
/// Used by Levenberg-Marquardt to size the initial damping parameter
/// `μ₀ = τ · max diag(J(x₀)ᵀ J(x₀))` (Nielsen 1999 eq. 1.10).
///
/// # Contract
///
/// - **Caller must:** pass a square `self`. Backends panic otherwise.
/// - **Caller must (sparse precondition):** for sparse impls, missing
///   diagonal entries from the CSC pattern are treated as the implicit
///   zero. The Gram of any `A` with no zero columns has all-positive
///   diagonal entries, so the relevant case for LM is unaffected.
/// - **Implementor must:** return `maxᵢ self[(i, i)]` as `f64`. For an
///   empty matrix (0×0) the result is unspecified; backends may return
///   `0.0` or `f64::NEG_INFINITY`. Callers should not invoke on empty
///   matrices.
///
/// # Backends
///
/// Same coverage as [`AddDiagonalInPlace`].
pub trait MaxDiagonal {
    /// Compute the maximum diagonal entry as `f64`.
    fn max_diagonal(&self) -> f64;
}

/// Extract the diagonal `diag(self) ∈ R^n` of a square matrix into a
/// freshly allocated vector. CMA-ES with bounds (S9) needs the per-axis
/// variances `σ² · diag(C)` for the adaptive boundary penalty.
///
/// # Contract
///
/// - **Caller must:** pass a square `self`. Backends panic otherwise.
/// - **Implementor must:** return a fresh `V` of length `self.nrows()`
///   with `out[i] = self[(i, i)]`. The op is `O(n)`.
///
/// # Backends
///
/// Implemented for `nalgebra::DMatrix<f64>` (over `DVector<f64>`),
/// `faer::Mat<f64>` (over `Col<f64>`), and [`DenseMatrix`](super::DenseMatrix)
/// (over `Vec<f64>`) — the dense matrix backends that support
/// [`SymmetricEigen`], the gating requirement of CMA-ES.
pub trait MatDiagonal<V> {
    /// Return `diag(self)` as a fresh `V` of length `self.nrows()`.
    fn diagonal(&self) -> V;
}

/// In-place diagonal augmentation `A ← A + scalar · I`. The minimal
/// op needed to express the Levenberg-Marquardt damped normal-equations
/// matrix `JᵀJ + μI` without materializing the identity.
///
/// # Contract
///
/// - **Caller must:** pass a square `self`. Backends panic otherwise.
/// - **Caller must (sparse precondition):** for sparse impls, every
///   diagonal entry `(i, i)` must already exist in the sparsity
///   pattern of `self`. The Gram matrix `G = AᵀA` of any `A` with no
///   zero columns satisfies this (`Gᵢᵢ = ‖A·,ᵢ‖² > 0`), so callers
///   that only invoke `add_diagonal_in_place` on a freshly computed
///   [`GramMatrix::gram`] result are safe by construction. Backends
///   panic on a missing diagonal entry rather than silently growing
///   the pattern.
/// - **Implementor must:** add `scalar` to every diagonal entry of
///   `self` in place. Off-diagonal entries are untouched. The op is
///   `O(n)` for an `n × n` matrix.
///
/// # Backends
///
/// Implemented for `nalgebra::DMatrix<f64>` and `faer::Mat<f64>` at the
/// dense tier, and for `nalgebra_sparse::CscMatrix<f64>` and
/// `faer::sparse::SparseColMat<usize, f64>` at the sparse tier — same
/// coverage as [`GramMatrix`] / [`LinearSolveSpd`].
pub trait AddDiagonalInPlace {
    /// Add `scalar` to every diagonal entry of `self` in place.
    fn add_diagonal_in_place(&mut self, scalar: f64);
}

/// In-place diagonal augmentation `A ← A + diag(d)` for a vector `d`.
/// The vector counterpart of [`AddDiagonalInPlace`]. The diagonal of
/// the BCL trust-region-reflective subproblem is `c + μ·d²` — both
/// vectors — and this trait expresses the addition in one in-place
/// pass without materializing a full diagonal matrix.
///
/// # Contract
///
/// - **Caller must:** pass a square `self` and a `diag` of length
///   `self.nrows()`. Backends panic on shape mismatch or non-square.
/// - **Caller must (sparse precondition):** every diagonal entry
///   `(i, i)` must already exist in the sparsity pattern of `self`,
///   the same precondition as scalar [`AddDiagonalInPlace`]. The Gram
///   matrix `G = AᵀA` of any `A` with no zero columns satisfies this
///   (`Gᵢᵢ = ‖A·,ᵢ‖² > 0`); callers that only invoke
///   `add_diagonal_vector_in_place` on a freshly computed
///   [`GramMatrix::gram`] result are safe by construction.
/// - **Implementor must:** add `diag[i]` to `self[(i, i)]` for every
///   `i` in `0..self.nrows()`. Off-diagonal entries are untouched.
///   The op is `O(n)` for an `n × n` matrix.
///
/// # Backends
///
/// Same coverage as [`AddDiagonalInPlace`]: dense nalgebra
/// (`DMatrix<f64>` over `DVector<f64>`), dense faer (`Mat<f64>` over
/// `Col<f64>`), sparse nalgebra (`CscMatrix<f64>` over
/// `DVector<f64>`), sparse faer (`SparseColMat<usize, f64>` over
/// `Col<f64>`).
pub trait AddDiagonalVectorInPlace<V> {
    /// Add `diag[i]` to `self[(i, i)]` for every `i`, in place.
    fn add_diagonal_vector_in_place(&mut self, diag: &V);
}

/// `n × n` identity matrix constructor. The smallest piece of matrix
/// "fabric" needed by CMA-ES, which initializes its covariance as
/// `C = I` by default (Hansen 2016, "Initialization" in Figure 6) but is
/// generic over the matrix type. (An anisotropic initial covariance from
/// per-coordinate stds uses [`MatrixFromDiagonal`] instead.)
///
/// # Contract
///
/// - **Implementor must:** return a square `n × n` matrix with `1.0` on
///   the diagonal and `0.0` elsewhere. Allocates fresh storage per call.
///
/// # Backends
///
/// Implemented for `nalgebra::DMatrix<f64>` and `faer::Mat<f64>`. Sparse
/// counterparts are intentionally unimplemented — sparse identity is a
/// degenerate sparsity pattern that no current solver wants to construct
/// at runtime.
pub trait MatrixIdentity {
    /// Build the `n × n` identity matrix.
    fn identity(n: usize) -> Self;
}

/// `n × n` diagonal-matrix constructor from a vector: builds the matrix
/// with `diag[i]` on the diagonal and `0` elsewhere. The constructor
/// counterpart of [`MatDiagonal::diagonal`] (which extracts the diagonal).
/// CMA-ES uses it to seed an anisotropic initial covariance
/// `C = diag(stds²)` from a per-coordinate initial step-size vector
/// (`CmaEs::with_stds`); the isotropic default still uses
/// [`MatrixIdentity::identity`].
///
/// # Contract
///
/// - **Implementor must:** return a square `n × n` matrix (`n = diag`'s
///   length) with `out[(i, i)] = diag[i]` and `0` off the diagonal.
///   Allocates fresh storage per call. The op is `O(n²)` (dense fill).
///
/// # Backends
///
/// Implemented for `nalgebra::DMatrix<f64>` (over `DVector<f64>`),
/// `faer::Mat<f64>` (over `Col<f64>`), and [`DenseMatrix`](super::DenseMatrix)
/// (over `Vec<f64>`) — the dense matrix backends that support
/// [`SymmetricEigen`], CMA-ES's gating requirement. Sparse counterparts
/// are intentionally unimplemented, matching [`MatrixIdentity`].
pub trait MatrixFromDiagonal<V> {
    /// Build the `n × n` matrix with `diag[i]` on the diagonal, `0`
    /// elsewhere.
    fn from_diagonal(diag: &V) -> Self;
}

/// Maps a vector backend to its canonical dense matrix type and builds a
/// matrix of that type from a per-entry closure. Implemented on the
/// *vector* type so that finite-difference differentiation
/// ([`crate::core::numdiff`]) can synthesize a `Jacobian` / `Hessian`
/// matrix from a problem whose only typed handle is the parameter vector
/// `V`, with **one** generic impl rather than a per-backend blanket impl
/// (two `impl<P> Jacobian for FiniteDiff<P> where P: Residual<Param = …>`
/// blocks would collide under coherence — both have the head
/// `impl<P> … for FiniteDiff<P>`, and Rust does not use the associated-type
/// values of `where`-bounds to prove disjointness). Routing the matrix-type
/// choice through `V::Matrix` keeps the impl single-headed.
///
/// # Contract
///
/// - **Implementor must:** return a freshly allocated `rows × cols` matrix
///   with entry `(i, j) = f(i, j)`. `f` is called once per entry; the call
///   order is backend-defined (column-major for both current backends), so
///   callers that care about evaluation order must precompute their data
///   and let `f` be a pure read.
///
/// # Backends
///
/// Implemented for `nalgebra::DVector<f64>` (`Matrix = DMatrix<f64>`) and
/// `faer::Col<f64>` (`Matrix = Mat<f64>`). `Vec<f64>` and `ndarray` do not
/// implement it — they have no honest dense matrix type — so finite-
/// difference `Jacobian` / `Hessian` over them is a compile-time error
/// (tenet 5 in `AGENTS.md`), mirroring the analytic
/// [`Jacobian`](crate::core::problem::Jacobian) backend coverage.
pub trait DenseMatrixFromFn: Sized {
    /// The dense matrix type paired with this vector backend.
    type Matrix;

    /// Build a `rows × cols` matrix with entry `(i, j) = f(i, j)`.
    fn dense_from_fn<F: FnMut(usize, usize) -> f64>(rows: usize, cols: usize, f: F)
        -> Self::Matrix;
}

/// Symmetric (self-adjoint) eigendecomposition `A = U diag(λ) Uᵀ`. The
/// load-bearing op for CMA-ES, which factors its covariance every
/// iteration to compute both the sampling map `B D z_k = y_k ~ N(0, C)`
/// (Hansen 2016 eq. 39) and the conjugate-path scaling
/// `C^{−1/2} = B D^{−1} Bᵀ` (eq. 43).
///
/// # Contract
///
/// - **Caller must:** pass a square, symmetric `self`. Backends treat the
///   lower triangle as authoritative; non-symmetric inputs produce
///   meaningless eigenpairs without an error.
/// - **Implementor must:** return `(eigenvectors, eigenvalues)` where
///   columns of `eigenvectors` are an orthonormal basis and
///   `eigenvalues[i]` is the eigenvalue paired with column `i`. The
///   `eigenvectors` matrix is `n × n` and the `eigenvalues` vector is
///   length `n` (with `n = self.nrows()`).
/// - **Implementor may:** return eigenvalues in any order — backends
///   currently produce ascending order (faer) or unsorted (nalgebra),
///   and CMA-ES doesn't depend on either. Callers that need a specific
///   ordering must sort the pairs themselves.
/// - **Implementor must:** return [`SymmetricEigenError::Failed`] when
///   the underlying QR/Jacobi/divide-and-conquer iteration fails to
///   converge. Numerically near-singular but well-defined inputs
///   succeed; the eigenvalues may include very small values that
///   callers can clamp before taking square roots.
///
/// # Backends
///
/// Implemented for `nalgebra::DMatrix<f64>` (with `V = DVector<f64>`)
/// via `nalgebra::SymmetricEigen` (pure-Rust QR iteration), for
/// `faer::Mat<f64>` (with `V = faer::Col<f64>`) via
/// `faer::linalg::evd::self_adjoint_evd`, and for
/// [`DenseMatrix`](super::DenseMatrix) (with `V = Vec<f64>`) via a
/// pure-Rust cyclic Jacobi solver — so CMA-ES runs on the default
/// backend. The `DenseMatrix` impl is the worked precedent for tenet 5's
/// "broaden backend coverage when an op can be done honestly": the *solve*
/// factorizations ([`LinearSolveSpd`] / [`GramMatrix`]) are simply not yet
/// implemented for `Vec<f64>`, not categorically off-limits.
pub trait SymmetricEigen<V> {
    /// Eigendecompose a symmetric `self` into `(B, λ)` such that
    /// `self ≈ B diag(λ) Bᵀ` in floating-point arithmetic.
    ///
    /// Named `try_eigh` rather than `symmetric_eigen` so that calls go
    /// to the trait method without colliding with nalgebra's inherent
    /// `Matrix::symmetric_eigen` (which consumes `self` and returns a
    /// `SymmetricEigen` struct, not a `Result`).
    fn try_eigh(&self) -> Result<(Self, V), SymmetricEigenError>
    where
        Self: Sized;
}

/// Reasons a [`SymmetricEigen::symmetric_eigen`] call can fail. Variants
/// are backend-agnostic — backends translate their native error types
/// into these.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymmetricEigenError {
    /// The eigensolver iteration failed to converge to the requested
    /// accuracy. Both backends use bounded-iteration QR / divide-and-
    /// conquer methods that can in principle fail on pathological
    /// inputs.
    Failed,
}

impl core::fmt::Display for SymmetricEigenError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Failed => f.write_str("symmetric eigendecomposition failed to converge"),
        }
    }
}

impl core::error::Error for SymmetricEigenError {}

/// In-place rank-one update `self ← self + α · v · vᵀ`. CMA-ES uses
/// this twice per iteration: once with `(α, v) = (c_1, p_c)` for the
/// rank-one path-update term, and again, repeatedly, with
/// `(α, v) = (c_µ · w_i°, y_{i:λ})` for the rank-µ recombination term
/// (Hansen 2016 eq. 47).
///
/// # Contract
///
/// - **Caller must:** pass a square `self` and a `v` of length
///   `self.nrows()`. Backends panic on shape mismatch.
/// - **Implementor must:** add `α · v[i] · v[j]` to `self[(i, j)]` for
///   every `(i, j)`, in place. Off-diagonal entries are touched. The op
///   is `O(n²)` for an `n × n` matrix.
///
/// # Backends
///
/// Implemented for `nalgebra::DMatrix<f64>` (with `V = DVector<f64>`)
/// via `ger`, for `faer::Mat<f64>` (with `V = faer::Col<f64>`) via
/// the `matmul` accumulator, and for [`DenseMatrix`](super::DenseMatrix)
/// (with `V = Vec<f64>`, delegating to [`GeneralRankOneUpdate`] with
/// `u = v`) — so CMA-ES runs on the default backend. Sparse backends do
/// *not* implement this — a rank-one update of a sparse matrix would
/// densify the pattern, and CMA-ES's covariance is dense by construction
/// anyway.
pub trait RankOneUpdate<V> {
    /// Compute `self ← self + α · v · vᵀ` in place.
    fn rank_one_update(&mut self, alpha: f64, v: &V);
}

/// In-place *general* rank-one update `self ← self + α · u · vᵀ` with two
/// distinct vectors — BLAS `ger`. The asymmetric generalization of
/// [`RankOneUpdate`] (which is the `u == v` special case). BFGS needs the
/// asymmetric form: its inverse-Hessian update carries the cross terms
/// `−ρ · s · (Hy)ᵀ` and `−ρ · (Hy) · sᵀ`, where `s ≠ Hy` in general.
///
/// # Contract
///
/// - **Caller must:** pass a square `self`, a `u` of length `self.nrows()`,
///   and a `v` of length `self.ncols()`. Backends panic on shape mismatch.
/// - **Implementor must:** add `α · u[i] · v[j]` to `self[(i, j)]` for every
///   `(i, j)`, in place. Off-diagonal entries are touched. The op is `O(n²)`
///   for an `n × n` matrix.
///
/// The method is named `general_rank_one_update` rather than `ger` so calls
/// go to the trait method without colliding with nalgebra's inherent
/// `Matrix::ger` (a 4-arg `ger(α, x, y, β)` with a `β·self` term) — the same
/// defensive naming as [`SymmetricEigen::try_eigh`].
///
/// # Backends
///
/// Implemented for `nalgebra::DMatrix<f64>` (with `V = DVector<f64>`) via
/// `ger`, for `faer::Mat<f64>` (with `V = faer::Col<f64>`) via the `matmul`
/// accumulator, and for [`DenseMatrix`](super::DenseMatrix) (with
/// `V = Vec<f64>`) via a direct double loop — so BFGS runs on `Vec<f64>`,
/// nalgebra, and faer. Sparse backends do *not* implement this, matching
/// [`RankOneUpdate`].
pub trait GeneralRankOneUpdate<V> {
    /// Compute `self ← self + α · u · vᵀ` in place.
    fn general_rank_one_update(&mut self, alpha: f64, u: &V, v: &V);
}

/// Reasons a linear-solve trait call can fail. Variants are
/// backend-agnostic — backends translate their native error types
/// into these.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinearSolveError {
    /// Cholesky factorization encountered a zero or negative pivot.
    /// The matrix is not positive definite (it may still be positive
    /// semi-definite, e.g. when `A = Jᵀ J` with rank-deficient `J`).
    /// Reported by [`LinearSolveSpd`].
    NotPositiveDefinite,
    /// The matrix is numerically rank-deficient (zero on the diagonal
    /// of an `R` or `U` factor). Reported by [`LinearSolveLstsq`] for
    /// rank-deficient QR.
    Singular,
}

impl core::fmt::Display for LinearSolveError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotPositiveDefinite => f.write_str("matrix is not positive definite"),
            Self::Singular => f.write_str("matrix is singular"),
        }
    }
}

impl core::error::Error for LinearSolveError {}
