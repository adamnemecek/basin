//! Dense linear-algebra ops for LA-heavy solvers (Gauss-Newton,
//! Levenberg-Marquardt, TRF). This is the second math tier per
//! `AGENTS.md` tenet 5: only backends that can implement these
//! operations honestly (currently nalgebra and faer) carry impls.
//! `Vec<f64>` and `ndarray` deliberately do not — there's no honest
//! `LinearSolveSpd` story for either at the dense level (no matrix
//! type for `Vec`; `ndarray-linalg` requires system BLAS/LAPACK and
//! breaks the wasm-default tenet).
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
/// Implemented for `nalgebra::DMatrix<f64>` (with `V = DVector<f64>`)
/// and `faer::Mat<f64>` (with `V = faer::Col<f64>`) when the
/// respective backend feature is enabled.
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
