//! Dense linear-algebra ops for LA-heavy solvers (Gauss-Newton,
//! Levenberg-Marquardt, TRF). This is the second math tier per
//! `AGENTS.md` tenet 5: only backends that can implement these
//! operations honestly (currently nalgebra and faer) carry impls.
//! `Vec<f64>` and `ndarray` deliberately do not — there's no honest
//! `LinearSolveSpd` story for either at the dense level (no matrix
//! type for `Vec`; `ndarray-linalg` requires system BLAS/LAPACK and
//! breaks the wasm-default tenet).
//!
//! The op set is intentionally lean — exactly what a Gauss-Newton
//! inner step needs:
//!
//! - [`MatVec`]: `y = A x` (matrix-vector product).
//! - [`MatTransposeVec`]: `y = Aᵀ x` (transposed matrix-vector
//!   product — used to form `Jᵀ r` without materializing `Jᵀ`).
//! - [`GramMatrix`]: `G = Aᵀ A` (the SPD normal-equations matrix).
//! - [`LinearSolveSpd`]: `A x = b` for SPD `A` (Cholesky inside).
//!
//! QR / LU paths and `(AᵀA) x` for matrix-free Krylov inner solves
//! are deliberately deferred — they land alongside the first solver
//! that actually wants them (S3 / post-S6 respectively).

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

/// Reasons a [`LinearSolveSpd`] (or future linear-solve trait) call
/// can fail. Variants are backend-agnostic — backends translate their
/// native error types into these.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinearSolveError {
    /// Cholesky factorization encountered a zero or negative pivot.
    /// The matrix is not positive definite (it may still be positive
    /// semi-definite, e.g. when `A = Jᵀ J` with rank-deficient `J`).
    NotPositiveDefinite,
    /// The matrix is singular (zero pivot in a non-Cholesky
    /// factorization). Reserved for future LU / QR paths; currently
    /// unused by [`LinearSolveSpd`], which reports rank deficiency
    /// as [`Self::NotPositiveDefinite`].
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
