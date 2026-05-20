//! Math abstraction the solvers depend on.
//!
//! Two tiers per `AGENTS.md` tenet 5:
//!
//! - **Vector tier** (this module): small ops every backend can implement
//!   well — [`ScaledAdd`], [`NormSquared`], [`NormInfinity`], [`Dot`],
//!   [`NegInPlace`]. Backend-generic solvers (gradient descent,
//!   Nelder-Mead) bound on these.
//! - **`linalg` tier** ([`linalg`]): LA-heavy ops — [`MatVec`],
//!   [`MatTransposeVec`], [`GramMatrix`], [`LinearSolveSpd`],
//!   [`LinearSolveLstsq`] — that only the matrix-capable backends
//!   (nalgebra, faer; sparse counterparts in S2b) implement. LA-heavy
//!   solvers (Gauss-Newton, LM) bound on these so other backends
//!   produce compile-time errors instead of runtime surprises.

/// In-place `self ← self + scalar · other`. Backend-generic vector update.
pub trait ScaledAdd<S> {
    /// Add `scalar · other` into `self` in place.
    fn scaled_add(&mut self, scalar: S, other: &Self);
}

/// `‖x‖₂² = Σ xᵢ²`. Avoids the `sqrt` cost when the squared form is
/// what's actually needed (most quadratic-cost convergence checks).
pub trait NormSquared {
    /// Compute `Σ xᵢ²` as `f64`.
    fn norm_squared(&self) -> f64;
}

/// `‖x‖_∞ = maxᵢ |xᵢ|`. Used by first-order optimality stopping rules
/// (e.g. `‖∇f‖_∞ ≤ tol`).
pub trait NormInfinity {
    /// Compute `maxᵢ |xᵢ|` as `f64`.
    fn norm_infinity(&self) -> f64;
}

/// Inner product of two same-shaped values. Used by line searches that take
/// an explicit search direction (Armijo and curvature checks both need
/// `gᵀd`). Generalizes `NormSquared`: `x.norm_squared() == x.dot(x)`.
pub trait Dot {
    /// Compute `Σᵢ self[i] · other[i]` as `f64`.
    fn dot(&self, other: &Self) -> f64;
}

/// In-place negation. Lets solvers compute `direction = -gradient` in a
/// backend-generic way without allocating per-iteration scratch types.
pub trait NegInPlace {
    /// Negate every component of `self` in place.
    fn neg_in_place(&mut self);
}

/// In-place scalar multiplication `self ← scalar · self`. Used by
/// CMA-ES to update the cumulation paths (`p_σ ← (1−c_σ) p_σ + …`,
/// Hansen 2016 eq. 31) and the covariance matrix
/// (`C ← (1 + c_1 δ_h − c_1 − c_µ Σ w_j) C + …`, eq. 47) without
/// allocating a clone per iteration.
///
/// `ScaledAdd<f64>` already covers `self ← self + s · other`; the
/// borrow checker forbids `self.scaled_add(s, &self)`, so an honest
/// in-place scale needs its own trait.
pub trait ScaleInPlace {
    /// Multiply every component of `self` by `scalar` in place.
    fn scale_in_place(&mut self, scalar: f64);
}

/// Number of components in a 1-D vector. Used by CMA-ES to derive the
/// search-space dimension `n` from a template vector at solver
/// construction time, so callers don't have to thread `n` separately
/// from the initial mean. Method named `vec_len` to avoid colliding
/// with the inherent `len()` methods on `Vec`, `DVector`, `Array1`,
/// `Col`.
pub trait VectorLen {
    /// Number of components in `self`.
    fn vec_len(&self) -> usize;
}

/// In-place componentwise multiplication `self[i] ← self[i] · other[i]`.
/// CMA-ES uses this to apply the diagonal `D` (sqrt-eigenvalue) factor:
/// the sampling step `y_k = B D z_k` is `z.component_mul_assign(&d);
/// y = B.matvec(&z)`, and the conjugate-path step `C^{−1/2} v =
/// B (1/d ⊙ Bᵀv)` is the same pattern with `1/d`.
pub trait ComponentMulAssign {
    /// Multiply `self[i]` by `other[i]` for every `i`, in place.
    fn component_mul_assign(&mut self, other: &Self);
}

/// In-place componentwise maximum `self[i] ← max(self[i], other[i])`.
/// Levenberg-Marquardt uses this to maintain the monotone running-max
/// scaling diagonal `D_k = max(D_{k−1}, diag(JᵀJ))` of MINPACK-style
/// Marquardt damping (Moré 1978): a parameter whose column curvature
/// momentarily drops doesn't lose the damping floor accumulated from
/// earlier iterations.
pub trait ComponentMaxAssign {
    /// Set `self[i]` to `max(self[i], other[i])` for every `i`, in place.
    fn component_max_assign(&mut self, other: &Self);
}

/// In-place floor of non-positive entries to a positive `value`,
/// leaving strictly-positive entries untouched
/// (`self[i] ← value` where `self[i] ≤ 0`, else unchanged).
///
/// This is *not* a blanket lower-clamp: a legitimately small positive
/// entry keeps its value. It exists for MINPACK's zero-column guard in
/// Marquardt-scaled Levenberg-Marquardt — a Jacobian column that is
/// entirely zero gives `diag(JᵀJ)ⱼ = 0`, which would make the damping
/// `μ·D` vanish on that coordinate and leave the normal-equations
/// matrix singular there. MINPACK sets such a column's scale to `1`
/// (lmder, `mode = 1`); flooring zeros to `1` reproduces that, so a
/// fully-insensitive parameter simply stays put instead of failing the
/// Cholesky.
pub trait FloorZerosInPlace {
    /// Replace every entry `≤ 0` with `value`; leave positive entries
    /// unchanged.
    fn floor_zeros_in_place(&mut self, value: f64);
}

mod cl_scaling;
mod clamp;
mod linalg;
mod sample;
mod scalar;
mod vec;

#[cfg(feature = "nalgebra")]
mod nalgebra_backend;

#[cfg(feature = "nalgebra")]
mod nalgebra_sparse_backend;

#[cfg(feature = "ndarray")]
mod ndarray_backend;

#[cfg(feature = "faer")]
mod faer_backend;

#[cfg(feature = "faer")]
mod faer_sparse_backend;

pub use cl_scaling::BoxAffineScaling;
pub use clamp::ClampInPlace;
pub use linalg::{
    AddDiagonalInPlace, AddDiagonalVectorInPlace, GramMatrix, LinearSolveError, LinearSolveLstsq,
    LinearSolveSpd, MatDiagonal, MatTransposeVec, MatVec, MatrixIdentity, MaxDiagonal,
    RankOneUpdate, SymmetricEigen, SymmetricEigenError,
};
pub use sample::{SampleStandardNormal, SampleUniformBox};
