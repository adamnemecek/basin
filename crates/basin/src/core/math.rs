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

mod clamp;
mod linalg;
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

pub use clamp::ClampInPlace;
pub use linalg::{
    AddDiagonalInPlace, GramMatrix, LinearSolveError, LinearSolveLstsq, LinearSolveSpd,
    MatTransposeVec, MatVec, MaxDiagonal,
};
