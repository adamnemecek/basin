//! Component-wise clamp into a box `[lower, upper]`.
//!
//! First-class projection primitive for box-constrained solvers
//! ([`ProjectedGradientDescent`](crate::solver::ProjectedGradientDescent);
//! shared with the projected-gradient termination metric
//! [`ProjectedGradientTolerance`](crate::core::termination::ProjectedGradientTolerance)).

/// In-place component-wise `self[i] ← clamp(self[i], lower[i], upper[i])`.
///
/// # Contract
///
/// - **Caller must:** ensure `lower[i] ≤ upper[i]` for every component.
///   The implementor calls [`f64::clamp`], which panics on `NaN`
///   bounds or `lower > upper`.
/// - **Caller must:** pass values of the same shape as `self`. Backend
///   impls assert this and panic on mismatch.
/// - **Implementor must:** mutate `self` in place with no allocation.
pub trait ClampInPlace {
    /// Clamp every component of `self` into `[lower, upper]` element-wise.
    fn clamp_in_place(&mut self, lower: &Self, upper: &Self);
}
