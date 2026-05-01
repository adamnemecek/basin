pub trait ScaledAdd<S> {
    fn scaled_add(&mut self, scalar: S, other: &Self);
}

pub trait NormSquared {
    fn norm_squared(&self) -> f64;
}

pub trait NormInfinity {
    fn norm_infinity(&self) -> f64;
}

/// Inner product of two same-shaped values. Used by line searches that take
/// an explicit search direction (Armijo and curvature checks both need
/// `gᵀd`). Generalizes `NormSquared`: `x.norm_squared() == x.dot(x)`.
pub trait Dot {
    fn dot(&self, other: &Self) -> f64;
}

/// In-place negation. Lets solvers compute `direction = -gradient` in a
/// backend-generic way without allocating per-iteration scratch types.
pub trait NegInPlace {
    fn neg_in_place(&mut self);
}

mod scalar;
mod vec;

#[cfg(feature = "nalgebra")]
mod nalgebra_backend;

#[cfg(feature = "ndarray")]
mod ndarray_backend;

#[cfg(feature = "faer")]
mod faer_backend;
