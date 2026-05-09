//! Component-wise uniform sample in a box.
//!
//! Construction primitive for stochastic solvers that draw fresh
//! candidates from a feasible box. The first user is
//! [`RandomSearch`](crate::solver::RandomSearch); CMA-ES will land its
//! own (Gaussian) sampling primitive when S8 needs it.

use rand::Rng;

/// Per-component uniform sample in `[lower[i], upper[i]]`. Returns a
/// fresh value of the same shape as `lower` / `upper`.
///
/// # Contract
///
/// - **Caller must:** pass `lower` and `upper` of equal length, with
///   `lower[i] ≤ upper[i]` for every component. Equal bounds are
///   allowed (the corresponding component is pinned to that value).
/// - **Implementor must:** call
///   [`Rng::random_range`](rand::Rng::random_range) once per component
///   on the inclusive range `lower[i]..=upper[i]`, and return a freshly
///   allocated value of length `lower.len()`. Sampling is component-
///   independent — the same `(rng, lower, upper)` produces the same
///   draws across backends only modulo iteration order.
/// - **Implementor must:** advance `rng` by exactly `n` `f64` draws so
///   that reproducibility is well-defined for fixed-seed RNGs.
///
/// Bound semantics match `rand::distr::Uniform::new_inclusive` —
/// `lower[i] == upper[i]` deterministically returns that value.
pub trait SampleUniformBox: Sized {
    /// Sample a fresh value with each component drawn uniformly from
    /// `[lower[i], upper[i]]`.
    fn sample_uniform_box<R: Rng + ?Sized>(lower: &Self, upper: &Self, rng: &mut R) -> Self;
}
