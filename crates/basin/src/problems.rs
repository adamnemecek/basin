//! Standard optimization test problems.
//!
//! Each problem is exposed three ways:
//! - Raw functions on `&[f64]` slices (e.g. [`rosenbrock()`]) for callers that
//!   want to plug the math into their own wrappers, benchmarks, or visualizations.
//! - A pre-wrapped struct (e.g. [`Rosenbrock`]) implementing
//!   [`CostFunction`](crate::CostFunction) and [`Gradient`](crate::Gradient)
//!   for each enabled backend, so it can be handed straight to an
//!   [`Executor`](crate::Executor).
//! - A static [`ProblemSpec`] with metadata (dimensionality, mathematical
//!   properties, references) for use in catalog UIs and filtering. Iterate
//!   [`ALL_SPECS`] to enumerate the corpus.
//!
//! Most functions in this corpus are catalogued in Jamil & Yang (2013),
//! *A Literature Survey of Benchmark Functions For Global Optimisation
//! Problems*, arXiv:1308.4008. Per-problem references on each
//! [`ProblemSpec`] cite the original source where applicable.
//!
//! Gated behind the `problems` feature (default-on). Disable with
//! `default-features = false` to drop the corpus from the build.

pub mod beale;
pub mod booth;
pub mod goldstein_price;
pub mod matyas;
pub mod mccormick;
pub mod powell_singular;
pub mod rosenbrock;
pub mod sparse_least_squares;
pub mod spec;
pub mod sphere;

pub use beale::{beale, beale_gradient, Beale, BEALE_SPEC};
pub use booth::{booth, booth_gradient, Booth, BoothBoxed, BOOTH_SPEC};
pub use goldstein_price::{
    goldstein_price, goldstein_price_gradient, GoldsteinPrice, GOLDSTEIN_PRICE_SPEC,
};
pub use matyas::{matyas, matyas_gradient, Matyas, MATYAS_SPEC};
pub use mccormick::{mccormick, mccormick_gradient, McCormick, MCCORMICK_SPEC};
pub use powell_singular::{
    powell_singular, powell_singular_jacobian, powell_singular_residuals, PowellSingular,
    POWELL_SINGULAR_SPEC,
};
pub use rosenbrock::{
    rosenbrock, rosenbrock_gradient, rosenbrock_residuals, rosenbrock_residuals_jacobian,
    Rosenbrock, RosenbrockResiduals, ROSENBROCK_SPEC,
};
pub use sparse_least_squares::{SparseLeastSquares, SPARSE_LEAST_SQUARES_SPEC};
pub use spec::{Dimensionality, HasSpec, ProblemSpec, Properties, Reference};
pub use sphere::{sphere, sphere_gradient, Sphere, SPHERE_SPEC};

/// All catalogued problem specs, for browsing and filtering. Append new
/// problems here as they're added.
pub static ALL_SPECS: &[&ProblemSpec] = &[
    &ROSENBROCK_SPEC,
    &SPHERE_SPEC,
    &BEALE_SPEC,
    &BOOTH_SPEC,
    &MATYAS_SPEC,
    &MCCORMICK_SPEC,
    &GOLDSTEIN_PRICE_SPEC,
    &POWELL_SINGULAR_SPEC,
    &SPARSE_LEAST_SQUARES_SPEC,
];
