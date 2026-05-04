//! Standard optimization test problems.
//!
//! Each problem is exposed three ways:
//! - Raw functions on `&[f64]` slices (e.g. [`rosenbrock`]) for callers that
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

pub mod rosenbrock;
pub mod spec;
pub mod sphere;

pub use rosenbrock::{rosenbrock, rosenbrock_gradient, Rosenbrock, ROSENBROCK_SPEC};
pub use spec::{Dimensionality, HasSpec, ProblemSpec, Properties, Reference};
pub use sphere::{sphere, sphere_gradient, Sphere, SPHERE_SPEC};

/// All catalogued problem specs, for browsing and filtering. Append new
/// problems here as they're added.
pub static ALL_SPECS: &[&ProblemSpec] = &[&ROSENBROCK_SPEC, &SPHERE_SPEC];
