//! Standard optimization test problems.
//!
//! Each problem is exposed two ways:
//! - Raw functions on `&[f64]` slices (e.g. [`rosenbrock`]) for callers that
//!   want to plug the math into their own wrappers, benchmarks, or visualizations.
//! - A pre-wrapped struct (e.g. [`Rosenbrock`]) implementing
//!   [`CostFunction`](crate::CostFunction) and [`Gradient`](crate::Gradient)
//!   for each enabled backend, so it can be handed straight to an
//!   [`Executor`](crate::Executor).
//!
//! Gated behind the `problems` feature (default-on). Disable with
//! `default-features = false` to drop the corpus from the build.

pub mod rosenbrock;

pub use rosenbrock::{rosenbrock, rosenbrock_gradient, Rosenbrock};
