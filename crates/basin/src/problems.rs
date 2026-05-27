//! Standard optimization test problems.
//!
//! Each problem is exposed three ways:
//! - Raw functions on `&[f64]` slices (e.g.
//!   [`rosenbrock()`](crate::problems::rosenbrock::rosenbrock)) for callers that
//!   want to plug the math into their own wrappers, benchmarks, or visualizations.
//! - A pre-wrapped struct (e.g. [`Rosenbrock`](crate::problems::Rosenbrock))
//!   implementing
//!   [`CostFunction`](crate::CostFunction) and [`Gradient`](crate::Gradient)
//!   for each enabled backend, so it can be handed straight to an
//!   [`Executor`](crate::Executor).
//! - A static [`ProblemSpec`](crate::problems::ProblemSpec) with metadata
//!   (dimensionality, mathematical
//!   properties, references) for use in catalog UIs and filtering. Iterate
//!   [`ALL_SPECS`](crate::problems::ALL_SPECS) to enumerate the corpus.
//!
//! Most functions in this corpus are catalogued in Jamil & Yang (2013),
//! *A Literature Survey of Benchmark Functions For Global Optimisation
//! Problems*, arXiv:1308.4008. Per-problem references on each
//! [`ProblemSpec`](crate::problems::ProblemSpec) cite the original source where
//! applicable.
//!
//! Gated behind the `problems` feature (default-on). Disable with
//! `default-features = false` to drop the corpus from the build.

pub mod ackley;
pub mod beale;
pub mod booth;
pub mod bukin;
pub mod constrained_quadratic;
pub mod cross_in_tray;
pub mod easom;
pub mod eggholder;
pub mod equality_constrained_quadratic;
pub mod exponential_fit;
pub mod goldstein_price;
pub mod himmelblau;
pub mod holder_table;
pub mod levy;
pub mod matyas;
pub mod mccormick;
pub mod picheny;
pub mod powell_singular;
pub mod rastrigin;
pub mod rosenbrock;
pub mod schaffer;
pub mod sparse_least_squares;
pub mod spec;
pub mod sphere;
pub mod styblinski_tang;
pub mod three_hump_camel;
pub mod zero;

pub use ackley::{ackley, Ackley, AckleyBoxed, ACKLEY_SPEC};
pub use beale::{beale, beale_gradient, Beale, BEALE_SPEC};
pub use booth::{
    booth, booth_gradient, booth_residuals, booth_residuals_jacobian, Booth, BoothBoxed,
    BoothBoxedResiduals, BoothResiduals, BOOTH_SPEC,
};
pub use bukin::{bukin_n6, BukinN6, BUKIN_N6_SPEC};
pub use constrained_quadratic::{ConstrainedQuadratic, CONSTRAINED_QUADRATIC_SPEC};
pub use cross_in_tray::{cross_in_tray, CrossInTray, CROSS_IN_TRAY_SPEC};
pub use easom::{easom, easom_gradient, Easom, EASOM_SPEC};
pub use eggholder::{eggholder, Eggholder, EGGHOLDER_SPEC};
pub use equality_constrained_quadratic::{
    EqualityConstrainedQuadratic, EQUALITY_CONSTRAINED_QUADRATIC_SPEC,
};
pub use exponential_fit::{
    exponential_fit, exponential_fit_jacobian, exponential_fit_residuals, ExponentialFit,
    EXPONENTIAL_FIT_SPEC,
};
pub use goldstein_price::{
    goldstein_price, goldstein_price_gradient, GoldsteinPrice, GOLDSTEIN_PRICE_SPEC,
};
pub use himmelblau::{himmelblau, himmelblau_gradient, Himmelblau, HIMMELBLAU_SPEC};
pub use holder_table::{holder_table, HolderTable, HOLDER_TABLE_SPEC};
pub use levy::{levy, levy_gradient, Levy, LevyBoxed, LEVY_SPEC};
pub use matyas::{matyas, matyas_gradient, Matyas, MATYAS_SPEC};
pub use mccormick::{mccormick, mccormick_gradient, McCormick, MCCORMICK_SPEC};
pub use picheny::{picheny, picheny_gradient, Picheny, PICHENY_SPEC};
pub use powell_singular::{
    powell_singular, powell_singular_jacobian, powell_singular_residuals, PowellSingular,
    POWELL_SINGULAR_SPEC,
};
pub use rastrigin::{rastrigin, Rastrigin, RastriginBoxed, RASTRIGIN_SPEC};
pub use rosenbrock::{
    rosenbrock, rosenbrock_gradient, rosenbrock_residuals, rosenbrock_residuals_jacobian,
    Rosenbrock, RosenbrockResiduals, ROSENBROCK_SPEC,
};
pub use schaffer::{
    schaffer_n2, schaffer_n2_gradient, schaffer_n4, SchafferN2, SchafferN4, SCHAFFER_N2_SPEC,
    SCHAFFER_N4_SPEC,
};
pub use sparse_least_squares::{
    SparseLeastSquares, SparseLeastSquaresBoxed, SPARSE_LEAST_SQUARES_SPEC,
};
pub use spec::{Dimensionality, HasSpec, ProblemSpec, Properties, Reference};
pub use sphere::{sphere, sphere_gradient, Sphere, SPHERE_SPEC};
pub use styblinski_tang::{
    styblinski_tang, styblinski_tang_gradient, StyblinskiTang, StyblinskiTangBoxed,
    STYBLINSKI_TANG_SPEC,
};
pub use three_hump_camel::{
    three_hump_camel, three_hump_camel_gradient, ThreeHumpCamel, THREE_HUMP_CAMEL_SPEC,
};
pub use zero::{zero, zero_gradient, Zero, ZERO_SPEC};

/// All catalogued problem specs, for browsing and filtering. Append new
/// problems here as they're added.
pub static ALL_SPECS: &[&ProblemSpec] = &[
    &ROSENBROCK_SPEC,
    &SPHERE_SPEC,
    &BEALE_SPEC,
    &BOOTH_SPEC,
    &CONSTRAINED_QUADRATIC_SPEC,
    &EQUALITY_CONSTRAINED_QUADRATIC_SPEC,
    &MATYAS_SPEC,
    &MCCORMICK_SPEC,
    &GOLDSTEIN_PRICE_SPEC,
    &POWELL_SINGULAR_SPEC,
    &RASTRIGIN_SPEC,
    &SPARSE_LEAST_SQUARES_SPEC,
    &EXPONENTIAL_FIT_SPEC,
    &THREE_HUMP_CAMEL_SPEC,
    &PICHENY_SPEC,
    &ZERO_SPEC,
    &HIMMELBLAU_SPEC,
    &ACKLEY_SPEC,
    &LEVY_SPEC,
    &STYBLINSKI_TANG_SPEC,
    &SCHAFFER_N2_SPEC,
    &SCHAFFER_N4_SPEC,
    &BUKIN_N6_SPEC,
    &CROSS_IN_TRAY_SPEC,
    &EASOM_SPEC,
    &EGGHOLDER_SPEC,
    &HOLDER_TABLE_SPEC,
];
