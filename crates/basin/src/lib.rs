//! basin — a numerical optimization library.
//!
//! The framework lives in [`core`]: problem traits the user implements
//! ([`CostFunction`], [`Gradient`], [`BoxConstraints`],
//! [`LinearInequalityConstraints`]), state shapes
//! solvers iterate over ([`State`], [`GradientState`], [`SimplexState`]),
//! the [`Solver`] trait, and a pluggable termination layer
//! ([`TerminationCriterion`]). Concrete solvers are in [`solver`];
//! line searches in [`line_search`].
//!
//! Start at [`Executor`] for the user-facing driver, or [`core`] for the
//! trait taxonomy and the iteration-loop contract.
//!
//! See `AGENTS.md` at the repo root for the design tenets that shape
//! these APIs (notably tenet 3 on framework-level termination, tenet 4
//! on first-class constraints, and tenet 5 on backend tiering).
#![warn(missing_docs)]
#![warn(rustdoc::broken_intra_doc_links)]

pub mod core;
pub mod line_search;
/// Catalogue of test problems used by the example tests and benchmarks.
#[cfg(feature = "problems")]
pub mod problems;
/// Concrete solver implementations.
pub mod solver;

pub use crate::core::barrier::LogBarrier;
pub use crate::core::constraint::{BoxConstraints, LinearInequalityConstraints};
pub use crate::core::executor::{run_loop, Executor, OptimizationResult, StepOutcome, Stepper};
pub use crate::core::inner::InnerExecutor;
pub use crate::core::math::{
    AddDiagonalInPlace, AddDiagonalVectorInPlace, BoxAffineScaling, ClampInPlace,
    DenseMatrixFromFn, Dot, GramMatrix, LinearSolveError, LinearSolveSpd, MatTransposeVec, MatVec,
    MaxDiagonal, NegInPlace, NormInfinity, NormSquared, SampleUniformBox, ScaledAdd, VectorIndex,
};
pub use crate::core::numdiff::{
    central_difference_gradient, central_difference_hessian, central_difference_jacobian,
    forward_difference_gradient, forward_difference_hessian, forward_difference_jacobian,
    FiniteDiff, Method,
};
pub use crate::core::problem::{CostFunction, Gradient, Hessian, Jacobian, Residual};
pub use crate::core::solver::Solver;
#[cfg(feature = "nalgebra")]
pub use crate::core::state::QuasiNewtonState;
pub use crate::core::state::{
    BasicPopulationState, BasicSimplexState, BasicState, GradientState, IntoInitialSimplex,
    LbfgsState, PopulationState, SimplexState, State,
};
pub use crate::core::termination::{
    CostTolerance, GradientTolerance, MaxCostEvals, MaxGradientEvals, MaxIter, MaxTime,
    ParamTolerance, ProjectedGradientTolerance, RelativeCostTolerance, RelativeGradientTolerance,
    RelativeParamTolerance, SimplexTolerance, TerminationCriterion, TerminationReason,
};
pub use crate::line_search::{
    Backtracking, Constant, LineSearch, LineSearchResult, MoreThuente, Wolfe,
};
pub use crate::solver::lbfgs::{LBFGS, LBFGSB};
#[cfg(feature = "nalgebra")]
pub use crate::solver::BFGS;
pub use crate::solver::{
    BarrierMethod, BoundedCmaEs, BoundedCmaInject, Brent, ClosureInner, CmaEs, CmaInject,
    GaussNewton, GradientDescent, LevenbergMarquardt, MaLsChCma, MaLsChState, MemeticInner,
    NelderMead, ProjectedGradientDescent, RandomSearch, Ssga, Trf,
};
