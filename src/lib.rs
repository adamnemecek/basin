pub mod core;
pub mod line_search;
pub mod solver;

pub use crate::core::constraint::BoxConstrained;
pub use crate::core::executor::{run_loop, Executor, OptimizationResult};
pub use crate::core::math::{Dot, NegInPlace, NormInfinity, NormSquared, ScaledAdd};
pub use crate::core::problem::{CostFunction, Gradient};
pub use crate::core::solver::Solver;
#[cfg(feature = "nalgebra")]
pub use crate::core::state::QuasiNewtonState;
pub use crate::core::state::{
    BasicSimplexState, BasicState, GradientState, IntoInitialSimplex, SimplexState, State,
};
pub use crate::core::termination::{
    CostTolerance, GradientTolerance, MaxCostEvals, MaxGradientEvals, MaxIter, MaxTime,
    ParamTolerance, SimplexTolerance, TerminationCriterion, TerminationReason,
};
pub use crate::line_search::{Backtracking, Constant, LineSearch, LineSearchResult, Wolfe};
#[cfg(feature = "nalgebra")]
pub use crate::solver::BFGS;
pub use crate::solver::{Brent, GradientDescent, NelderMead};
