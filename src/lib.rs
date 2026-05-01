pub mod core;
pub mod solver;

pub use crate::core::executor::{Executor, OptimizationResult};
pub use crate::core::math::{NormInfinity, NormSquared, ScaledAdd};
pub use crate::core::problem::{CostFunction, Gradient};
pub use crate::core::solver::Solver;
pub use crate::core::state::{
    BasicSimplexState, BasicState, GradientState, IntoInitialSimplex, SimplexState, State,
};
pub use crate::core::termination::{
    CostTolerance, GradientTolerance, MaxCostEvals, MaxIter, MaxTime, ParamTolerance,
    SimplexTolerance, TerminationCriterion, TerminationReason,
};
pub use crate::solver::{
    Backtracking, Constant, GradientDescent, NelderMead, StepResult, StepSize,
};
