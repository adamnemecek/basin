pub mod core;
pub mod solver;

pub use crate::core::executor::Executor;
pub use crate::core::math::{NormSquared, ScaledAdd};
pub use crate::core::problem::{CostFunction, Gradient};
pub use crate::core::solver::Solver;
pub use crate::core::state::{BasicState, GradientState, State};
pub use crate::core::termination::{
    CostTolerance, GradientTolerance, MaxIter, MaxTime, ParamTolerance, TerminationCriterion,
    TerminationReason,
};
pub use crate::solver::{Backtracking, Constant, GradientDescent, StepSize};
