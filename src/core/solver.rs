use crate::core::state::State;
use crate::core::termination::TerminationReason;

pub trait Solver<P, S: State> {
    /// One-time setup before the iteration loop. Solvers that need to seed
    /// derived state (e.g. cost or gradient at the initial param) should do
    /// so here so termination criteria can inspect a complete iter-0 state.
    fn init(&mut self, _problem: &P, state: S) -> S {
        state
    }

    fn next_iter(&mut self, problem: &P, state: S) -> S;

    /// Solver-specific termination, evaluated after framework criteria.
    /// Returning `Some(_)` halts the executor.
    fn terminate(&self, _state: &S) -> Option<TerminationReason> {
        None
    }
}
