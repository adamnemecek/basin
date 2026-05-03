use crate::core::state::State;
use crate::core::termination::TerminationReason;

pub trait Solver<P, S: State> {
    /// One-time setup before the iteration loop. Solvers that need to seed
    /// derived state (e.g. cost or gradient at the initial param) should do
    /// so here so termination criteria can inspect a complete iter-0 state.
    fn init(&mut self, _problem: &P, state: S) -> S {
        state
    }

    /// Advance one iteration. Return the (possibly mutated) state, plus an
    /// optional termination reason if the solver determined mid-iter that
    /// it must halt — typically a line-search bailing out, a non-descent
    /// direction, or some other internal failure. Returning `Some(_)` stops
    /// the executor immediately; the iteration counter is *not* incremented
    /// in that case, so `state.iter()` reflects the last fully completed
    /// iteration.
    fn next_iter(&mut self, problem: &P, state: S) -> (S, Option<TerminationReason>);

    /// Pre-iter solver-specific termination, evaluated after framework
    /// criteria but before each `next_iter` call (including iter 0, after
    /// `init`). Returning `Some(_)` halts the executor. Use this for clean
    /// convergence tests that depend only on the current state; mid-iter
    /// failures should be reported via `next_iter`'s return value.
    fn terminate(&self, _state: &S) -> Option<TerminationReason> {
        None
    }
}
