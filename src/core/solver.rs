use crate::core::state::State;

pub trait Solver<P, S: State> {
    fn next_iter(&mut self, problem: &P, state: S) -> S;

    fn terminate(&self, _state: &S) -> bool {
        false
    }
}
