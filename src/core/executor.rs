use crate::core::solver::Solver;
use crate::core::state::State;

pub struct Executor<P, S, So> {
    problem: P,
    state: S,
    solver: So,
}

impl<P, S, So> Executor<P, S, So>
where
    S: State,
    So: Solver<P, S>,
{
    pub fn new(problem: P, solver: So, state: S) -> Self {
        Self {
            problem,
            state,
            solver,
        }
    }

    pub fn run(mut self) -> S {
        while !self.solver.terminate(&self.state) {
            self.state = self.solver.next_iter(&self.problem, self.state);
            self.state.increment_iter();
        }
        self.state
    }
}
