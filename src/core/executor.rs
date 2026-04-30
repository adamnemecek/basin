use crate::core::solver::Solver;
use crate::core::state::State;

pub struct Executor<P, S, So> {
    problem: P,
    state: S,
    solver: So,
    max_iter: u64,
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
            max_iter: 1000,
        }
    }

    pub fn max_iter(mut self, n: u64) -> Self {
        self.max_iter = n;
        self
    }

    pub fn run(mut self) -> S {
        while self.state.iter() < self.max_iter && !self.solver.terminate(&self.state) {
            self.state = self.solver.next_iter(&self.problem, self.state);
            self.state.increment_iter();
        }
        self.state
    }
}
