use crate::core::solver::Solver;
use crate::core::state::State;
use crate::core::termination::{TerminationCriterion, TerminationReason};

pub struct Executor<P, S, So> {
    problem: P,
    state: S,
    solver: So,
    max_iter: u64,
    criteria: Vec<Box<dyn TerminationCriterion<S>>>,
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
            criteria: Vec::new(),
        }
    }

    /// Convenience setter for the default `MaxIter` criterion. Equivalent
    /// effect to `terminate_on(MaxIter(n))` but mutates a dedicated field
    /// so subsequent calls replace rather than stack.
    pub fn max_iter(mut self, n: u64) -> Self {
        self.max_iter = n;
        self
    }

    /// Add a termination criterion. Criteria are checked in insertion
    /// order before each iteration (and before iter 0); the first to
    /// return `Some(_)` stops the run.
    pub fn terminate_on<C>(mut self, criterion: C) -> Self
    where
        C: TerminationCriterion<S> + 'static,
    {
        self.criteria.push(Box::new(criterion));
        self
    }

    pub fn run(mut self) -> (S, TerminationReason) {
        self.state = self.solver.init(&self.problem, self.state);
        loop {
            if self.state.iter() >= self.max_iter {
                return (self.state, TerminationReason::MaxIter);
            }
            for criterion in &mut self.criteria {
                if let Some(reason) = criterion.check(&self.state) {
                    return (self.state, reason);
                }
            }
            if let Some(reason) = self.solver.terminate(&self.state) {
                return (self.state, reason);
            }
            self.state = self.solver.next_iter(&self.problem, self.state);
            self.state.increment_iter();
        }
    }
}
