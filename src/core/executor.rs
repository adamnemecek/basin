use crate::core::solver::Solver;
use crate::core::state::State;
use crate::core::termination::{TerminationCriterion, TerminationReason};

/// Outcome of an optimisation run.
///
/// Owns the final solver state plus the reason the executor stopped.
/// Delegates `param()` / `cost()` / `iter()` to the underlying state so
/// callers don't need to import `State` for the common reads.
pub struct OptimizationResult<S> {
    pub state: S,
    pub reason: TerminationReason,
}

impl<S: State> OptimizationResult<S> {
    pub fn param(&self) -> &S::Param {
        self.state.param()
    }

    pub fn cost(&self) -> S::Float {
        self.state.cost()
    }

    pub fn iter(&self) -> u64 {
        self.state.iter()
    }

    pub fn cost_evals(&self) -> u64 {
        self.state.cost_evals()
    }

    pub fn into_state(self) -> S {
        self.state
    }
}

/// Drive a solver to completion against a borrowed problem.
///
/// `Executor` is a thin owning wrapper over this. Composed solvers
/// (e.g. CG inside CMA, NM inside DE) call `run_loop` directly so they
/// can run an inner solver against the outer's `&P` without taking
/// ownership of the problem.
///
/// Semantics match `Executor::run`: `init` is called once, then on each
/// iteration framework `criteria` are checked in insertion order before
/// the solver's own `terminate` hook, before stepping. `max_iter` is
/// checked against `state.iter()` and exits with `TerminationReason::MaxIter`.
pub fn run_loop<P, S, So>(
    problem: &P,
    mut state: S,
    solver: &mut So,
    criteria: &mut [Box<dyn TerminationCriterion<S>>],
    max_iter: u64,
) -> OptimizationResult<S>
where
    S: State,
    So: Solver<P, S>,
{
    state = solver.init(problem, state);
    loop {
        if state.iter() >= max_iter {
            return OptimizationResult {
                state,
                reason: TerminationReason::MaxIter,
            };
        }
        for criterion in criteria.iter_mut() {
            if let Some(reason) = criterion.check(&state) {
                return OptimizationResult { state, reason };
            }
        }
        if let Some(reason) = solver.terminate(&state) {
            return OptimizationResult { state, reason };
        }
        state = solver.next_iter(problem, state);
        state.increment_iter();
    }
}

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

    pub fn run(self) -> OptimizationResult<S> {
        let Executor {
            problem,
            state,
            mut solver,
            max_iter,
            mut criteria,
        } = self;
        run_loop(&problem, state, &mut solver, &mut criteria, max_iter)
    }
}
