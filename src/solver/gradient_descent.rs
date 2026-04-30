use crate::core::math::ScaledAdd;
use crate::core::problem::{CostFunction, Gradient};
use crate::core::solver::Solver;
use crate::core::state::BasicState;
use crate::solver::step_size::{Constant, StepSize};

pub struct GradientDescent<S> {
    step_size: S,
}

impl GradientDescent<Constant> {
    pub fn new(alpha: f64) -> Self {
        Self {
            step_size: Constant(alpha),
        }
    }
}

impl<S> GradientDescent<S> {
    pub fn with_step_size(step_size: S) -> Self {
        Self { step_size }
    }
}

impl<P, V, S> Solver<P, BasicState<V>> for GradientDescent<S>
where
    P: CostFunction<Param = V, Output = f64> + Gradient<Param = V, Gradient = V>,
    V: ScaledAdd<f64>,
    S: StepSize<P, V>,
{
    fn next_iter(&mut self, problem: &P, mut state: BasicState<V>) -> BasicState<V> {
        // Invariant: state.cost matches state.param on entry. BasicState::new
        // seeds cost as INFINITY, so evaluate it lazily on the first iteration.
        if !state.cost.is_finite() {
            state.cost = problem.cost(&state.param);
        }
        let grad = problem.gradient(&state.param);
        let alpha = self
            .step_size
            .next(problem, &state.param, state.cost, &grad);
        state.param.scaled_add(-alpha, &grad);
        state.cost = problem.cost(&state.param);
        state
    }
}
