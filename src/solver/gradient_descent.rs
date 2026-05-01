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
    fn init(&mut self, problem: &P, mut state: BasicState<V>) -> BasicState<V> {
        // Seed cost and gradient at the initial param so iter-0 termination
        // checks (e.g. `GradientTolerance` on a near-optimal start) see a
        // complete state. Same work we'd do on iter 1, hoisted.
        state.cost = Some(problem.cost(&state.param));
        state.gradient = Some(problem.gradient(&state.param));
        state
    }

    fn next_iter(&mut self, problem: &P, mut state: BasicState<V>) -> BasicState<V> {
        let grad = state
            .gradient
            .take()
            .expect("gradient not set: Solver::init must run before next_iter");
        let prev_cost = state
            .cost
            .expect("cost not set: Solver::init must run before next_iter");
        let alpha = self.step_size.next(problem, &state.param, prev_cost, &grad);
        state.param.scaled_add(-alpha, &grad);
        state.cost = Some(problem.cost(&state.param));
        state.gradient = Some(problem.gradient(&state.param));
        state
    }
}
