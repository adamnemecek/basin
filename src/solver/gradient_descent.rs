use crate::core::math::ScaledAdd;
use crate::core::problem::{CostFunction, Gradient};
use crate::core::solver::Solver;
use crate::core::state::BasicState;

pub struct GradientDescent {
    pub step_size: f64,
}

impl GradientDescent {
    pub fn new(step_size: f64) -> Self {
        Self { step_size }
    }
}

impl<P, V> Solver<P, BasicState<V>> for GradientDescent
where
    P: CostFunction<Param = V, Output = f64> + Gradient<Param = V, Gradient = V>,
    V: ScaledAdd<f64>,
{
    fn next_iter(&mut self, problem: &P, mut state: BasicState<V>) -> BasicState<V> {
        let grad = problem.gradient(&state.param);
        state.param.scaled_add(-self.step_size, &grad);
        state.cost = problem.cost(&state.param);
        state
    }
}
