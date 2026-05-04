use crate::core::math::{NegInPlace, ScaledAdd};
use crate::core::problem::{CostFunction, Gradient};
use crate::core::solver::Solver;
use crate::core::state::BasicState;
use crate::core::termination::TerminationReason;
use crate::line_search::{Constant, LineSearch};

/// Steepest-descent solver: step in the direction of `−∇f(x)` with a
/// pluggable line search.
///
/// The line search type parameter `S` is the strategy
/// (e.g. [`Constant`], [`Backtracking`](crate::line_search::Backtracking),
/// [`Wolfe`](crate::line_search::Wolfe)). Use [`GradientDescent::new`]
/// for a fixed step or
/// [`GradientDescent::with_line_search`] to pick a strategy explicitly.
///
/// # Backends
///
/// Backend-generic — works with any `V` implementing
/// [`ScaledAdd<f64>`](crate::core::math::ScaledAdd) +
/// [`NegInPlace`] + `Clone`. That covers
/// `Vec<f64>`, `nalgebra::DVector<f64>` (feature `nalgebra`),
/// `ndarray::Array1<f64>` (feature `ndarray`), and `faer::Col<f64>`
/// (feature `faer`).
pub struct GradientDescent<S> {
    line_search: S,
}

impl GradientDescent<Constant> {
    /// Gradient descent with a fixed step size `alpha`. Equivalent to
    /// `with_line_search(Constant(alpha))`.
    pub fn new(alpha: f64) -> Self {
        Self {
            line_search: Constant(alpha),
        }
    }
}

impl<S> GradientDescent<S> {
    /// Gradient descent with an explicit line-search strategy
    /// (e.g. [`Backtracking`](crate::line_search::Backtracking),
    /// [`Wolfe`](crate::line_search::Wolfe)).
    pub fn with_line_search(line_search: S) -> Self {
        Self { line_search }
    }
}

impl<P, V, S> Solver<P, BasicState<V>> for GradientDescent<S>
where
    P: CostFunction<Param = V, Output = f64> + Gradient<Param = V, Gradient = V>,
    V: ScaledAdd<f64> + NegInPlace + Clone,
    S: LineSearch<P, V>,
{
    fn init(&mut self, problem: &P, mut state: BasicState<V>) -> BasicState<V> {
        // Seed cost and gradient at the initial param so iter-0 termination
        // checks (e.g. `GradientTolerance` on a near-optimal start) see a
        // complete state. Same work we'd do on iter 1, hoisted.
        state.cost = Some(problem.cost(&state.param));
        state.gradient = Some(problem.gradient(&state.param));
        state.cost_evals += 1;
        state.gradient_evals += 1;
        state
    }

    fn next_iter(
        &mut self,
        problem: &P,
        mut state: BasicState<V>,
    ) -> (BasicState<V>, Option<TerminationReason>) {
        let grad = state
            .gradient
            .take()
            .expect("gradient not set: Solver::init must run before next_iter");
        let prev_cost = state
            .cost
            .expect("cost not set: Solver::init must run before next_iter");
        let mut direction = grad.clone();
        direction.neg_in_place();
        let step = self
            .line_search
            .next(problem, &state.param, prev_cost, &grad, &direction);
        state.cost_evals += step.cost_evals;
        state.gradient_evals += step.gradient_evals;
        state.param.scaled_add(step.alpha, &direction);
        state.cost = Some(problem.cost(&state.param));
        state.gradient = Some(problem.gradient(&state.param));
        state.cost_evals += 1;
        state.gradient_evals += 1;
        (state, None)
    }
}
