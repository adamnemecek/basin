use crate::core::math::{Dot, ScaledAdd};
use crate::core::problem::CostFunction;

pub mod wolfe;

pub use wolfe::Wolfe;

/// Compute a step size `α` along a caller-supplied descent direction `d`.
///
/// Convention: `direction` is a *descent* direction (`gᵀd < 0`); the caller
/// applies `x_new = x + α d`. Solvers that descend along `−∇f` (e.g. plain
/// gradient descent) compute `d = −∇f` themselves and pass it in.
pub trait LineSearch<P, V> {
    /// Returns an `α` plus the number of cost / gradient evaluations spent
    /// finding it (callers add these to the state's counters).
    fn next(
        &mut self,
        problem: &P,
        param: &V,
        cost: f64,
        gradient: &V,
        direction: &V,
    ) -> LineSearchResult;
}

#[derive(Debug, Clone, Copy)]
pub struct LineSearchResult {
    pub alpha: f64,
    pub cost_evals: u64,
    pub gradient_evals: u64,
}

pub struct Constant(pub f64);

impl Constant {
    pub fn new(alpha: f64) -> Self {
        Self(alpha)
    }
}

impl<P, V> LineSearch<P, V> for Constant {
    fn next(
        &mut self,
        _problem: &P,
        _param: &V,
        _cost: f64,
        _gradient: &V,
        _direction: &V,
    ) -> LineSearchResult {
        LineSearchResult {
            alpha: self.0,
            cost_evals: 0,
            gradient_evals: 0,
        }
    }
}

pub struct Backtracking {
    pub alpha_init: f64,
    pub rho: f64,
    pub c: f64,
    pub max_iter: u32,
}

impl Default for Backtracking {
    fn default() -> Self {
        Self {
            alpha_init: 1.0,
            rho: 0.5,
            c: 1e-4,
            max_iter: 50,
        }
    }
}

impl Backtracking {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn alpha_init(mut self, alpha_init: f64) -> Self {
        self.alpha_init = alpha_init;
        self
    }

    pub fn rho(mut self, rho: f64) -> Self {
        self.rho = rho;
        self
    }

    pub fn c(mut self, c: f64) -> Self {
        self.c = c;
        self
    }

    pub fn max_iter(mut self, max_iter: u32) -> Self {
        self.max_iter = max_iter;
        self
    }
}

impl<P, V> LineSearch<P, V> for Backtracking
where
    P: CostFunction<Param = V, Output = f64>,
    V: ScaledAdd<f64> + Dot + Clone,
{
    fn next(
        &mut self,
        problem: &P,
        param: &V,
        cost: f64,
        gradient: &V,
        direction: &V,
    ) -> LineSearchResult {
        // Armijo: f(x + α d) ≤ f(x) + c α (∇f · d). For a descent direction,
        // `g_dot_d` is negative, so the threshold drops with α.
        let g_dot_d = gradient.dot(direction);
        let mut alpha = self.alpha_init;
        let mut cost_evals = 0u64;
        for _ in 0..self.max_iter {
            let mut trial = param.clone();
            trial.scaled_add(alpha, direction);
            let trial_cost = problem.cost(&trial);
            cost_evals += 1;
            if trial_cost <= cost + self.c * alpha * g_dot_d {
                return LineSearchResult {
                    alpha,
                    cost_evals,
                    gradient_evals: 0,
                };
            }
            alpha *= self.rho;
        }
        LineSearchResult {
            alpha,
            cost_evals,
            gradient_evals: 0,
        }
    }
}
