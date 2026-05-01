use crate::core::math::{NormSquared, ScaledAdd};
use crate::core::problem::CostFunction;

pub trait LineSearch<P, V> {
    /// Compute a step size and report the number of cost evaluations spent
    /// finding it. Callers add `cost_evals` to their state's counter.
    fn next(&mut self, problem: &P, param: &V, cost: f64, gradient: &V) -> LineSearchResult;
}

#[derive(Debug, Clone, Copy)]
pub struct LineSearchResult {
    pub alpha: f64,
    pub cost_evals: u64,
}

pub struct Constant(pub f64);

impl Constant {
    pub fn new(alpha: f64) -> Self {
        Self(alpha)
    }
}

impl<P, V> LineSearch<P, V> for Constant {
    fn next(&mut self, _problem: &P, _param: &V, _cost: f64, _gradient: &V) -> LineSearchResult {
        LineSearchResult {
            alpha: self.0,
            cost_evals: 0,
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
    V: ScaledAdd<f64> + NormSquared + Clone,
{
    fn next(&mut self, problem: &P, param: &V, cost: f64, gradient: &V) -> LineSearchResult {
        // Armijo on direction d = -grad: f(x + α d) ≤ f(x) + c α (∇f · d).
        // With d = -grad, ∇f · d = -|grad|², so the threshold is f(x) - c α |grad|².
        let g_norm_sq = gradient.norm_squared();
        let mut alpha = self.alpha_init;
        let mut cost_evals = 0u64;
        for _ in 0..self.max_iter {
            let mut trial = param.clone();
            trial.scaled_add(-alpha, gradient);
            let trial_cost = problem.cost(&trial);
            cost_evals += 1;
            if trial_cost <= cost - self.c * alpha * g_norm_sq {
                return LineSearchResult { alpha, cost_evals };
            }
            alpha *= self.rho;
        }
        LineSearchResult { alpha, cost_evals }
    }
}
