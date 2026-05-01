pub mod backtracking;
pub mod wolfe;

pub use backtracking::Backtracking;
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
