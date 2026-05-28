//! Line searches: produce a step size `α` along a caller-supplied descent
//! direction. Used by first-order solvers (gradient descent, BFGS).

/// Backtracking line search (Armijo-only).
pub mod backtracking;
/// Moré–Thuente line search (MINPACK-2 `dcsrch` + `dcstep`).
pub mod more_thuente;
/// Strong-Wolfe line search (Nocedal & Wright algorithms 3.5/3.6).
pub mod wolfe;

pub use backtracking::Backtracking;
pub use more_thuente::MoreThuente;
pub use wolfe::Wolfe;

/// Compute a step size `α` along a caller-supplied descent direction `d`.
///
/// Convention: `direction` is a *descent* direction (`gᵀd < 0`); the caller
/// applies `x_new = x + α d`. Solvers that descend along `−∇f` (e.g. plain
/// gradient descent) compute `d = −∇f` themselves and pass it in.
///
/// # Error type
///
/// `Error` is the hard-abort error the line search propagates — concrete
/// impls set `type Error = P::Error;` (with `P: CostFunction`) so the
/// user's typed problem error bubbles untouched through the solver out of
/// [`Executor::run`](crate::Executor::run). See the
/// [`problem`](crate::core::problem) module docs for the soft-reject /
/// hard-abort split.
pub trait LineSearch<P, V> {
    /// Hard-abort error type, mirroring the underlying problem's `Error`.
    type Error;

    /// Returns an `α` plus the number of cost / gradient evaluations spent
    /// finding it (callers add these to the state's counters). Returns
    /// `Err` if any inner `problem.cost` / `problem.gradient` call does.
    fn next(
        &mut self,
        problem: &P,
        param: &V,
        cost: f64,
        gradient: &V,
        direction: &V,
    ) -> Result<LineSearchResult, Self::Error>;
}

/// Outcome of a [`LineSearch::next`] call: the chosen step plus how much
/// of the executor's evaluation budget was spent finding it.
#[derive(Debug, Clone, Copy)]
pub struct LineSearchResult {
    /// The chosen step size.
    pub alpha: f64,
    /// Cost evaluations the line search consumed.
    pub cost_evals: u64,
    /// Gradient evaluations the line search consumed.
    pub gradient_evals: u64,
}

/// Constant step size — returns the wrapped `α` regardless of input.
/// Useful when the caller already knows a good fixed step.
pub struct Constant(pub f64);

impl Constant {
    /// Constant step of size `alpha`.
    pub fn new(alpha: f64) -> Self {
        Self(alpha)
    }
}

impl<P: crate::core::problem::CostFunction, V> LineSearch<P, V> for Constant {
    // `Constant` makes no problem calls and so could declare any error
    // type, but solver bounds expect `L::Error = P::Error`; matching here
    // means callers never need a conversion glue layer.
    type Error = P::Error;

    fn next(
        &mut self,
        _problem: &P,
        _param: &V,
        _cost: f64,
        _gradient: &V,
        _direction: &V,
    ) -> Result<LineSearchResult, Self::Error> {
        Ok(LineSearchResult {
            alpha: self.0,
            cost_evals: 0,
            gradient_evals: 0,
        })
    }
}
