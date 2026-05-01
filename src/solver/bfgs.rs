use nalgebra::{DMatrix, DVector};

use crate::core::problem::{CostFunction, Gradient};
use crate::core::solver::Solver;
use crate::core::state::QuasiNewtonState;
use crate::core::termination::TerminationReason;
use crate::line_search::{LineSearch, Wolfe};

/// BFGS quasi-Newton solver.
///
/// Maintains a dense inverse-Hessian approximation `H` updated by the
/// rank-2 BFGS formula. The search direction is `d = −H·∇f(x)`; the step
/// length is set by a configurable line search (default: strong Wolfe,
/// which is what guarantees `yᵀs > 0` so each update preserves positive
/// definiteness).
///
/// On the first accepted step we rescale `H ← (sᵀy / yᵀy)·I` (Nocedal &
/// Wright (6.20)) — cheap, large convergence improvement on poorly scaled
/// problems.
///
/// **Curvature failure (`yᵀs ≤ ε · |y| · |s|`):** the H update is skipped
/// for that iteration. Strong Wolfe with `c2 < 1` guarantees `yᵀs > 0` in
/// exact arithmetic, so this branch is a numerical safeguard, not the
/// primary path. (Damped BFGS / Powell's modification is overkill when
/// strong Wolfe is in place — see plan.)
///
/// # Backends
///
/// `nalgebra` only. Bound on `DVector<f64>` / `DMatrix<f64>`; using BFGS
/// with `Vec<f64>`, `ndarray`, or `faer` is a compile-time error per
/// tenet 5. Other backends are a follow-up PR.
pub struct BFGS<S = Wolfe> {
    line_search: S,
    epsilon: f64,
    /// Set to true on a no-progress step (line search returned α = 0,
    /// e.g. because the search direction wasn't a descent direction —
    /// happens at machine-precision optima). Read by `terminate` to halt
    /// instead of spinning forever.
    converged: bool,
}

impl Default for BFGS<Wolfe> {
    fn default() -> Self {
        Self::new()
    }
}

impl BFGS<Wolfe> {
    pub fn new() -> Self {
        Self {
            line_search: Wolfe::new(),
            epsilon: 1e-10,
            converged: false,
        }
    }
}

impl<S> BFGS<S> {
    pub fn with_line_search(line_search: S) -> Self {
        Self {
            line_search,
            epsilon: 1e-10,
            converged: false,
        }
    }

    /// Relative threshold for the curvature condition `yᵀs > ε · |y| · |s|`.
    /// Iterations where this fails skip the H update (rare with strong
    /// Wolfe). Default `1e-10`.
    pub fn epsilon(mut self, epsilon: f64) -> Self {
        assert!(epsilon >= 0.0, "epsilon must be ≥ 0");
        self.epsilon = epsilon;
        self
    }
}

impl<P, S> Solver<P, QuasiNewtonState<DVector<f64>, DMatrix<f64>>> for BFGS<S>
where
    P: CostFunction<Param = DVector<f64>, Output = f64>
        + Gradient<Param = DVector<f64>, Gradient = DVector<f64>>,
    S: LineSearch<P, DVector<f64>>,
{
    fn init(
        &mut self,
        problem: &P,
        mut state: QuasiNewtonState<DVector<f64>, DMatrix<f64>>,
    ) -> QuasiNewtonState<DVector<f64>, DMatrix<f64>> {
        state.cost = Some(problem.cost(&state.param));
        state.gradient = Some(problem.gradient(&state.param));
        state.cost_evals += 1;
        state.gradient_evals += 1;
        state
    }

    fn next_iter(
        &mut self,
        problem: &P,
        mut state: QuasiNewtonState<DVector<f64>, DMatrix<f64>>,
    ) -> QuasiNewtonState<DVector<f64>, DMatrix<f64>> {
        let g = state
            .gradient
            .take()
            .expect("gradient not set: Solver::init must run before next_iter");
        let cost_old = state
            .cost
            .expect("cost not set: Solver::init must run before next_iter");

        // Quasi-Newton direction: d = −H g. With H positive definite this
        // is automatically a descent direction (gᵀd = −gᵀHg < 0).
        let direction = -(&state.inverse_hessian * &g);

        let step = self
            .line_search
            .next(problem, &state.param, cost_old, &g, &direction);
        state.cost_evals += step.cost_evals;
        state.gradient_evals += step.gradient_evals;

        // Line search bailed (α = 0): direction wasn't descent, or we're
        // at numerical convergence. Restore gradient/cost so the state
        // stays consistent and signal converged for `terminate`. NaN
        // routes here too (`NaN > 0.0` is false).
        if !(step.alpha.is_finite() && step.alpha > 0.0) {
            state.gradient = Some(g);
            state.cost = Some(cost_old);
            self.converged = true;
            return state;
        }

        // s = α d, x ← x + s.
        let s = step.alpha * &direction;
        state.param += &s;

        let g_new = problem.gradient(&state.param);
        state.gradient_evals += 1;

        let y = &g_new - &g;
        let sy = s.dot(&y);
        let s_norm = s.norm();
        let y_norm = y.norm();

        if sy > self.epsilon * s_norm * y_norm {
            // Initial-Hessian rescaling: align H₀ with the local curvature
            // before applying the first BFGS update. Without this, the
            // identity-initialized H produces a unit step that's far too
            // large or small on poorly scaled problems.
            if !state.initial_scaling_done {
                let yy = y.dot(&y);
                if yy > 0.0 {
                    let scale = sy / yy;
                    let n = state.param.len();
                    state.inverse_hessian = DMatrix::identity(n, n) * scale;
                }
                state.initial_scaling_done = true;
            }

            let rho = 1.0 / sy;
            let hy = &state.inverse_hessian * &y;
            let yhy = y.dot(&hy);
            let coef = rho * (1.0 + rho * yhy);

            // H ← H + coef · s sᵀ − ρ · (s (Hy)ᵀ + (Hy) sᵀ).
            // Three rank-1 updates, all in place.
            state.inverse_hessian.ger(coef, &s, &s, 1.0);
            state.inverse_hessian.ger(-rho, &s, &hy, 1.0);
            state.inverse_hessian.ger(-rho, &hy, &s, 1.0);
        }
        // else: curvature failure (very rare with strong Wolfe). Skip the
        // H update; the line search still produced a descent step, so we
        // continue. If this persists, max_iter / GradientTolerance halt.

        state.cost = Some(problem.cost(&state.param));
        state.cost_evals += 1;
        state.gradient = Some(g_new);
        state
    }

    fn terminate(
        &self,
        _state: &QuasiNewtonState<DVector<f64>, DMatrix<f64>>,
    ) -> Option<TerminationReason> {
        if self.converged {
            Some(TerminationReason::SolverConverged)
        } else {
            None
        }
    }
}
