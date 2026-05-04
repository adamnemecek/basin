use crate::core::math::{
    GramMatrix, LinearSolveSpd, MatTransposeVec, NegInPlace, NormInfinity, NormSquared, ScaledAdd,
};
use crate::core::problem::{Jacobian, Residual};
use crate::core::solver::Solver;
use crate::core::state::BasicState;
use crate::core::termination::TerminationReason;

/// Pure Gauss-Newton solver for nonlinear least-squares problems
/// `min ½‖r(x)‖²`.
///
/// Each iteration solves the normal equations `(JᵀJ) δ = −Jᵀr` via
/// Cholesky on the Gram matrix `JᵀJ` and takes the full step
/// `x ← x + δ`. No damping, no line search — that's what
/// Levenberg-Marquardt is for. See Madsen, Nielsen, Tingleff (2004),
/// *Methods for Non-Linear Least Squares Problems*, §3.1.
///
/// **Cholesky-on-`JᵀJ` vs QR-on-`J`.** Cholesky on the Gram is the
/// simple path and the only one the
/// [`linalg`](crate::core::math) tier exposes today. It squares the
/// condition number of `J` and fails noisily on rank-deficient `J` —
/// see the [`solve_spd` failure path](#failure-modes) below. QR-on-`J`
/// is more numerically robust but adds a second factorization to the
/// linalg surface; deferred until a solver actually needs it. Pure GN
/// is the right vehicle for Cholesky: when `J` is so ill-conditioned
/// that QR matters in practice, you wanted LM (or TRF) anyway.
///
/// # Failure modes
///
/// - **Rank-deficient `J`** (`JᵀJ` not positive definite) → the
///   Cholesky inside [`LinearSolveSpd`] returns
///   [`NotPositiveDefinite`](crate::core::math::LinearSolveError::NotPositiveDefinite),
///   and the solver returns [`TerminationReason::SolverFailed`]. This
///   is the *correct* behavior for pure GN — Powell's singular
///   function is the canonical example. Reach for Levenberg-Marquardt
///   when this fires.
/// - **Divergence on highly nonlinear / poorly initialized problems.**
///   No safeguard here either; pure GN trusts the linear model. Catch
///   this with a finite [`MaxIter`](crate::core::termination::MaxIter)
///   or [`CostTolerance`](crate::core::termination::CostTolerance) on
///   the executor.
///
/// # Termination
///
/// Beyond the framework criteria
/// ([`MaxIter`](crate::core::termination::MaxIter),
/// [`CostTolerance`](crate::core::termination::CostTolerance),
/// [`ParamTolerance`](crate::core::termination::ParamTolerance), …),
/// the solver emits [`TerminationReason::SolverConverged`] when the
/// first-order optimality measure
/// `‖Jᵀr‖_∞ ≤ tol_grad` (Madsen et al. eq. 3.3a) is satisfied.
/// Default `tol_grad = 1e-8`; set to `0.0` to disable the check.
///
/// # Backends
///
/// LA-heavy: nalgebra (`DVector<f64>` / `DMatrix<f64>`) and faer
/// (`Col<f64>` / `Mat<f64>`). `Vec<f64>` and `ndarray::Array1<f64>`
/// produce a compile-time error per tenet 5 — neither has an honest
/// [`Jacobian`] impl. Sparse `Jacobian::Output` types land in S2b and
/// satisfy the same bound set with no solver-side change.
///
/// # State convention
///
/// `state.cost` carries the LM convention `½‖r‖²`, derived from the
/// residual the solver computes itself. The bound on `P` is
/// [`Residual`] + [`Jacobian`], not [`CostFunction`](crate::core::problem::CostFunction);
/// problems whose user-facing `cost()` uses an unscaled `Σ rᵢ²` form
/// (e.g. Rosenbrock-as-residuals) will see `state.cost()` differ from
/// `problem.cost(state.param())` by a factor of two. Both go to zero
/// at the optimum, so cost-based termination criteria are unaffected.
pub struct GaussNewton {
    tol_grad: f64,
}

impl Default for GaussNewton {
    fn default() -> Self {
        Self::new()
    }
}

impl GaussNewton {
    /// Pure Gauss-Newton with the default first-order optimality
    /// tolerance (`tol_grad = 1e-8`).
    pub fn new() -> Self {
        Self { tol_grad: 1e-8 }
    }

    /// First-order optimality tolerance: emit
    /// [`TerminationReason::SolverConverged`] when `‖Jᵀr‖_∞ ≤ tol`.
    /// Set to `0.0` to disable the check and rely solely on framework
    /// termination criteria. Default `1e-8`.
    pub fn tol_grad(mut self, tol: f64) -> Self {
        assert!(tol >= 0.0, "tol_grad must be ≥ 0");
        self.tol_grad = tol;
        self
    }
}

impl<P, V, M> Solver<P, BasicState<V>> for GaussNewton
where
    P: Residual<Param = V, Output = V> + Jacobian<Param = V, Output = M>,
    V: ScaledAdd<f64> + NormSquared + NormInfinity + NegInPlace + Clone,
    M: GramMatrix + MatTransposeVec<V> + LinearSolveSpd<V>,
{
    fn init(&mut self, problem: &P, mut state: BasicState<V>) -> BasicState<V> {
        // Seed cost so iter-0 termination criteria see a populated state.
        // Jacobian is recomputed in next_iter — caching it across iterates
        // would bloat BasicState for one eval saved at iter 0.
        let r = problem.residual(&state.param);
        state.cost = Some(0.5 * r.norm_squared());
        state.cost_evals += 1;
        state
    }

    fn next_iter(
        &mut self,
        problem: &P,
        mut state: BasicState<V>,
    ) -> (BasicState<V>, Option<TerminationReason>) {
        let r = problem.residual(&state.param);
        let j = problem.jacobian(&state.param);
        state.cost_evals += 1;
        state.gradient_evals += 1;

        // g = Jᵀr is the gradient of ½‖r‖². First-order optimality
        // (Madsen/Nielsen/Tingleff eq. 3.3a) is the canonical NLLS
        // convergence test.
        let g = j.mat_transpose_vec(&r);
        if self.tol_grad > 0.0 && g.norm_infinity() <= self.tol_grad {
            return (state, Some(TerminationReason::SolverConverged));
        }

        // Solve (JᵀJ) δ = −Jᵀr. Cholesky failure means JᵀJ is not
        // positive definite (rank-deficient J) — pure GN can't recover,
        // LM in S4 will.
        let gram = j.gram();
        let mut neg_g = g;
        neg_g.neg_in_place();
        let delta = match gram.solve_spd(&neg_g) {
            Ok(d) => d,
            Err(_) => return (state, Some(TerminationReason::SolverFailed)),
        };

        // Full GN step. Refresh state.cost from a fresh residual so the
        // post-iteration state is consistent (Solver contract).
        state.param.scaled_add(1.0, &delta);
        let r_new = problem.residual(&state.param);
        state.cost = Some(0.5 * r_new.norm_squared());
        state.cost_evals += 1;

        (state, None)
    }
}
