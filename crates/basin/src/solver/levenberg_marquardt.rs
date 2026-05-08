use crate::core::math::{
    AddDiagonalInPlace, Dot, GramMatrix, LinearSolveSpd, MatTransposeVec, MaxDiagonal, NegInPlace,
    NormInfinity, NormSquared, ScaledAdd,
};
use crate::core::problem::{Jacobian, Residual};
use crate::core::solver::Solver;
use crate::core::state::BasicState;
use crate::core::termination::TerminationReason;

/// Levenberg-Marquardt solver for nonlinear least-squares problems
/// `min ½‖r(x)‖²`, with the Nielsen 1999 smooth μ-update.
///
/// Each iteration solves the damped normal equations
/// `(JᵀJ + μI) h = −Jᵀr` via Cholesky, then adapts the damping
/// parameter μ from the gain ratio
/// `ρ = (F(x) − F(x+h)) / (L(0) − L(h))` (Nielsen eq. 2.2). On a
/// successful step (ρ > 0) μ is reduced via the smooth cubic
/// `μ ← μ · max(1/3, 1 − (2ρ−1)³)`; on a failed step (ρ ≤ 0) μ grows
/// geometrically `μ ← μ·ν, ν ← 2ν` with ν initialized to 2 — Nielsen
/// shows this avoids the discontinuities of the classical
/// multiply/divide threshold rule and lands roughly 25 % fewer
/// iterations on average. See Nielsen, *Damping Parameter in
/// Marquardt's Method* (IMM-REP-1999-05) for the derivation and
/// Madsen, Nielsen, Tingleff (2004), *Methods for Non-Linear Least
/// Squares Problems*, §3.2.
///
/// Initial damping is `μ₀ = τ · max diag(J(x₀)ᵀ J(x₀))` (Nielsen eq.
/// 1.10) with τ in the range `[10⁻⁸, 1]` depending on closeness to
/// the optimum. Default `τ = 10⁻³` matches Nielsen's "moderate trust"
/// recommendation.
///
/// **Cholesky-on-(JᵀJ + μI) vs QR-on-stacked-system.** The damping
/// makes the SPD path strictly better-conditioned than pure
/// Gauss-Newton's `JᵀJ` — μ regularizes the rank deficiency that
/// makes GN fail. We stay on the SPD path because that's the only one
/// the [`linalg`](crate::core::math) tier exposes today, and the
/// regularization is sufficient for unconstrained LM.
/// QR-on-stacked-system (`[J; √μ I]`) is more robust to ill-conditioned
/// `J` near rank deficiency but adds a second factorization route to
/// the linalg surface; deferred until S6 (TRF), where rank-deficient
/// Jacobians and box constraints make QR materially better.
///
/// # Failure modes
///
/// - **Cholesky failure under bumped μ.** When the initial damping is
///   too small to make `JᵀJ + μI` SPD (effectively never, for any
///   sensible `JᵀJ` and finite μ), the inner damping loop bumps μ via
///   `μ := μ·ν, ν := 2ν` and retries. Default
///   [`max_inner_attempts`](Self::max_inner_attempts) is 50 — far more
///   than enough; in practice the first attempt succeeds. If the cap
///   is exhausted (μ overflowing to `inf`), the solver returns
///   [`TerminationReason::SolverFailed`].
/// - **Divergence on highly nonlinear / poorly initialized problems.**
///   The damping itself prevents divergent steps (failed steps are
///   rejected via the gain-ratio test), so divergence manifests as
///   μ growing without bound. Catch this with
///   [`MaxIter`](crate::core::termination::MaxIter) on the executor.
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
/// LM deliberately leaves `state.gradient = None` — the framework's
/// [`GradientTolerance`](crate::core::termination::GradientTolerance)
/// uses the L2 squared norm and is the wrong metric for NLLS, where
/// the canonical first-order test is the ∞-norm of `Jᵀr`. Same choice
/// as [`GaussNewton`](super::GaussNewton).
///
/// # Backends
///
/// LA-heavy: nalgebra (`DVector<f64>` / `DMatrix<f64>`) and faer
/// (`Col<f64>` / `Mat<f64>`) at the dense tier; nalgebra-sparse
/// (`DVector<f64>` / `CscMatrix<f64>`) and faer-sparse (`Col<f64>` /
/// `SparseColMat<usize, f64>`) at the sparse tier. `Vec<f64>` and
/// `ndarray::Array1<f64>` produce a compile-time error per tenet 5.
/// The sparse damping path requires the diagonal of `JᵀJ` to be in the
/// CSC pattern (always true when `J` has no zero columns); see
/// [`AddDiagonalInPlace`].
///
/// # State convention
///
/// `state.cost` carries the LM convention `½‖r‖²`, derived from the
/// residual the solver evaluates itself. The bound on `P` is
/// [`Residual`] + [`Jacobian`], not
/// [`CostFunction`](crate::core::problem::CostFunction); problems
/// whose user-facing `cost()` uses an unscaled `Σ rᵢ²` form will see
/// `state.cost()` differ from `problem.cost(state.param())` by a
/// factor of two. Both go to zero at the optimum, so cost-based
/// termination criteria are unaffected.
pub struct LevenbergMarquardt {
    tol_grad: f64,
    tau: f64,
    max_inner_attempts: u32,

    // Runtime state, populated by `init` and mutated by `next_iter`
    // through `&mut self`.
    mu: Option<f64>,
    nu: f64,
}

impl Default for LevenbergMarquardt {
    fn default() -> Self {
        Self::new()
    }
}

impl LevenbergMarquardt {
    /// Levenberg-Marquardt with Nielsen's defaults: `tol_grad = 1e-8`,
    /// `tau = 1e-3`, `max_inner_attempts = 50`.
    pub fn new() -> Self {
        Self {
            tol_grad: 1e-8,
            tau: 1e-3,
            max_inner_attempts: 50,
            mu: None,
            nu: 2.0,
        }
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

    /// Initial damping scale `τ` in `μ₀ = τ · max diag(J(x₀)ᵀ J(x₀))`
    /// (Nielsen eq. 1.10). Use a smaller value (e.g. `1e-6`) when `x₀`
    /// is believed close to the optimum; a larger value (e.g. `1.0`)
    /// when far from it. Default `1e-3`.
    pub fn tau(mut self, tau: f64) -> Self {
        assert!(tau > 0.0, "tau must be > 0");
        self.tau = tau;
        self
    }

    /// Maximum number of damping bumps inside a single outer iteration
    /// before giving up with [`TerminationReason::SolverFailed`]. Each
    /// bump multiplies μ by ν (initially 2) and doubles ν. With the
    /// default 50, μ grows by a factor of `2^50 ≈ 10¹⁵` before bailing
    /// — effectively unreachable in practice. Default `50`.
    pub fn max_inner_attempts(mut self, n: u32) -> Self {
        assert!(n > 0, "max_inner_attempts must be > 0");
        self.max_inner_attempts = n;
        self
    }
}

impl<P, V, M> Solver<P, BasicState<V>> for LevenbergMarquardt
where
    P: Residual<Param = V, Output = V> + Jacobian<Param = V, Output = M>,
    V: ScaledAdd<f64> + NormSquared + NormInfinity + NegInPlace + Dot + Clone,
    M: GramMatrix
        + MatTransposeVec<V>
        + LinearSolveSpd<V>
        + AddDiagonalInPlace
        + MaxDiagonal
        + Clone,
{
    fn init(&mut self, problem: &P, mut state: BasicState<V>) -> BasicState<V> {
        // Seed cost so iter-0 termination criteria see a populated
        // state. Also evaluate J(x₀) once to seed μ₀ via Nielsen eq.
        // 1.10. The Jacobian is recomputed in the first next_iter —
        // caching it across the init/iter-0 boundary would add an
        // Option<M> cache field that's not worth its complexity for
        // one saved evaluation.
        let r = problem.residual(&state.param);
        let j = problem.jacobian(&state.param);
        state.cost = Some(0.5 * r.norm_squared());
        state.cost_evals += 1;
        state.gradient_evals += 1;

        // μ₀ = τ · max diag(JᵀJ). The diagonal extraction goes through
        // MaxDiagonal so sparse and dense backends share the same
        // path. If max diag is non-positive (degenerate J with zero
        // columns), fall back to τ alone — Nielsen's recommendation
        // assumes the diagonal scaling is meaningful.
        let gram = j.gram();
        let max_diag = gram.max_diagonal().max(1.0);
        self.mu = Some(self.tau * max_diag);
        self.nu = 2.0;
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
        // (Madsen et al. eq. 3.3a) is the canonical NLLS test.
        let g = j.mat_transpose_vec(&r);
        if self.tol_grad > 0.0 && g.norm_infinity() <= self.tol_grad {
            return (state, Some(TerminationReason::SolverConverged));
        }

        let mut neg_g = g.clone();
        neg_g.neg_in_place();
        let a = j.gram();

        let mut mu = self
            .mu
            .expect("mu not set: Solver::init must run before next_iter");
        let mut nu = self.nu;

        // Inner damping loop: bump μ on Cholesky failure. In practice
        // the first attempt succeeds — a properly damped (JᵀJ + μI) is
        // SPD by construction. The retry path matters only for
        // pathological cases where the initial μ is too small to
        // overcome arithmetic roundoff.
        let h;
        let mut attempts: u32 = 0;
        loop {
            let mut a_damped = a.clone();
            a_damped.add_diagonal_in_place(mu);
            match a_damped.solve_spd(&neg_g) {
                Ok(step) => {
                    h = step;
                    break;
                }
                Err(_) => {
                    attempts += 1;
                    if attempts >= self.max_inner_attempts || !mu.is_finite() {
                        self.mu = Some(mu);
                        self.nu = nu;
                        return (state, Some(TerminationReason::SolverFailed));
                    }
                    mu *= nu;
                    nu *= 2.0;
                }
            }
        }

        // L(0) − L(h) = ½ hᵀ(μh − g) (Nielsen eq. 2.3). Both terms in
        // the parenthesis make the predicted reduction positive: μh⊤h
        // > 0 and h points opposite to g (descent direction), so
        // -h⊤g > 0. Compute scalar form to avoid materializing
        // (μh − g).
        let l_diff = 0.5 * (mu * h.norm_squared() - h.dot(&g));

        // Trial step.
        let mut x_trial = state.param.clone();
        x_trial.scaled_add(1.0, &h);
        let r_trial = problem.residual(&x_trial);
        state.cost_evals += 1;
        let f_trial = 0.5 * r_trial.norm_squared();

        let prev_cost = state
            .cost
            .expect("cost not set: Solver::init must run before next_iter");
        let actual_diff = prev_cost - f_trial;
        let rho = if l_diff > 0.0 {
            actual_diff / l_diff
        } else {
            0.0
        };

        if rho > 0.0 {
            // Accept. Update x and cost; adapt μ via Nielsen eq. 2.5
            // with β=2, γ=3, p=3.
            state.param = x_trial;
            state.cost = Some(f_trial);
            let factor = 1.0 - (2.0 * rho - 1.0).powi(3);
            mu *= factor.max(1.0 / 3.0);
            nu = 2.0;
        } else {
            // Reject. Keep state; bump μ geometrically and double ν so
            // consecutive failures escalate damping faster.
            mu *= nu;
            nu *= 2.0;
        }

        self.mu = Some(mu);
        self.nu = nu;
        (state, None)
    }
}
