use crate::core::math::{
    AddDiagonalVectorInPlace, ComponentMaxAssign, ComponentMulAssign, Dot, FloorZerosInPlace,
    GramMatrix, LinearSolveSpd, MatDiagonal, MatTransposeVec, NegInPlace, NormInfinity,
    NormSquared, ScaleInPlace, ScaledAdd,
};
use crate::core::problem::{Jacobian, Residual};
use crate::core::solver::Solver;
use crate::core::state::BasicState;
use crate::core::termination::TerminationReason;

/// Levenberg-Marquardt solver for nonlinear least-squares problems
/// `min ½‖r(x)‖²`, with Marquardt diagonal scaling and the Nielsen
/// 1999 smooth μ-update.
///
/// Each iteration solves the damped normal equations
/// `(JᵀJ + μ·D) h = −Jᵀr` via Cholesky, then adapts the damping
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
/// **Marquardt scaling (`μ·D`, not `μI`).** The damping matrix is the
/// diagonal of the Gram, `D = diag(JᵀJ)` — the per-parameter curvature
/// — rather than the identity. This makes the trust region ellipsoidal
/// in the metric of the columns of `J`, so the algorithm is invariant
/// to diagonal rescaling of the parameters (Marquardt 1963; Moré 1978,
/// *The Levenberg-Marquardt Algorithm: Implementation and Theory*).
/// Isotropic `μI` damping over-damps well-scaled directions and
/// under-damps poorly-scaled ones when the columns of `J` have very
/// different norms (e.g. parameters in a mixed log/linear/angle
/// encoding), which biases the step and can pull the iterate into a
/// worse basin. `D` is maintained as a **monotone running max**
/// `D_k = max(D_{k−1}, diag(J(x_k)ᵀJ(x_k)))` so a column whose
/// curvature momentarily drops keeps the damping floor it earned
/// earlier (Moré 1978; the same safeguard MINPACK applies to its
/// column-norm scaling). Columns that are exactly zero at `x₀` (a
/// parameter with no first-order effect on any residual) would make
/// `μ·D` vanish there and the Gram singular; following MINPACK, their
/// scale is floored to `1` at `init` (see [`FloorZerosInPlace`]), so a
/// fully-insensitive parameter stays put rather than failing Cholesky.
///
/// Initial damping is `μ₀ = τ` — dimensionless, because the
/// per-parameter magnitude now lives in `D` (the initial per-column
/// damping is `τ·diag(J(x₀)ᵀJ(x₀))`). τ is the *relative* trust
/// parameter; use a smaller value (e.g. `1e-6`) when `x₀` is believed
/// close to the optimum, larger (e.g. `1.0`) when far. Default
/// `τ = 10⁻³` matches Nielsen's "moderate trust" recommendation.
///
/// **Cholesky-on-(JᵀJ + μ·D) vs QR-on-stacked-system.** The damping
/// makes the SPD path strictly better-conditioned than pure
/// Gauss-Newton's `JᵀJ` — `μ·D` regularizes the rank deficiency that
/// makes GN fail. We stay on the SPD path because that's the only one
/// the [`linalg`](crate::core::math) tier exposes today, and the
/// regularization is sufficient for unconstrained LM.
/// QR-on-stacked-system (`[J; √(μD)]`) is more robust to ill-conditioned
/// `J` near rank deficiency but adds a second factorization route to
/// the linalg surface; deferred until S6 (TRF), where rank-deficient
/// Jacobians and box constraints make QR materially better.
///
/// # Failure modes
///
/// - **Cholesky failure under bumped μ.** When the initial damping is
///   too small to make `JᵀJ + μ·D` SPD (effectively never, for any
///   sensible `JᵀJ` and finite μ), the inner damping loop bumps μ via
///   `μ := μ·ν, ν := 2ν` and retries. Default
///   [`max_inner_attempts`](Self::max_inner_attempts) is 50 — far more
///   than enough; in practice the first attempt succeeds. If the cap
///   is exhausted (μ overflowing to `inf`), the solver returns
///   [`TerminationReason::SolverFailed`]. Note that bumping μ cannot
///   rescue a coordinate whose `D` entry is zero (`μ·0 = 0`); the
///   `init` zero-column floor exists precisely to keep `D > 0`.
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
/// [`AddDiagonalVectorInPlace`] and [`MatDiagonal`].
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
pub struct LevenbergMarquardt<V, M> {
    tol_grad: f64,
    tau: f64,
    max_inner_attempts: u32,

    // Runtime state, populated by `init` and mutated by `next_iter`
    // through `&mut self`.
    mu: Option<f64>,
    nu: f64,

    // Marquardt scaling diagonal `D` — the monotone running max of
    // `diag(JᵀJ)` (Moré 1978). Seeded at `init` from `diag(J(x₀)ᵀJ(x₀))`
    // with zero columns floored to 1, then maxed against each
    // iteration's Gram diagonal. The damped system is `(JᵀJ + μ·D) h =
    // −Jᵀr`.
    diag: Option<V>,

    // Residual and Jacobian caches across iterations. `r_cache` is
    // refreshed whenever the iterate moves (to `r(x_trial)` on accept,
    // to the unchanged old `r` on reject); `j_cache` is preserved on
    // reject but cleared on accept, since `J(x_trial)` wasn't computed
    // in the gain-ratio test. Skipping these caches re-evaluates the
    // same `(r, J)` pair at the same point — Madsen-Nielsen Algorithm
    // 3.16 assigns `J` only after acceptance (line 13).
    r_cache: Option<V>,
    j_cache: Option<M>,
}

impl<V, M> Default for LevenbergMarquardt<V, M> {
    fn default() -> Self {
        Self::new()
    }
}

impl<V, M> LevenbergMarquardt<V, M> {
    /// Levenberg-Marquardt with Nielsen's defaults: `tol_grad = 1e-8`,
    /// `tau = 1e-3`, `max_inner_attempts = 50`.
    pub fn new() -> Self {
        Self {
            tol_grad: 1e-8,
            tau: 1e-3,
            max_inner_attempts: 50,
            mu: None,
            nu: 2.0,
            diag: None,
            r_cache: None,
            j_cache: None,
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

    /// Relative initial damping `τ`: `μ₀ = τ`, giving an initial
    /// per-column damping of `τ·diag(J(x₀)ᵀJ(x₀))` under Marquardt
    /// scaling. Use a smaller value (e.g. `1e-6`) when `x₀` is believed
    /// close to the optimum; a larger value (e.g. `1.0`) when far from
    /// it. Default `1e-3` (Nielsen's "moderate trust").
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

impl<P, V, M> Solver<P, BasicState<V>> for LevenbergMarquardt<V, M>
where
    P: Residual<Param = V, Output = V> + Jacobian<Param = V, Output = M>,
    V: ScaledAdd<f64>
        + NormSquared
        + NormInfinity
        + NegInPlace
        + Dot
        + ScaleInPlace
        + ComponentMulAssign
        + ComponentMaxAssign
        + FloorZerosInPlace
        + Clone,
    M: GramMatrix
        + MatTransposeVec<V>
        + LinearSolveSpd<V>
        + AddDiagonalVectorInPlace<V>
        + MatDiagonal<V>
        + Clone,
{
    fn init(&mut self, problem: &P, mut state: BasicState<V>) -> BasicState<V> {
        // Seed cost so iter-0 termination criteria see a populated
        // state. Also evaluate J(x₀) once to seed the Marquardt scaling
        // diagonal `D`. `r` and `J(x₀)` are stashed into the caches so
        // the first `next_iter` reuses them — no redundant evaluation
        // at the init/iter-0 boundary.
        let r = problem.residual(&state.param);
        let j = problem.jacobian(&state.param);
        state.cost = Some(0.5 * r.norm_squared());
        state.cost_evals += 1;
        state.gradient_evals += 1;

        // D₀ = diag(J(x₀)ᵀJ(x₀)), the per-parameter curvature. A column
        // that's exactly zero at x₀ contributes 0 here, which would
        // make `μ·D` vanish on that coordinate and the Gram singular;
        // following MINPACK we floor those to 1 so an insensitive
        // parameter simply doesn't move. The running max in `next_iter`
        // then keeps `D` monotone.
        let mut d = j.gram().diagonal();
        d.floor_zeros_in_place(1.0);
        self.diag = Some(d);

        // μ₀ = τ. Dimensionless: the per-parameter magnitude lives in
        // `D`, so the initial per-column damping is `τ·diag(J(x₀)ᵀJ(x₀))`.
        self.mu = Some(self.tau);
        self.nu = 2.0;
        self.r_cache = Some(r);
        self.j_cache = Some(j);
        state
    }

    fn next_iter(
        &mut self,
        problem: &P,
        mut state: BasicState<V>,
    ) -> (BasicState<V>, Option<TerminationReason>) {
        // Use cached `r` / `J` when available — they're at the current
        // `state.param` after either init (initial point) or the
        // previous iteration's bookkeeping (post-accept: r at the new
        // iterate; post-reject: both unchanged). Only count an eval
        // when the cache misses, so `cost_evals` / `gradient_evals`
        // grow with *actual* problem invocations.
        let r = match self.r_cache.take() {
            Some(r) => r,
            None => {
                state.cost_evals += 1;
                problem.residual(&state.param)
            }
        };
        let j = match self.j_cache.take() {
            Some(j) => j,
            None => {
                state.gradient_evals += 1;
                problem.jacobian(&state.param)
            }
        };

        // g = Jᵀr is the gradient of ½‖r‖². First-order optimality
        // (Madsen et al. eq. 3.3a) is the canonical NLLS test.
        let g = j.mat_transpose_vec(&r);
        if self.tol_grad > 0.0 && g.norm_infinity() <= self.tol_grad {
            // Restore the caches so a subsequent `run()` (e.g. via
            // `InnerExecutor`) doesn't see corrupted state — though
            // in practice `init` resets them on each reuse.
            self.r_cache = Some(r);
            self.j_cache = Some(j);
            return (state, Some(TerminationReason::SolverConverged));
        }

        let mut neg_g = g.clone();
        neg_g.neg_in_place();
        let a = j.gram();

        // Marquardt scaling: maintain `D` as the monotone running max of
        // `diag(JᵀJ)` (Moré 1978). `D` was floored away from zero at
        // `init`, and the max only grows entries, so it stays strictly
        // positive — the damped Gram below is SPD by construction.
        let mut d = self
            .diag
            .take()
            .expect("diag not set: Solver::init must run before next_iter");
        d.component_max_assign(&a.diagonal());

        let mut mu = self
            .mu
            .expect("mu not set: Solver::init must run before next_iter");
        let mut nu = self.nu;

        // Inner damping loop: bump μ on Cholesky failure. In practice
        // the first attempt succeeds — a properly damped (JᵀJ + μ·D) is
        // SPD by construction. The retry path matters only for
        // pathological cases where the initial μ is too small to
        // overcome arithmetic roundoff.
        let h;
        let mut attempts: u32 = 0;
        loop {
            let mut a_damped = a.clone();
            // damping = μ·D, added to the Gram diagonal.
            let mut damping = d.clone();
            damping.scale_in_place(mu);
            a_damped.add_diagonal_vector_in_place(&damping);
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
                        self.diag = Some(d);
                        // State unchanged; both caches still valid at
                        // the current iterate.
                        self.r_cache = Some(r);
                        self.j_cache = Some(j);
                        return (state, Some(TerminationReason::SolverFailed));
                    }
                    mu *= nu;
                    nu *= 2.0;
                }
            }
        }

        // L(0) − L(h) = ½ hᵀ(μ·D·h − g) = ½(μ·hᵀD h − hᵀg) (Nielsen eq.
        // 2.3, with the scaling diagonal D folded into the quadratic
        // term — `μI` is the D = I special case). Both terms make the
        // predicted reduction positive: μ·hᵀD h > 0 since D > 0, and
        // −hᵀg > 0 since h is a descent direction. Form hᵀD h as
        // h·(D ⊙ h) to avoid materializing μ·D·h − g.
        let mut dh = d.clone();
        dh.component_mul_assign(&h);
        let l_diff = 0.5 * (mu * h.dot(&dh) - h.dot(&g));

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
            // with β=2, γ=3, p=3. The trial residual is at the new
            // iterate — stash it; the Jacobian at the new iterate
            // hasn't been computed, so leave `j_cache` empty.
            state.param = x_trial;
            state.cost = Some(f_trial);
            let factor = 1.0 - (2.0 * rho - 1.0).powi(3);
            mu *= factor.max(1.0 / 3.0);
            nu = 2.0;
            self.r_cache = Some(r_trial);
            self.j_cache = None;
        } else {
            // Reject. Keep state; bump μ geometrically and double ν so
            // consecutive failures escalate damping faster. Both r and
            // J remain valid at the unchanged iterate.
            mu *= nu;
            nu *= 2.0;
            self.r_cache = Some(r);
            self.j_cache = Some(j);
        }

        self.mu = Some(mu);
        self.nu = nu;
        // `d` is unchanged by accept/reject (the running max happens
        // once, above) — persist it for the next iteration.
        self.diag = Some(d);
        (state, None)
    }
}
