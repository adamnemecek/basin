use crate::core::executor::OptimizationResult;
use crate::core::inner::InnerExecutor;
use crate::core::math::{
    ComponentMulAssign, MatTransposeVec, MatVec, MatrixIdentity, NormSquared, RankOneUpdate,
    SampleStandardNormal, ScaleInPlace, ScaledAdd, SymmetricEigen, VectorLen,
};
use crate::core::problem::CostFunction;
use crate::core::solver::Solver;
use crate::core::state::{BasicPopulationState, BasicSimplexState, State};
use crate::core::termination::{TerminationCriterion, TerminationReason};
use crate::solver::cma_es::{sort_population_ascending, CmaEs};
use crate::solver::nelder_mead::NelderMead;

/// Memetic CMA-ES with Hansen (2011) injection: outer CMA-ES proposes
/// `λ` candidates per generation, an inner Nelder-Mead refines the best
/// `k`, and the refined points are injected back into the population
/// for the *next* CMA update.
///
/// The only departure from the standard
/// [`CmaEs`](crate::solver::cma_es::CmaEs) update is clipping each
/// injected point's normalised step in Mahalanobis distance:
///
/// ```text
///   y_i ← min(1, c_y / ‖C^{-1/2} y_i‖) · y_i        (Hansen 2011 eq. 4)
///   c_y = √n + 2n/(n+2)                              (Table 1 default)
/// ```
///
/// with `y_i = (x_i − m)/σ` and `C^{-1/2} = B D^{-1} Bᵀ` from the
/// post-update eigendecomposition CMA-ES already maintains. After
/// clipping, replaced candidates re-enter the population on equal
/// footing with regular samples — all subsequent CMA updates
/// (m, p_σ, p_c, C, σ) run the standard equations unchanged. Lamarckian
/// by construction; no Baldwinian mode in the paper.
///
/// # Inner solver
///
/// This first cut hard-wires the inner to [`NelderMead`] over
/// [`BasicSimplexState<V>`], the empirical winner from
/// Melo & Iacca 2014 on 5/9 of their constrained problems. Genericity
/// over the inner solver type (`CmaInject<I: Solver>`) is deferred to
/// the second concrete inner (e.g. L-BFGS-B once that solver lands) —
/// designing the "seed inner state from a candidate" abstraction
/// against a single consumer is premature (AGENTS.md tenet 4 spirit).
///
/// # Backends
///
/// Same coverage as [`CmaEs`]: nalgebra (`DVector` / `DMatrix`) and
/// faer (`Col` / `Mat`). `Vec<f64>` and `ndarray` produce a
/// compile-time error per tenet 5 (no honest matrix type or no
/// pure-Rust eigendecomposition).
pub struct CmaInject<V, M> {
    cma: CmaEs<V, M>,
    inner: InnerExecutor<BasicSimplexState<V>, NelderMead>,
    k: usize,
    c_y_override: Option<f64>,
    /// Edge length of the seed simplex is `inner_simplex_scale · σ`.
    /// Default `1.0` ties the inner's exploration scale to the CMA
    /// distribution's current spread, so it shrinks with `σ`.
    inner_simplex_scale: f64,
}

impl<V, M> CmaInject<V, M> {
    /// Wrap a configured [`CmaEs`] with Hansen-2011 injection.
    ///
    /// Defaults: `k = 1` (one refined point per generation, matching
    /// Hansen's preliminary experiments — RR-7748 §4), inner
    /// `max_iter = 50`, `c_y = √n + 2n/(n+2)` (Hansen 2011 Table 1),
    /// `inner_simplex_scale = 1.0` (inner simplex edge = current σ).
    pub fn new(cma: CmaEs<V, M>) -> Self {
        Self {
            cma,
            inner: InnerExecutor::new(NelderMead::adaptive()).max_iter(50),
            k: 1,
            c_y_override: None,
            inner_simplex_scale: 1.0,
        }
    }

    /// Number of best-ranked candidates to refine and inject each
    /// generation. Default `1`.
    ///
    /// # Panics
    ///
    /// Panics if `k == 0` (nothing to inject) or `k > λ` is not
    /// caught here — exceeding `λ` is silently clamped at runtime to
    /// the population size.
    pub fn with_k(mut self, k: usize) -> Self {
        assert!(k >= 1, "CmaInject requires k >= 1, got {}", k);
        self.k = k;
        self
    }

    /// Override the Hansen-2011 clipping threshold `c_y` (default
    /// `√n + 2n/(n+2)`). Larger values disable clipping for further
    /// injected points; `c_y = ∞` recovers the no-clipping limit.
    ///
    /// # Panics
    ///
    /// Panics if `c_y <= 0`.
    pub fn with_c_y(mut self, c_y: f64) -> Self {
        assert!(c_y > 0.0, "CmaInject requires c_y > 0, got {}", c_y);
        self.c_y_override = Some(c_y);
        self
    }

    /// Inner Nelder-Mead iteration budget per outer generation
    /// (default `50`).
    pub fn with_inner_max_iter(mut self, n: u64) -> Self {
        // `InnerExecutor::max_iter` is a consuming builder; swap
        // through a throwaway placeholder so any criteria already
        // registered via `inner_terminate_on` carry through.
        let prior = std::mem::replace(&mut self.inner, InnerExecutor::new(NelderMead::adaptive()));
        self.inner = prior.max_iter(n);
        self
    }

    /// Edge length of the inner Nelder-Mead seed simplex as a multiple
    /// of the current CMA step size `σ` (default `1.0`). The simplex
    /// vertices are `x_i + scale · σ · e_j` for `j = 1..=n`.
    ///
    /// # Panics
    ///
    /// Panics if `scale <= 0`.
    pub fn with_inner_simplex_scale(mut self, scale: f64) -> Self {
        assert!(
            scale > 0.0,
            "CmaInject requires inner_simplex_scale > 0, got {}",
            scale
        );
        self.inner_simplex_scale = scale;
        self
    }

    /// Register a termination criterion on the inner Nelder-Mead loop.
    /// Criteria are reused across every outer iteration's inner run,
    /// so they MUST be stateless across calls
    /// — `MaxIter`, `CostTolerance`, `ParamTolerance`, and
    /// `SimplexTolerance` are safe;
    /// [`MaxTime`](crate::core::termination::MaxTime) is **not** (its
    /// internal `start` instant carries over and fires prematurely on
    /// later runs). See `AGENTS.md` "Solver composition" rule 2.
    pub fn inner_terminate_on<C>(mut self, criterion: C) -> Self
    where
        C: TerminationCriterion<BasicSimplexState<V>> + 'static,
    {
        let prior = std::mem::replace(&mut self.inner, InnerExecutor::new(NelderMead::adaptive()));
        self.inner = prior.terminate_on(criterion);
        self
    }
}

/// Hansen 2011 Table 1: `c_y = √n + 2n/(n+2)`, chosen so <10% of
/// regular `y_i` would be clipped at typical `n` and <1% for `n > 10`.
fn default_c_y(n: usize) -> f64 {
    let n = n as f64;
    n.sqrt() + 2.0 * n / (n + 2.0)
}

impl<P, V, M> Solver<P, BasicPopulationState<V>> for CmaInject<V, M>
where
    P: CostFunction<Param = V, Output = f64>,
    V: VectorLen
        + Clone
        + ScaledAdd<f64>
        + ScaleInPlace
        + ComponentMulAssign
        + NormSquared
        + SampleStandardNormal
        + std::ops::Index<usize, Output = f64>
        + std::ops::IndexMut<usize, Output = f64>,
    M: MatrixIdentity
        + MatVec<V>
        + MatTransposeVec<V>
        + ScaleInPlace
        + RankOneUpdate<V>
        + SymmetricEigen<V>
        + Clone,
{
    fn init(&mut self, problem: &P, state: BasicPopulationState<V>) -> BasicPopulationState<V> {
        // Delegate: CmaEs::init samples & evaluates the first
        // generation. Hansen's preliminary experiments inject from
        // iter 1 onward, so we don't refine the initial population.
        self.cma.init(problem, state)
    }

    fn next_iter(
        &mut self,
        problem: &P,
        state: BasicPopulationState<V>,
    ) -> (BasicPopulationState<V>, Option<TerminationReason>) {
        // 1. Standard CMA-ES iteration first: update m, σ, C from
        //    the previous generation and sample a fresh one. `state`
        //    on return has λ candidates, sorted ascending by cost.
        let (mut state, reason) = self.cma.next_iter(problem, state);
        if let Some(r) = reason {
            return (state, Some(r));
        }

        // Snapshot the CMA-ES internals we need for clipping. Clone
        // m / d_inv so we can mutate `state` and call problem.cost
        // without aliasing the &CmaEs borrow.
        let (n, m, sigma) = {
            let w = self
                .cma
                .working()
                .expect("CmaEs::init must run before CmaInject::next_iter");
            (w.n, w.m.clone(), w.sigma)
        };
        let c_y = self.c_y_override.unwrap_or_else(|| default_c_y(n));
        let edge = self.inner_simplex_scale * sigma;
        let refine = self.k.min(state.candidates.len());

        for i in 0..refine {
            // 2. Seed inner simplex around x_i with absolute,
            //    axis-aligned perturbations of size `edge = scale · σ`.
            //    Matches the CMA distribution's current spread and
            //    shrinks with σ.
            let x_i = state.candidates[i].clone();
            let mut vertices = Vec::with_capacity(n + 1);
            vertices.push(x_i.clone());
            for j in 0..n {
                let mut v = x_i.clone();
                v[j] += edge;
                vertices.push(v);
            }
            let inner_state = BasicSimplexState::from_simplex(vertices);

            // 3. Drive the inner solver. NelderMead::init evaluates
            //    all n+1 simplex vertices; subsequent iters add 1–n+1
            //    more evaluations each. cost_evals roll up in
            //    inner_state.cost_evals.
            let inner_result: OptimizationResult<BasicSimplexState<V>> =
                self.inner.run(problem, inner_state);

            // 4. Eval aggregation (AGENTS.md "Solver composition" rule 1).
            state.cost_evals += inner_result.state.cost_evals();

            // 5. Failure routing (rule 3): bubble SolverFailed only.
            if inner_result.reason.is_failure() {
                return (state, Some(inner_result.reason));
            }

            // 6. Extract the inner's best vertex.
            let x_refined = inner_result.state.param().clone();

            // 7. Compute y = (x_refined − m) / σ.
            let mut y = x_refined;
            y.scaled_add(-1.0, &m);
            y.scale_in_place(1.0 / sigma);

            // 8. ‖C^{-1/2} y‖ = ‖D^{-1} ⊙ Bᵀ y‖ (orthogonal B).
            //    Mirrors the negative-weight rescaling pattern in
            //    cma_es.rs (rank-µ update).
            let inv_sqrt_norm = {
                let w = self.cma.working().expect("working still populated");
                let mut bt_y = w.b.mat_transpose_vec(&y);
                bt_y.component_mul_assign(&w.d_inv);
                bt_y.norm_squared().sqrt()
            };

            // 9. Clipping factor α = min(1, c_y / ‖C^{-1/2} y‖)
            //    (Hansen 2011 eq. 4 + eq. 10). y = 0 (refined point
            //    landed exactly at the mean) leaves the point as-is.
            if inv_sqrt_norm > 0.0 {
                let alpha = (c_y / inv_sqrt_norm).min(1.0);
                if alpha < 1.0 {
                    y.scale_in_place(alpha);
                }
            }

            // 10. Reconstruct x_inj = m + σ · y_clipped.
            let mut x_inj = m.clone();
            x_inj.scaled_add(sigma, &y);

            // 11. Re-evaluate cost: the inner moved x_i, and clipping
            //     may have moved it further. The cost field MUST
            //     match the geometry, otherwise the next CMA update
            //     ranks the wrong point.
            let cost_new = problem.cost(&x_inj);
            state.cost_evals += 1;

            state.candidates[i] = x_inj;
            state.costs[i] = cost_new;
        }

        // 12. Re-sort: a clipped point can rank worse than its
        //     original sibling, and the rank-µ update depends on the
        //     order.
        if refine > 0 {
            sort_population_ascending(&mut state.candidates, &mut state.costs);
        }

        (state, None)
    }

    fn terminate(&self, state: &BasicPopulationState<V>) -> Option<TerminationReason> {
        // TolX inherits from CmaEs unchanged — injection doesn't
        // change the convergence criterion. CmaEs's Solver impl is
        // generic over the problem `P`, but `terminate` only reads
        // solver-internal state, so we disambiguate via the outer
        // impl's `P`.
        <CmaEs<V, M> as Solver<P, BasicPopulationState<V>>>::terminate(&self.cma, state)
    }
}
