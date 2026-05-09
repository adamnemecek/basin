use crate::core::constraint::BoxConstrained;
use crate::core::math::SampleUniformBox;
use crate::core::problem::CostFunction;
use crate::core::rng::{ChaCha8Rng, SeedableRng};
use crate::core::solver::Solver;
use crate::core::state::BasicPopulationState;
use crate::core::termination::TerminationReason;

/// Elitist (1+λ) random search over a feasible box.
///
/// The first stochastic solver in basin and the smallest vehicle for the
/// new [`BasicPopulationState`] / [`PopulationState`] story
/// — a derivative-free, population-based method that exercises the
/// reproducible RNG infrastructure (see [`crate::core::rng`]) without
/// pulling in any covariance / distribution machinery (those land in
/// S8 alongside CMA-ES).
///
/// # Algorithm
///
/// At [`init`](Solver::init) the solver fills `state.candidates` with
/// `λ` candidates drawn component-wise uniformly from the problem's
/// box `[lower, upper]`, evaluates each, and sorts by ascending cost.
///
/// Each [`next_iter`](Solver::next_iter):
///
/// ```text
/// elite ← (candidates[0], costs[0])           # current best
/// resample λ candidates uniformly in [lower, upper]
/// evaluate each, append along with the elite (now λ + 1 entries)
/// sort ascending by cost
/// truncate back to λ                            # drops the worst
/// ```
///
/// The elite carry-over keeps `state.cost()` non-increasing across
/// generations, so the framework's
/// [`CostTolerance`](crate::core::termination::CostTolerance) and
/// [`ParamTolerance`](crate::core::termination::ParamTolerance) work
/// honestly without redesign. (CMA-ES is genuinely non-monotone, and
/// the "no monotone cost" termination story will be designed alongside
/// it in S8 / S9.)
///
/// # Reproducibility
///
/// The solver carries a [`ChaCha8Rng`] seeded from the `seed: u64`
/// passed to [`new`](Self::new). Same seed → same trajectory, on every
/// platform basin builds for (including `wasm32-unknown-unknown`). To
/// vary runs, vary the seed; to share runs, share the seed.
///
/// # Contract
///
/// - **Caller must:** implement [`BoxConstrained`] on the problem with
///   `lower[i] ≤ upper[i]` for every component. Equal bounds are
///   allowed (the corresponding component is pinned).
/// - **Caller must:** hand in a [`BasicPopulationState`] sized to match
///   the solver's `lambda`. The two natural constructors are
///   [`BasicPopulationState::with_size(lambda)`](BasicPopulationState::with_size)
///   (solver fills the population in `init`) or
///   [`from_population`](BasicPopulationState::from_population) (caller
///   supplies a custom initial distribution). `with_size` is the
///   common case.
/// - **Implementor (this solver) must:** maintain feasibility (every
///   candidate after `init` is in the box) and the sorted-by-cost
///   invariant on
///   [`PopulationState`](crate::core::state::PopulationState) at the
///   start and end of each iteration.
///
/// # Termination
///
/// No solver-internal optimality test — random search has no canonical
/// fixed-point criterion. Use the framework's
/// [`MaxIter`](crate::core::termination::MaxIter),
/// [`MaxCostEvals`](crate::core::termination::MaxCostEvals),
/// [`MaxTime`](crate::core::termination::MaxTime),
/// [`CostTolerance`](crate::core::termination::CostTolerance), or
/// [`ParamTolerance`](crate::core::termination::ParamTolerance). The
/// elite-carryover makes cost monotonicity honest, so cost-based budgets
/// behave as expected.
///
/// # Backends
///
/// Backend-generic — works with any `V` implementing
/// [`SampleUniformBox`] + `Clone`. That covers `Vec<f64>`,
/// `nalgebra::DVector<f64>` (feature `nalgebra`),
/// `ndarray::Array1<f64>` (feature `ndarray`), and `faer::Col<f64>`
/// (feature `faer`). The problem must implement [`BoxConstrained`].
pub struct RandomSearch {
    lambda: usize,
    rng: ChaCha8Rng,
}

impl RandomSearch {
    /// New `RandomSearch` with population size `lambda` and PRNG seed
    /// `seed`. Same `seed` → same iterate trajectory on the same
    /// problem; vary `seed` to vary the run.
    ///
    /// # Panics
    ///
    /// Panics if `lambda == 0`. A non-empty population is the smallest
    /// thing this solver can iterate on.
    pub fn new(lambda: usize, seed: u64) -> Self {
        assert!(lambda >= 1, "RandomSearch requires lambda >= 1");
        Self {
            lambda,
            rng: ChaCha8Rng::seed_from_u64(seed),
        }
    }
}

/// Sort `candidates` and `costs` jointly by ascending cost. NaN costs
/// sort last so a single bad evaluation can't drag itself to the
/// front. Mirrors `nelder_mead::sort_simplex`.
fn sort_population_ascending<V>(candidates: &mut [V], costs: &mut [f64]) {
    let n = candidates.len();
    debug_assert_eq!(n, costs.len());
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| {
        costs[i]
            .partial_cmp(&costs[j])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    apply_permutation(candidates, &idx);
    apply_permutation(costs, &idx);
}

fn apply_permutation<T>(slice: &mut [T], idx: &[usize]) {
    let mut visited = vec![false; slice.len()];
    for start in 0..slice.len() {
        if visited[start] || idx[start] == start {
            visited[start] = true;
            continue;
        }
        let mut current = start;
        loop {
            let next = idx[current];
            visited[current] = true;
            if next == start {
                break;
            }
            slice.swap(current, next);
            current = next;
        }
    }
}

impl<P, V> Solver<P, BasicPopulationState<V>> for RandomSearch
where
    P: CostFunction<Param = V, Output = f64> + BoxConstrained<Param = V>,
    V: SampleUniformBox + Clone,
{
    fn init(&mut self, problem: &P, mut state: BasicPopulationState<V>) -> BasicPopulationState<V> {
        let lo = problem.lower();
        let hi = problem.upper();
        // The state can arrive with a caller-supplied initial population
        // (`from_population`) or empty (`with_size`). The solver always
        // owns the *first* generation here: clear and resample so the
        // RNG-determined trajectory is reproducible regardless of which
        // constructor was used. Callers who genuinely want a custom
        // initial population should call `from_population` and skip
        // `init` by stepping the solver themselves — `init` is the
        // place where the solver's seeded RNG seeds the run.
        state.candidates.clear();
        state.costs.clear();
        for _ in 0..self.lambda {
            let x = V::sample_uniform_box(lo, hi, &mut self.rng);
            let c = problem.cost(&x);
            state.candidates.push(x);
            state.costs.push(c);
        }
        state.cost_evals += self.lambda as u64;
        sort_population_ascending(&mut state.candidates, &mut state.costs);
        state
    }

    fn next_iter(
        &mut self,
        problem: &P,
        mut state: BasicPopulationState<V>,
    ) -> (BasicPopulationState<V>, Option<TerminationReason>) {
        // Snapshot the elite before resampling — this is what makes
        // state.cost() monotone.
        let elite_x = state.candidates[0].clone();
        let elite_c = state.costs[0];

        let lo = problem.lower();
        let hi = problem.upper();
        state.candidates.clear();
        state.costs.clear();
        state.candidates.push(elite_x);
        state.costs.push(elite_c);
        for _ in 0..self.lambda {
            let x = V::sample_uniform_box(lo, hi, &mut self.rng);
            let c = problem.cost(&x);
            state.candidates.push(x);
            state.costs.push(c);
        }
        state.cost_evals += self.lambda as u64;
        sort_population_ascending(&mut state.candidates, &mut state.costs);
        // Drop the worst back down to λ. Sort puts the elite first
        // when it's still the best, so truncation never drops it.
        state.candidates.truncate(self.lambda);
        state.costs.truncate(self.lambda);
        (state, None)
    }
}
