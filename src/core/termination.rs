use web_time::{Duration, Instant};

use crate::core::math::{NormInfinity, NormSquared, ScaledAdd};
use crate::core::state::{GradientState, SimplexState, State};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerminationReason {
    MaxIter,
    GradientTolerance,
    ParamTolerance,
    CostTolerance,
    SimplexTolerance,
    MaxTime,
    /// Solver determined it has converged (e.g. fixed point reached).
    SolverConverged,
    /// Solver cannot make further progress (e.g. line search failure).
    SolverFailed,
}

/// A pluggable termination check evaluated by the executor.
///
/// Criteria are checked *before* each iteration (including iter 0) so an
/// already-optimal initial point exits immediately. Each criterion's bound
/// on `S` decides whether it can be applied to a given state — e.g.
/// `GradientTolerance` requires `S: GradientState`, so derivative-free
/// states cannot be paired with it (compile error, not runtime "N/A").
pub trait TerminationCriterion<S> {
    fn check(&mut self, state: &S) -> Option<TerminationReason>;
}

/// Stop after `state.iter() >= n` iterations.
pub struct MaxIter(pub u64);

impl<S: State> TerminationCriterion<S> for MaxIter {
    fn check(&mut self, state: &S) -> Option<TerminationReason> {
        if state.iter() >= self.0 {
            Some(TerminationReason::MaxIter)
        } else {
            None
        }
    }
}

/// Stop when `‖∇f(x)‖ ≤ tol`. Skipped silently when the state has no
/// gradient populated yet (e.g. iter 0 before `init` has run).
pub struct GradientTolerance(pub f64);

impl<S> TerminationCriterion<S> for GradientTolerance
where
    S: GradientState,
    S::Param: NormSquared,
{
    fn check(&mut self, state: &S) -> Option<TerminationReason> {
        let g = state.gradient()?;
        if g.norm_squared() <= self.0 * self.0 {
            Some(TerminationReason::GradientTolerance)
        } else {
            None
        }
    }
}

/// Stop when `‖x_k − x_{k−1}‖ ≤ tol`. Holds its own copy of the previous
/// iterate so it doesn't depend on state-side history.
pub struct ParamTolerance<P> {
    tol_squared: f64,
    last: Option<P>,
}

impl<P> ParamTolerance<P> {
    pub fn new(tol: f64) -> Self {
        Self {
            tol_squared: tol * tol,
            last: None,
        }
    }
}

impl<S, P> TerminationCriterion<S> for ParamTolerance<P>
where
    S: State<Param = P>,
    P: ScaledAdd<f64> + NormSquared + Clone,
{
    fn check(&mut self, state: &S) -> Option<TerminationReason> {
        let curr = state.param();
        let triggered = if let Some(last) = &self.last {
            let mut diff = curr.clone();
            diff.scaled_add(-1.0, last);
            diff.norm_squared() <= self.tol_squared
        } else {
            false
        };
        self.last = Some(curr.clone());
        triggered.then_some(TerminationReason::ParamTolerance)
    }
}

/// Stop when `|f_k − f_{k−1}| ≤ tol`. Holds its own copy of the previous
/// cost.
pub struct CostTolerance {
    tol: f64,
    last: Option<f64>,
}

impl CostTolerance {
    pub fn new(tol: f64) -> Self {
        Self { tol, last: None }
    }
}

impl<S> TerminationCriterion<S> for CostTolerance
where
    S: State<Float = f64>,
{
    fn check(&mut self, state: &S) -> Option<TerminationReason> {
        let curr = state.cost();
        let triggered = self
            .last
            .is_some_and(|l| (l - curr).abs() <= self.tol && curr.is_finite());
        self.last = Some(curr);
        triggered.then_some(TerminationReason::CostTolerance)
    }
}

/// Simplex-collapse test for simplex-based solvers (e.g. Nelder-Mead),
/// per Lagarias et al. (1998), eq. (T1):
///
/// stop when `max_i ‖x_i − x_1‖_∞ ≤ tol_x` **and**
/// `max_i |f_i − f_1| ≤ tol_f`, where `x_1` / `f_1` are the best vertex
/// and its cost.
///
/// Bound on `S: SimplexState` so derivative-free solvers that don't carry
/// a simplex can't be paired with it.
pub struct SimplexTolerance {
    tol_x: f64,
    tol_f: f64,
}

impl SimplexTolerance {
    pub fn new(tol_x: f64, tol_f: f64) -> Self {
        Self { tol_x, tol_f }
    }
}

impl<S> TerminationCriterion<S> for SimplexTolerance
where
    S: SimplexState<Float = f64>,
    S::Param: Clone + ScaledAdd<f64> + NormInfinity,
{
    fn check(&mut self, state: &S) -> Option<TerminationReason> {
        let vertices = state.vertices();
        let costs = state.costs();
        let best = &vertices[0];
        let best_cost = costs[0];

        for x_i in &vertices[1..] {
            let mut diff = x_i.clone();
            diff.scaled_add(-1.0, best);
            if diff.norm_infinity() > self.tol_x {
                return None;
            }
        }
        for &f_i in &costs[1..] {
            if (f_i - best_cost).abs() > self.tol_f {
                return None;
            }
        }
        Some(TerminationReason::SimplexTolerance)
    }
}

/// Stop after wall-clock time `limit` has elapsed since the first `check`.
///
/// Uses `web-time::Instant` so it works on both native and
/// `wasm32-unknown-unknown` without feature gating.
pub struct MaxTime {
    limit: Duration,
    start: Option<Instant>,
}

impl MaxTime {
    pub fn new(limit: Duration) -> Self {
        Self { limit, start: None }
    }
}

impl<S> TerminationCriterion<S> for MaxTime {
    fn check(&mut self, _state: &S) -> Option<TerminationReason> {
        let start = *self.start.get_or_insert_with(Instant::now);
        if start.elapsed() >= self.limit {
            Some(TerminationReason::MaxTime)
        } else {
            None
        }
    }
}
