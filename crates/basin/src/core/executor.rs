//! Iteration driver. The high-level entry point is [`Executor`];
//! [`Stepper`] exposes one-iteration-at-a-time control, and [`run_loop`]
//! is the borrowed-problem variant used by composed solvers.
//!
//! # Canonical iteration ordering
//!
//! [`Executor::run`] (and the equivalent [`Stepper`] / [`run_loop`]
//! paths) drive the solver through this exact sequence — every
//! contract elsewhere in the framework cross-links here:
//!
//! 1. [`Solver::init`] is called **once**, on the initial state. The
//!    returned state is what iter-0 sees.
//! 2. Then, repeatedly, before each [`Solver::next_iter`] call
//!    (including the first):
//!    1. The built-in [`MaxIter`](crate::core::termination::MaxIter)
//!       limit is checked against [`State::iter`]. If
//!       `state.iter() >= max_iter`, the run stops with
//!       [`TerminationReason::MaxIter`].
//!    2. Each registered [`TerminationCriterion`] is checked **in
//!       insertion order**. The **first to return `Some(reason)` halts
//!       the run** — later criteria do not run that iteration.
//!    3. The solver's own [`Solver::terminate`] hook is checked.
//!       `Some(_)` halts the run.
//! 3. If nothing fired, [`Solver::next_iter`] is called. It may itself
//!    report a mid-iter termination via its return tuple; in that case
//!    the iteration counter is **not** incremented, so the final
//!    [`State::iter`] reflects the last *fully completed* iteration.
//! 4. Otherwise the iteration counter is incremented and we go back to
//!    step 2.
//!
//! Because checks happen *before* iter 0, an already-optimal initial
//! point exits immediately with the corresponding reason rather than
//! taking one redundant step.

use crate::core::solver::Solver;
use crate::core::state::State;
use crate::core::termination::{TerminationCriterion, TerminationReason};

/// Outcome of an optimisation run.
///
/// Owns the final solver state plus the reason the executor stopped.
/// Delegates `param()` / `cost()` / `iter()` to the underlying state so
/// callers don't need to import `State` for the common reads.
pub struct OptimizationResult<S> {
    pub state: S,
    pub reason: TerminationReason,
}

impl<S: State> OptimizationResult<S> {
    pub fn param(&self) -> &S::Param {
        self.state.param()
    }

    pub fn cost(&self) -> S::Float {
        self.state.cost()
    }

    pub fn iter(&self) -> u64 {
        self.state.iter()
    }

    pub fn cost_evals(&self) -> u64 {
        self.state.cost_evals()
    }

    pub fn into_state(self) -> S {
        self.state
    }
}

/// Outcome of a single [`Stepper::step`] call.
///
/// `Stopped` carries the same [`TerminationReason`] the executor would
/// have returned. After `Stopped` is returned once, subsequent calls to
/// `step` keep returning the same `Stopped(reason)` so callers don't
/// have to track whether they're done.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepOutcome {
    Continue,
    Stopped(TerminationReason),
}

/// Drive a solver one iteration at a time.
///
/// Owns the problem, state, solver and termination criteria, runs
/// `solver.init` exactly once on construction, and exposes
/// [`step`](Self::step) / [`run_to_end`](Self::run_to_end) so callers can
/// interleave their own work between iterations — recording trajectories,
/// animating from a UI, pausing on a button press, evaluating a custom
/// budget, etc.
///
/// [`Executor::run`] is `self.into_stepper().run_to_end()`; the stepper
/// is the building block, the executor is the convenience wrapper.
///
/// # Example
///
/// ```ignore
/// let mut stepper = Executor::new(problem, solver, state)
///     .max_iter(100)
///     .terminate_on(GradientTolerance(1e-6))
///     .into_stepper();
///
/// let reason = loop {
///     match stepper.step() {
///         StepOutcome::Continue => { /* observe `stepper.state()` */ }
///         StepOutcome::Stopped(reason) => break reason,
///     }
/// };
/// ```
pub struct Stepper<P, S, So> {
    problem: P,
    // `Option<S>` because `Solver::next_iter` consumes the state by
    // value. Take it out, hand it to the solver, put the returned state
    // back. The slot is `Some` whenever a caller can observe it (between
    // `step` calls and at construction / drop), so `state()` and
    // `into_state` can unwrap without checks.
    state: Option<S>,
    solver: So,
    criteria: Vec<Box<dyn TerminationCriterion<S>>>,
    max_iter: u64,
    finished: Option<TerminationReason>,
}

impl<P, S, So> Stepper<P, S, So>
where
    S: State,
    So: Solver<P, S>,
{
    /// Read-only access to the current state, between steps.
    pub fn state(&self) -> &S {
        self.state
            .as_ref()
            .expect("state slot is Some between steps")
    }

    /// Termination reason if the stepper has stopped, else `None`.
    pub fn finished(&self) -> Option<&TerminationReason> {
        self.finished.as_ref()
    }

    /// Total iterations that have completed so far. Convenience read
    /// equivalent to `self.state().iter()`.
    pub fn iter(&self) -> u64 {
        self.state().iter()
    }

    /// Advance one iteration. Once a `Stopped` outcome has been returned
    /// the stepper is sticky: subsequent calls keep returning the same
    /// `Stopped(reason)` without touching the state or solver.
    pub fn step(&mut self) -> StepOutcome {
        if let Some(reason) = self.finished {
            return StepOutcome::Stopped(reason);
        }
        let outcome = step_once(
            &self.problem,
            &mut self.state,
            &mut self.solver,
            &mut self.criteria,
            self.max_iter,
        );
        if let StepOutcome::Stopped(reason) = outcome {
            self.finished = Some(reason);
        }
        outcome
    }

    /// Drive [`step`](Self::step) to completion and return an
    /// [`OptimizationResult`].
    pub fn run_to_end(mut self) -> OptimizationResult<S> {
        loop {
            if let StepOutcome::Stopped(reason) = self.step() {
                return OptimizationResult {
                    state: self.state.take().expect("state slot is Some on stop"),
                    reason,
                };
            }
        }
    }

    /// Consume the stepper and return the final state.
    pub fn into_state(self) -> S {
        self.state.expect("state slot is Some at drop")
    }
}

/// Single-iteration core, shared by [`Stepper::step`] (owned) and
/// [`run_loop`] (borrowed). Reads the current state via `state_slot`,
/// checks termination, and either returns `Stopped` (slot left
/// untouched) or hands the state to `solver.next_iter`, increments the
/// iteration counter, and puts the returned state back in `state_slot`.
///
/// Invariant: `state_slot` is `Some` on entry and `Some` on return.
fn step_once<P, S, So>(
    problem: &P,
    state_slot: &mut Option<S>,
    solver: &mut So,
    criteria: &mut [Box<dyn TerminationCriterion<S>>],
    max_iter: u64,
) -> StepOutcome
where
    S: State,
    So: Solver<P, S>,
{
    {
        let state = state_slot
            .as_ref()
            .expect("step_once called with empty state slot");
        if state.iter() >= max_iter {
            return StepOutcome::Stopped(TerminationReason::MaxIter);
        }
        for criterion in criteria.iter_mut() {
            if let Some(reason) = criterion.check(state) {
                return StepOutcome::Stopped(reason);
            }
        }
        if let Some(reason) = solver.terminate(state) {
            return StepOutcome::Stopped(reason);
        }
    }
    let prev = state_slot.take().unwrap();
    let (mut next, mid_iter_reason) = solver.next_iter(problem, prev);
    if let Some(reason) = mid_iter_reason {
        *state_slot = Some(next);
        return StepOutcome::Stopped(reason);
    }
    next.increment_iter();
    *state_slot = Some(next);
    StepOutcome::Continue
}

/// Drive a solver to completion against a borrowed problem.
///
/// `Executor` is a thin owning wrapper over this. Composed solvers
/// (e.g. CG inside CMA, NM inside DE) call `run_loop` directly so they
/// can run an inner solver against the outer's `&P` without taking
/// ownership of the problem.
///
/// Semantics match `Executor::run`: `init` is called once, then on each
/// iteration framework `criteria` are checked in insertion order before
/// the solver's own `terminate` hook, before stepping. `max_iter` is
/// checked against `state.iter()` and exits with `TerminationReason::MaxIter`.
/// `next_iter` may also report a mid-iter termination via its return tuple;
/// in that case the iteration counter is left untouched so the final
/// `state.iter()` still reflects the last fully completed iteration.
pub fn run_loop<P, S, So>(
    problem: &P,
    state: S,
    solver: &mut So,
    criteria: &mut [Box<dyn TerminationCriterion<S>>],
    max_iter: u64,
) -> OptimizationResult<S>
where
    S: State,
    So: Solver<P, S>,
{
    let state = solver.init(problem, state);
    let mut slot = Some(state);
    let reason = loop {
        match step_once(problem, &mut slot, solver, criteria, max_iter) {
            StepOutcome::Continue => continue,
            StepOutcome::Stopped(reason) => break reason,
        }
    };
    OptimizationResult {
        state: slot.take().expect("state slot is Some on stop"),
        reason,
    }
}

pub struct Executor<P, S, So> {
    problem: P,
    state: S,
    solver: So,
    max_iter: u64,
    criteria: Vec<Box<dyn TerminationCriterion<S>>>,
}

impl<P, S, So> Executor<P, S, So>
where
    S: State,
    So: Solver<P, S>,
{
    pub fn new(problem: P, solver: So, state: S) -> Self {
        Self {
            problem,
            state,
            solver,
            max_iter: 1000,
            criteria: Vec::new(),
        }
    }

    /// Convenience setter for the default `MaxIter` criterion. Equivalent
    /// effect to `terminate_on(MaxIter(n))` but mutates a dedicated field
    /// so subsequent calls replace rather than stack.
    pub fn max_iter(mut self, n: u64) -> Self {
        self.max_iter = n;
        self
    }

    /// Add a termination criterion. Criteria are checked in insertion
    /// order before each iteration (and before iter 0); the first to
    /// return `Some(_)` stops the run. See the [module docs](self) for
    /// the full per-iteration ordering.
    pub fn terminate_on<C>(mut self, criterion: C) -> Self
    where
        C: TerminationCriterion<S> + 'static,
    {
        self.criteria.push(Box::new(criterion));
        self
    }

    /// Convert the executor into a [`Stepper`] for one-iteration-at-a-time
    /// control. `solver.init` runs here so the returned stepper sits at
    /// iter 0 with a complete state.
    pub fn into_stepper(self) -> Stepper<P, S, So> {
        let Executor {
            problem,
            state,
            mut solver,
            max_iter,
            criteria,
        } = self;
        let state = solver.init(&problem, state);
        Stepper {
            problem,
            state: Some(state),
            solver,
            criteria,
            max_iter,
            finished: None,
        }
    }

    pub fn run(self) -> OptimizationResult<S> {
        self.into_stepper().run_to_end()
    }
}
