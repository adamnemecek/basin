//! The [`Solver`] trait every concrete solver implements. See the trait
//! contract for the lifecycle (`init` once, then repeated `next_iter`,
//! with an optional `terminate` hook) and the
//! [`executor`](crate::core::executor) module for the canonical iteration
//! ordering.

use crate::core::state::State;
use crate::core::termination::TerminationReason;

/// A concrete optimization algorithm. Implementations carry the
/// solver's configuration and any internal scratch state; the iterate
/// itself lives in `S: State`.
///
/// # Contract
///
/// - **Caller must:** drive the solver through
///   [`Executor`](crate::core::executor::Executor) (or
///   [`run_loop`](crate::core::executor::run_loop) for composed
///   solvers). The executor calls [`init`](Self::init) exactly once
///   before any [`next_iter`](Self::next_iter) call, and runs
///   termination checks before each iteration including iter 0. See the
///   [`executor`](crate::core::executor) module docs for the canonical
///   loop ordering.
/// - **Implementor must:** populate every state field termination
///   criteria might read before returning from
///   [`init`](Self::init) — at minimum [`State::cost`] for any state
///   whose `cost()` panics on missing data, and
///   [`GradientState::gradient`](crate::core::state::GradientState::gradient)
///   for first-order solvers. After every successful
///   [`next_iter`](Self::next_iter), the same fields must again
///   correspond to the *current* [`State::param`].
/// - **Implementor must:** report mid-iteration failures
///   (line-search bailout, non-descent direction, etc.) via
///   [`next_iter`](Self::next_iter)'s `Option<TerminationReason>`
///   return rather than panicking; and use [`terminate`](Self::terminate)
///   only for clean convergence tests on the current state.
pub trait Solver<P, S: State> {
    /// One-time setup before the iteration loop.
    ///
    /// # Contract
    ///
    /// - **Implementor must:** seed every state field that termination
    ///   criteria or downstream
    ///   [`next_iter`](Self::next_iter) calls will read at iter 0 — at
    ///   minimum [`State::cost`], plus
    ///   [`GradientState::gradient`](crate::core::state::GradientState::gradient)
    ///   for first-order solvers and the parallel cost array for
    ///   [`SimplexState`](crate::core::state::SimplexState) solvers.
    ///   Termination criteria run *before* the first
    ///   [`next_iter`](Self::next_iter) call (see the
    ///   [`executor`](crate::core::executor) module docs), so an
    ///   already-optimal initial point must be detectable from the state
    ///   `init` returns.
    /// - **Implementor must:** count work it does here against the
    ///   eval counters (`state.cost_evals`, `state.gradient_evals`).
    fn init(&mut self, _problem: &P, state: S) -> S {
        state
    }

    /// Advance one iteration.
    ///
    /// # Contract
    ///
    /// - **Implementor must:** return a state whose
    ///   [`State::param`], [`State::cost`], and (if `S: GradientState`)
    ///   [`GradientState::gradient`](crate::core::state::GradientState::gradient)
    ///   are mutually consistent at the new iterate. Termination
    ///   criteria evaluated *before* the next iteration assume these
    ///   fields agree.
    /// - **Implementor must:** report mid-iteration failures
    ///   (line-search bailout, non-descent direction, ill-conditioned
    ///   subproblem, …) via the returned
    ///   `Option<TerminationReason>` rather than panicking. The executor
    ///   stops immediately when `Some(_)` is returned and the
    ///   iteration counter is *not* incremented, so
    ///   `state.iter()` reflects the last *fully completed* iteration.
    /// - **Implementor must:** count every cost / gradient call against
    ///   the corresponding eval counter on the state.
    /// - **Implementor must (composition):** when running an inner solver
    ///   via [`InnerExecutor`](crate::core::inner::InnerExecutor) or
    ///   [`run_loop`](crate::core::executor::run_loop), roll the inner
    ///   result's
    ///   [`State::cost_evals`](crate::core::state::State::cost_evals)
    ///   into the outer state via
    ///   [`State::increment_cost_evals`](crate::core::state::State::increment_cost_evals)
    ///   (and the gradient analogue when both inner and outer are
    ///   [`GradientState`](crate::core::state::GradientState)).
    ///   `MaxCostEvals` budgets and the public `result.cost_evals()`
    ///   read are wrong otherwise. See `AGENTS.md` "Solver composition"
    ///   for the full contract (eval aggregation, criteria
    ///   statelessness, failure routing).
    fn next_iter(&mut self, problem: &P, state: S) -> (S, Option<TerminationReason>);

    /// Optional pre-iteration solver-specific termination test.
    ///
    /// Called after framework
    /// [`TerminationCriterion`](crate::core::termination::TerminationCriterion)
    /// checks but before each [`next_iter`](Self::next_iter) (including
    /// iter 0, after [`init`](Self::init)). Returning `Some(_)` halts
    /// the executor. Use for clean convergence tests that depend only
    /// on the current state; mid-iter failures should be reported via
    /// [`next_iter`](Self::next_iter)'s return value instead. See the
    /// [`executor`](crate::core::executor) module docs for the full
    /// ordering.
    fn terminate(&self, _state: &S) -> Option<TerminationReason> {
        None
    }
}
