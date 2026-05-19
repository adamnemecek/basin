#![cfg(feature = "nalgebra")]

use basin::problems::{PowellSingular, RosenbrockResiduals};
use basin::{BasicState, Executor, GradientState, LevenbergMarquardt, TerminationReason};
use nalgebra::DVector;

#[test]
fn levenberg_marquardt_converges_on_rosenbrock_residuals() {
    // LM should converge on Rosenbrock-as-residuals from the classical
    // start. Unlike GN's exact two-step convergence (the linear model
    // is exact along y at fixed x), LM takes a few extra iterations
    // because the damping starts non-zero — but it still reaches the
    // optimum cleanly and emits SolverConverged.
    let problem = RosenbrockResiduals::<DVector<f64>>::new();
    let initial = DVector::from_vec(vec![-1.2, 1.0]);

    let result = Executor::new(problem, LevenbergMarquardt::new(), BasicState::new(initial))
        .max_iter(50)
        .run();

    assert_eq!(result.reason, TerminationReason::SolverConverged);
    assert!(result.cost() < 1e-15, "cost = {}", result.cost());
    assert!(
        (result.param()[0] - 1.0).abs() < 1e-7,
        "x[0] = {}",
        result.param()[0]
    );
    assert!(
        (result.param()[1] - 1.0).abs() < 1e-7,
        "x[1] = {}",
        result.param()[1]
    );
}

#[test]
fn levenberg_marquardt_recovers_on_rank_deficient_powell_singular() {
    // Load-bearing "why LM" test, mirror of GN's failure at the same
    // point. At x = (1, 2, 1, 1) Powell's quadratic-residual rows
    // r₂, r₃ have vanishing Jacobian rows (J has rank 2 < 4), so JᵀJ
    // is singular and pure GN fails Cholesky. LM's damping makes
    // (JᵀJ + μI) SPD by construction, so it should converge cleanly
    // — the canonical demonstration that LM strictly subsumes GN.
    let problem = PowellSingular::<DVector<f64>>::new();
    let initial = DVector::from_vec(vec![1.0, 2.0, 1.0, 1.0]);

    let result = Executor::new(problem, LevenbergMarquardt::new(), BasicState::new(initial))
        .max_iter(200)
        .run();

    assert_eq!(result.reason, TerminationReason::SolverConverged);
    assert!(
        result.cost() < 1e-10,
        "cost = {} (LM should drive Powell to the origin)",
        result.cost()
    );
    // Powell's optimum is x* = 0; check each component drifted toward it.
    for (i, &xi) in result.param().iter().enumerate() {
        assert!(xi.abs() < 1e-2, "x[{}] = {} (expected near 0)", i, xi);
    }
}

#[test]
fn levenberg_marquardt_converges_on_powell_singular_classical_start() {
    // Classical start (3, −1, 0, 1). GN converges here in 12 iterations
    // (per the S3 session notes) because the rank deficiency only
    // bites at the optimum. LM with default Nielsen damping should
    // converge in a comparable iteration count.
    let problem = PowellSingular::<DVector<f64>>::new();
    let initial = DVector::from_vec(vec![3.0, -1.0, 0.0, 1.0]);

    let result = Executor::new(problem, LevenbergMarquardt::new(), BasicState::new(initial))
        .max_iter(100)
        .run();

    assert_eq!(result.reason, TerminationReason::SolverConverged);
    assert!(
        result.cost() < 1e-10,
        "cost = {} (Powell from classical start should reach near-zero)",
        result.cost()
    );
}

#[test]
fn levenberg_marquardt_emits_solver_converged_via_first_order_optimality() {
    // Convergence path lands SolverConverged (not MaxIter): LM's
    // internal ‖Jᵀr‖_∞ ≤ tol_grad check fires once the iterate is at
    // the optimum. Mirror of the GN test for the same property.
    let problem = RosenbrockResiduals::<DVector<f64>>::new();
    let initial = DVector::from_vec(vec![-1.2, 1.0]);

    let result = Executor::new(problem, LevenbergMarquardt::new(), BasicState::new(initial))
        .max_iter(100)
        .run();

    assert_eq!(result.reason, TerminationReason::SolverConverged);
}

#[test]
fn levenberg_marquardt_caches_residual_and_jacobian_across_iterations() {
    // Regression test for the Madsen-Nielsen caching contract (Alg.
    // 3.16, line 13: J reassigned only after acceptance). At the top
    // of each `next_iter`, LM reuses the residual and Jacobian stashed
    // by either `init` or the previous iteration's bookkeeping —
    // re-evaluating them at the same point is wasted work.
    //
    // Disable the internal `‖Jᵀr‖_∞ ≤ tol_grad` check so termination
    // is purely by MaxIter; the early-exit path otherwise evaluates J
    // an extra time on a not-yet-counted iter and muddies the count.
    //
    // For K completed iters on Rosenbrock-as-residuals from the
    // classical start, LM's μ-update accepts every step (no rejections),
    // so:
    //   - cost_evals = 1 (init) + K (one trial per iter)
    //   - gradient_evals = K (init's J carries iter 1; each subsequent
    //     iter's J is recomputed because the previous accept cleared
    //     the cache — the last iter's accept clears it but no
    //     follow-up iter consumes it under MaxIter exit).
    let problem = RosenbrockResiduals::<DVector<f64>>::new();
    let initial = DVector::from_vec(vec![-1.2, 1.0]);

    let result = Executor::new(
        problem,
        LevenbergMarquardt::new().tol_grad(0.0),
        BasicState::new(initial),
    )
    .max_iter(3)
    .run();

    assert_eq!(result.reason, TerminationReason::MaxIter);
    assert_eq!(result.iter(), 3);
    assert_eq!(
        result.cost_evals(),
        4,
        "expected init (1) + one trial per iter (3) = 4 — uncached LM would also \
         re-evaluate the start-of-iter residual and produce 1 + 2·iters = 7"
    );
    assert!(
        result.state.gradient_evals() <= 3,
        "gradient_evals = {} should be ≤ iters (3): init's J carries iter 1, and \
         rejected steps reuse J at the unchanged iterate. Uncached LM produces \
         1 + iters = 4.",
        result.state.gradient_evals()
    );
}
