#![cfg(feature = "nalgebra")]

use basin::problems::{PowellSingular, RosenbrockResiduals};
use basin::{BasicState, Executor, LevenbergMarquardt, TerminationReason};
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
