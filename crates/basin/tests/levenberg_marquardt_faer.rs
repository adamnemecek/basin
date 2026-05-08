#![cfg(feature = "faer")]

use basin::problems::{PowellSingular, RosenbrockResiduals};
use basin::{BasicState, Executor, LevenbergMarquardt, TerminationReason};
use faer::Col;

#[test]
fn levenberg_marquardt_converges_on_rosenbrock_residuals() {
    let problem = RosenbrockResiduals::<Col<f64>>::new();
    let initial = Col::from_fn(2, |i| if i == 0 { -1.2 } else { 1.0 });

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
    // Mirror of the nalgebra "why LM" test. At (1, 2, 1, 1) GN's
    // Cholesky fails on the singular JᵀJ; LM's damping recovers and
    // drives Powell toward x* = 0.
    let problem = PowellSingular::<Col<f64>>::new();
    let initial = Col::from_fn(4, |i| match i {
        0 => 1.0,
        1 => 2.0,
        2 => 1.0,
        _ => 1.0,
    });

    let result = Executor::new(problem, LevenbergMarquardt::new(), BasicState::new(initial))
        .max_iter(200)
        .run();

    assert_eq!(result.reason, TerminationReason::SolverConverged);
    assert!(
        result.cost() < 1e-10,
        "cost = {} (LM should drive Powell to the origin)",
        result.cost()
    );
    for i in 0..4 {
        let xi = result.param()[i];
        assert!(xi.abs() < 1e-2, "x[{}] = {}", i, xi);
    }
}

#[test]
fn levenberg_marquardt_converges_on_powell_singular_classical_start() {
    let problem = PowellSingular::<Col<f64>>::new();
    let initial = Col::from_fn(4, |i| match i {
        0 => 3.0,
        1 => -1.0,
        2 => 0.0,
        _ => 1.0,
    });

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
    let problem = RosenbrockResiduals::<Col<f64>>::new();
    let initial = Col::from_fn(2, |i| if i == 0 { -1.2 } else { 1.0 });

    let result = Executor::new(problem, LevenbergMarquardt::new(), BasicState::new(initial))
        .max_iter(100)
        .run();

    assert_eq!(result.reason, TerminationReason::SolverConverged);
}
