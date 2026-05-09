#![cfg(feature = "nalgebra")]

use basin::problems::{Rosenbrock, Sphere};
use basin::{
    BasicPopulationState, CmaEs, Executor, PopulationState, StepOutcome, TerminationReason,
};
use nalgebra::{DMatrix, DVector};

/// Same seed → same trajectory, on the nalgebra backend. Reproducibility
/// is the load-bearing contract for the seeded RNG (per
/// [`crate::core::rng`] docs); regression-guards a constant-RNG bug.
#[test]
fn same_seed_yields_identical_trajectory() {
    let m0 = DVector::from_vec(vec![0.5, 0.5, 0.5, 0.5, 0.5]);

    let result_a = Executor::new(
        Sphere::<DVector<f64>>::new(),
        CmaEs::<DVector<f64>, DMatrix<f64>>::new(m0.clone(), 0.3, 42),
        BasicPopulationState::<DVector<f64>>::with_size(
            CmaEs::<DVector<f64>, DMatrix<f64>>::default_lambda(5),
        ),
    )
    .max_iter(30)
    .run();

    let result_b = Executor::new(
        Sphere::<DVector<f64>>::new(),
        CmaEs::<DVector<f64>, DMatrix<f64>>::new(m0, 0.3, 42),
        BasicPopulationState::<DVector<f64>>::with_size(
            CmaEs::<DVector<f64>, DMatrix<f64>>::default_lambda(5),
        ),
    )
    .max_iter(30)
    .run();

    assert_eq!(result_a.cost(), result_b.cost());
    assert_eq!(result_a.param(), result_b.param());
}

/// Different seeds → different trajectories. Regression-guards an
/// always-zero / always-constant RNG bug.
#[test]
fn different_seeds_yield_different_trajectories() {
    let m0 = DVector::from_vec(vec![0.5, 0.5, 0.5]);
    let lambda = CmaEs::<DVector<f64>, DMatrix<f64>>::default_lambda(3);

    let result_a = Executor::new(
        Sphere::<DVector<f64>>::new(),
        CmaEs::<DVector<f64>, DMatrix<f64>>::new(m0.clone(), 0.3, 1),
        BasicPopulationState::<DVector<f64>>::with_size(lambda),
    )
    .max_iter(5)
    .run();

    let result_b = Executor::new(
        Sphere::<DVector<f64>>::new(),
        CmaEs::<DVector<f64>, DMatrix<f64>>::new(m0, 0.3, 2),
        BasicPopulationState::<DVector<f64>>::with_size(lambda),
    )
    .max_iter(5)
    .run();

    assert_ne!(result_a.param(), result_b.param());
}

/// Sphere 5-D: the canonical "every solver should solve it cleanly"
/// canary. CMA-ES drives `σ` and `‖x‖` to zero geometrically;
/// 80 iterations × λ ≈ 8 evals/iter is far more than needed for
/// 1e-6 accuracy.
#[test]
fn converges_on_sphere_5d() {
    let m0 = DVector::from_vec(vec![1.0; 5]);
    let lambda = CmaEs::<DVector<f64>, DMatrix<f64>>::default_lambda(5);

    let result = Executor::new(
        Sphere::<DVector<f64>>::new(),
        CmaEs::<DVector<f64>, DMatrix<f64>>::new(m0, 0.5, 7),
        BasicPopulationState::<DVector<f64>>::with_size(lambda),
    )
    .max_iter(80)
    .run();

    assert!(
        result.cost() < 1e-6,
        "sphere 5-D cost = {}, expected < 1e-6",
        result.cost()
    );
}

/// Rosenbrock 2-D from `(-1, 1)`. The non-convex banana valley is the
/// canonical CMA-ES test case; the algorithm is a famous good fit for
/// this function. λ_default = 6 at n = 2; 800 iterations ≈ 4800 evals
/// is well within Hansen's ballpark for 2-D.
#[test]
fn converges_on_rosenbrock_2d() {
    let m0 = DVector::from_vec(vec![-1.0, 1.0]);
    let lambda = CmaEs::<DVector<f64>, DMatrix<f64>>::default_lambda(2);

    let result = Executor::new(
        Rosenbrock::<DVector<f64>>::new(),
        CmaEs::<DVector<f64>, DMatrix<f64>>::new(m0, 0.3, 17),
        BasicPopulationState::<DVector<f64>>::with_size(lambda),
    )
    .max_iter(800)
    .run();

    let p = result.param();
    assert!(
        (p[0] - 1.0).abs() < 1e-3 && (p[1] - 1.0).abs() < 1e-3,
        "rosenbrock 2-D iterate = ({}, {}), expected ≈ (1, 1)",
        p[0],
        p[1]
    );
}

/// Solver-internal TolX termination: with a tight tolerance, CMA-ES
/// emits `SolverConverged` on Sphere as `σ · max d_i` collapses below
/// `1e−12 · initial_sigma`.
#[test]
fn sphere_terminates_solver_converged_on_tol_x() {
    let m0 = DVector::from_vec(vec![0.5, 0.5, 0.5]);
    let lambda = CmaEs::<DVector<f64>, DMatrix<f64>>::default_lambda(3);

    let result = Executor::new(
        Sphere::<DVector<f64>>::new(),
        CmaEs::<DVector<f64>, DMatrix<f64>>::new(m0, 0.3, 11),
        BasicPopulationState::<DVector<f64>>::with_size(lambda),
    )
    .max_iter(2000)
    .run();

    assert_eq!(result.reason, TerminationReason::SolverConverged);
}

/// Sort + length invariants survive iteration. PopulationState's
/// contract requires `candidates`/`costs` to remain in parallel,
/// length-`λ`, sorted-ascending.
#[test]
fn population_invariants_hold_after_iteration() {
    let m0 = DVector::from_vec(vec![0.3, 0.4]);
    let lambda = 12;

    let mut stepper = Executor::new(
        Sphere::<DVector<f64>>::new(),
        CmaEs::<DVector<f64>, DMatrix<f64>>::new(m0, 0.5, 1234).with_lambda(lambda),
        BasicPopulationState::<DVector<f64>>::with_size(lambda),
    )
    .max_iter(10)
    .into_stepper();

    for _ in 0..10 {
        let StepOutcome::Continue = stepper.step() else {
            break;
        };
        let state = stepper.state();
        assert_eq!(state.candidates().len(), lambda);
        assert_eq!(state.costs().len(), lambda);
        for window in state.costs().windows(2) {
            assert!(window[0] <= window[1]);
        }
    }
}
