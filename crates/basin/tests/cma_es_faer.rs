#![cfg(feature = "faer")]

use basin::problems::{Rosenbrock, Sphere};
use basin::{
    BasicPopulationState, CmaEs, Executor, PopulationState, StepOutcome, TerminationReason,
};
use faer::{Col, Mat};

/// Same seed → same trajectory, on the faer backend. Reproducibility
/// across faer's `Col<f64>` / `Mat<f64>` types is independent of the
/// nalgebra trajectory (different eigendecomposition routine, possibly
/// different eigenvector ordering) — same-backend reproducibility is
/// the load-bearing contract.
#[test]
fn same_seed_yields_identical_trajectory() {
    let m0 = Col::<f64>::from_fn(5, |_| 0.5);

    let result_a = Executor::new(
        Sphere::<Col<f64>>::new(),
        CmaEs::<Col<f64>, Mat<f64>>::new(m0.clone(), 0.3, 42),
        BasicPopulationState::<Col<f64>>::with_size(CmaEs::<Col<f64>, Mat<f64>>::default_lambda(5)),
    )
    .max_iter(30)
    .run()
    .unwrap();

    let result_b = Executor::new(
        Sphere::<Col<f64>>::new(),
        CmaEs::<Col<f64>, Mat<f64>>::new(m0, 0.3, 42),
        BasicPopulationState::<Col<f64>>::with_size(CmaEs::<Col<f64>, Mat<f64>>::default_lambda(5)),
    )
    .max_iter(30)
    .run()
    .unwrap();

    assert_eq!(result_a.cost(), result_b.cost());
    let (a, b) = (result_a.param(), result_b.param());
    assert_eq!(a.nrows(), b.nrows());
    for i in 0..a.nrows() {
        assert_eq!(a[i], b[i]);
    }
}

/// Sphere 5-D convergence on the faer backend. Same target as the
/// nalgebra test; iterate-trajectory may differ (different eigen
/// decomposition under the hood) but both converge.
#[test]
fn converges_on_sphere_5d() {
    let m0 = Col::<f64>::from_fn(5, |_| 1.0);
    let lambda = CmaEs::<Col<f64>, Mat<f64>>::default_lambda(5);

    let result = Executor::new(
        Sphere::<Col<f64>>::new(),
        CmaEs::<Col<f64>, Mat<f64>>::new(m0, 0.5, 7),
        BasicPopulationState::<Col<f64>>::with_size(lambda),
    )
    .max_iter(80)
    .run()
    .unwrap();

    assert!(
        result.cost() < 1e-6,
        "sphere 5-D cost = {}, expected < 1e-6",
        result.cost()
    );
}

/// Rosenbrock 2-D convergence on the faer backend.
#[test]
fn converges_on_rosenbrock_2d() {
    let m0 = Col::<f64>::from_fn(2, |i| if i == 0 { -1.0 } else { 1.0 });
    let lambda = CmaEs::<Col<f64>, Mat<f64>>::default_lambda(2);

    let result = Executor::new(
        Rosenbrock::<Col<f64>>::new(),
        CmaEs::<Col<f64>, Mat<f64>>::new(m0, 0.3, 17),
        BasicPopulationState::<Col<f64>>::with_size(lambda),
    )
    .max_iter(800)
    .run()
    .unwrap();

    let p = result.param();
    assert!(
        (p[0] - 1.0).abs() < 1e-3 && (p[1] - 1.0).abs() < 1e-3,
        "rosenbrock 2-D iterate = ({}, {}), expected ≈ (1, 1)",
        p[0],
        p[1]
    );
}

/// Solver-internal TolX termination on the faer backend.
#[test]
fn sphere_terminates_solver_converged_on_tol_x() {
    let m0 = Col::<f64>::from_fn(3, |_| 0.5);
    let lambda = CmaEs::<Col<f64>, Mat<f64>>::default_lambda(3);

    let result = Executor::new(
        Sphere::<Col<f64>>::new(),
        CmaEs::<Col<f64>, Mat<f64>>::new(m0, 0.3, 11),
        BasicPopulationState::<Col<f64>>::with_size(lambda),
    )
    .max_iter(2000)
    .run()
    .unwrap();

    assert_eq!(result.reason, TerminationReason::SolverConverged);
}

/// `with_stds(ones)` reproduces the isotropic default bit-for-bit on the
/// faer backend (the identity matvec / unit component-mul are exact, so
/// the `m + σ B (D ⊙ z)` path equals `m + σ z`).
#[test]
fn with_stds_ones_matches_default() {
    let m0 = Col::<f64>::from_fn(5, |_| 0.5);
    let lambda = CmaEs::<Col<f64>, Mat<f64>>::default_lambda(5);
    let ones = Col::<f64>::from_fn(5, |_| 1.0);

    let default = Executor::new(
        Sphere::<Col<f64>>::new(),
        CmaEs::<Col<f64>, Mat<f64>>::new(m0.clone(), 0.3, 42),
        BasicPopulationState::<Col<f64>>::with_size(lambda),
    )
    .max_iter(40)
    .run()
    .unwrap();

    let with_ones = Executor::new(
        Sphere::<Col<f64>>::new(),
        CmaEs::<Col<f64>, Mat<f64>>::new(m0, 0.3, 42).with_stds(ones),
        BasicPopulationState::<Col<f64>>::with_size(lambda),
    )
    .max_iter(40)
    .run()
    .unwrap();

    assert_eq!(default.cost(), with_ones.cost());
    let (a, b) = (default.param(), with_ones.param());
    assert_eq!(a.nrows(), b.nrows());
    for i in 0..a.nrows() {
        assert_eq!(a[i], b[i]);
    }
    assert_eq!(default.reason, with_ones.reason);
}

/// Anisotropic stds still converge on Sphere (faer backend).
#[test]
fn with_stds_anisotropic_converges_on_sphere() {
    let m0 = Col::<f64>::from_fn(5, |_| 1.0);
    let lambda = CmaEs::<Col<f64>, Mat<f64>>::default_lambda(5);
    let stds = Col::<f64>::from_fn(5, |i| [1.0, 0.1, 10.0, 0.5, 2.0][i]);

    let result = Executor::new(
        Sphere::<Col<f64>>::new(),
        CmaEs::<Col<f64>, Mat<f64>>::new(m0, 0.5, 7).with_stds(stds),
        BasicPopulationState::<Col<f64>>::with_size(lambda),
    )
    .max_iter(120)
    .run()
    .unwrap();

    assert!(
        result.cost() < 1e-6,
        "anisotropic sphere 5-D cost = {}, expected < 1e-6",
        result.cost()
    );
}

/// PopulationState invariants survive iteration on faer.
#[test]
fn population_invariants_hold_after_iteration() {
    let m0 = Col::<f64>::from_fn(2, |i| if i == 0 { 0.3 } else { 0.4 });
    let lambda = 12;

    let mut stepper = Executor::new(
        Sphere::<Col<f64>>::new(),
        CmaEs::<Col<f64>, Mat<f64>>::new(m0, 0.5, 1234).with_lambda(lambda),
        BasicPopulationState::<Col<f64>>::with_size(lambda),
    )
    .max_iter(10)
    .into_stepper()
    .unwrap();

    for _ in 0..10 {
        let StepOutcome::Continue = stepper.step().unwrap() else {
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
