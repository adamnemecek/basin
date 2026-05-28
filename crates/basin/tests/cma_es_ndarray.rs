#![cfg(feature = "ndarray")]

use basin::problems::{Rosenbrock, Sphere};
use basin::{
    BasicPopulationState, CmaEs, CostFunction, Executor, PopulationState, StepOutcome,
    TerminationReason,
};
use ndarray::{Array1, Array2};

/// Same seed → same trajectory, on the ndarray backend. Reproducibility
/// is the load-bearing contract for the seeded RNG (per
/// [`crate::core::rng`] docs); regression-guards a constant-RNG bug.
#[test]
fn same_seed_yields_identical_trajectory() {
    let m0 = Array1::from_vec(vec![0.5, 0.5, 0.5, 0.5, 0.5]);

    let result_a = Executor::new(
        Sphere::<Array1<f64>>::new(),
        CmaEs::<Array1<f64>, Array2<f64>>::new(m0.clone(), 0.3, 42),
        BasicPopulationState::<Array1<f64>>::with_size(
            CmaEs::<Array1<f64>, Array2<f64>>::default_lambda(5),
        ),
    )
    .max_iter(30)
    .run();

    let result_b = Executor::new(
        Sphere::<Array1<f64>>::new(),
        CmaEs::<Array1<f64>, Array2<f64>>::new(m0, 0.3, 42),
        BasicPopulationState::<Array1<f64>>::with_size(
            CmaEs::<Array1<f64>, Array2<f64>>::default_lambda(5),
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
    let m0 = Array1::from_vec(vec![0.5, 0.5, 0.5]);
    let lambda = CmaEs::<Array1<f64>, Array2<f64>>::default_lambda(3);

    let result_a = Executor::new(
        Sphere::<Array1<f64>>::new(),
        CmaEs::<Array1<f64>, Array2<f64>>::new(m0.clone(), 0.3, 1),
        BasicPopulationState::<Array1<f64>>::with_size(lambda),
    )
    .max_iter(5)
    .run();

    let result_b = Executor::new(
        Sphere::<Array1<f64>>::new(),
        CmaEs::<Array1<f64>, Array2<f64>>::new(m0, 0.3, 2),
        BasicPopulationState::<Array1<f64>>::with_size(lambda),
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
    let m0 = Array1::from_elem(5, 1.0);
    let lambda = CmaEs::<Array1<f64>, Array2<f64>>::default_lambda(5);

    let result = Executor::new(
        Sphere::<Array1<f64>>::new(),
        CmaEs::<Array1<f64>, Array2<f64>>::new(m0, 0.5, 7),
        BasicPopulationState::<Array1<f64>>::with_size(lambda),
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
    let m0 = Array1::from_vec(vec![-1.0, 1.0]);
    let lambda = CmaEs::<Array1<f64>, Array2<f64>>::default_lambda(2);

    let result = Executor::new(
        Rosenbrock::<Array1<f64>>::new(),
        CmaEs::<Array1<f64>, Array2<f64>>::new(m0, 0.3, 17),
        BasicPopulationState::<Array1<f64>>::with_size(lambda),
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
    let m0 = Array1::from_vec(vec![0.5, 0.5, 0.5]);
    let lambda = CmaEs::<Array1<f64>, Array2<f64>>::default_lambda(3);

    let result = Executor::new(
        Sphere::<Array1<f64>>::new(),
        CmaEs::<Array1<f64>, Array2<f64>>::new(m0, 0.3, 11),
        BasicPopulationState::<Array1<f64>>::with_size(lambda),
    )
    .max_iter(2000)
    .run();

    assert_eq!(result.reason, TerminationReason::SolverConverged);
}

/// `with_stds(ones)` must reproduce the isotropic `C = I` default
/// bit-for-bit: the only difference is the first-generation sampling path
/// (`m + σ B (D ⊙ z)` vs `m + σ z`), which is exactly identical when
/// `B = I` and `D = 1`. Guards the bit-identity claim that lets existing
/// users keep today's behavior unchanged.
#[test]
fn with_stds_ones_matches_default() {
    let m0 = Array1::from_vec(vec![0.5, 0.5, 0.5, 0.5, 0.5]);
    let lambda = CmaEs::<Array1<f64>, Array2<f64>>::default_lambda(5);
    let ones = Array1::from_elem(5, 1.0);

    let default = Executor::new(
        Sphere::<Array1<f64>>::new(),
        CmaEs::<Array1<f64>, Array2<f64>>::new(m0.clone(), 0.3, 42),
        BasicPopulationState::<Array1<f64>>::with_size(lambda),
    )
    .max_iter(40)
    .run();

    let with_ones = Executor::new(
        Sphere::<Array1<f64>>::new(),
        CmaEs::<Array1<f64>, Array2<f64>>::new(m0, 0.3, 42).with_stds(ones),
        BasicPopulationState::<Array1<f64>>::with_size(lambda),
    )
    .max_iter(40)
    .run();

    assert_eq!(default.cost(), with_ones.cost());
    assert_eq!(default.param(), with_ones.param());
    assert_eq!(default.reason, with_ones.reason);
}

/// Anisotropic stds on the (well-conditioned) Sphere must still converge:
/// per-coordinate scaling rescales the initial distribution but does not
/// break the adaptation. Correctness check, not a speedup claim.
#[test]
fn with_stds_anisotropic_converges_on_sphere() {
    let m0 = Array1::from_elem(5, 1.0);
    let lambda = CmaEs::<Array1<f64>, Array2<f64>>::default_lambda(5);
    let stds = Array1::from_vec(vec![1.0, 0.1, 10.0, 0.5, 2.0]);

    let result = Executor::new(
        Sphere::<Array1<f64>>::new(),
        CmaEs::<Array1<f64>, Array2<f64>>::new(m0, 0.5, 7).with_stds(stds),
        BasicPopulationState::<Array1<f64>>::with_size(lambda),
    )
    .max_iter(120)
    .run();

    assert!(
        result.cost() < 1e-6,
        "anisotropic sphere 5-D cost = {}, expected < 1e-6",
        result.cost()
    );
}

/// The motivating case: an ill-scaled quadratic `f(x) = x₀² + 1e6 · x₁²`.
/// Preconditioning with `with_stds([1, 1e-3])` rescales the search to ~unit
/// conditioning, so CMA-ES reaches a low cost within a modest budget
/// instead of spending generations learning the 1e6 scale ratio.
#[test]
fn with_stds_preconditions_ill_scaled_quadratic() {
    struct IllScaledQuadratic;
    impl CostFunction for IllScaledQuadratic {
        type Param = Array1<f64>;
        type Output = f64;
        fn cost(&self, x: &Array1<f64>) -> f64 {
            x[0] * x[0] + 1e6 * x[1] * x[1]
        }
    }

    let m0 = Array1::from_vec(vec![2.0, 2.0]);
    let lambda = CmaEs::<Array1<f64>, Array2<f64>>::default_lambda(2);
    let stds = Array1::from_vec(vec![1.0, 1e-3]);

    let result = Executor::new(
        IllScaledQuadratic,
        CmaEs::<Array1<f64>, Array2<f64>>::new(m0, 0.5, 7).with_stds(stds),
        BasicPopulationState::<Array1<f64>>::with_size(lambda),
    )
    .max_iter(300)
    .run();

    assert!(
        result.cost() < 1e-6,
        "preconditioned ill-scaled quadratic cost = {}, expected < 1e-6",
        result.cost()
    );
}

/// `with_stds` panics when the std vector length doesn't match the mean.
#[test]
#[should_panic(expected = "stds.len() == initial_mean.len()")]
fn with_stds_panics_on_length_mismatch() {
    let m0 = Array1::from_vec(vec![0.0, 0.0, 0.0]);
    let _ = CmaEs::<Array1<f64>, Array2<f64>>::new(m0, 0.3, 42)
        .with_stds(Array1::from_vec(vec![1.0, 1.0]));
}

/// `with_stds` panics on a non-positive std (would make `1/std` non-finite
/// in the `C^{−1/2}` factor).
#[test]
#[should_panic(expected = "every std > 0")]
fn with_stds_panics_on_nonpositive() {
    let m0 = Array1::from_vec(vec![0.0, 0.0]);
    let _ = CmaEs::<Array1<f64>, Array2<f64>>::new(m0, 0.3, 42)
        .with_stds(Array1::from_vec(vec![1.0, 0.0]));
}

/// Sort + length invariants survive iteration. PopulationState's
/// contract requires `candidates`/`costs` to remain in parallel,
/// length-`λ`, sorted-ascending.
#[test]
fn population_invariants_hold_after_iteration() {
    let m0 = Array1::from_vec(vec![0.3, 0.4]);
    let lambda = 12;

    let mut stepper = Executor::new(
        Sphere::<Array1<f64>>::new(),
        CmaEs::<Array1<f64>, Array2<f64>>::new(m0, 0.5, 1234).with_lambda(lambda),
        BasicPopulationState::<Array1<f64>>::with_size(lambda),
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
