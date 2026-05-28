#![cfg(feature = "nalgebra")]

//! L-BFGS-B convergence tests over the nalgebra backend.
//!
//! Mirrors `tests/lbfgsb_vec.rs` to confirm
//! [`basin::backend::AsFloatSliceMut`]'s `DVector<f64>` impl plumbs
//! through correctly and parity with the Vec backend holds.

use basin::problems::BoothBoxed;
use basin::{
    BoxConstraints, CostFunction, Executor, Gradient, LbfgsState, MaxIter, MoreThuente,
    ProjectedGradientTolerance, QuasiNewtonState, BFGS, LBFGSB,
};
use nalgebra::{DMatrix, DVector};

struct Rosen {
    l: DVector<f64>,
    u: DVector<f64>,
}
impl CostFunction for Rosen {
    type Param = DVector<f64>;
    type Output = f64;
    fn cost(&self, x: &DVector<f64>) -> f64 {
        (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2)
    }
}
impl Gradient for Rosen {
    type Param = DVector<f64>;
    type Gradient = DVector<f64>;
    fn gradient(&self, x: &DVector<f64>) -> DVector<f64> {
        let dfdx0 = -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] * x[0]);
        let dfdx1 = 200.0 * (x[1] - x[0] * x[0]);
        DVector::from_vec(vec![dfdx0, dfdx1])
    }
}
impl basin::CostAndGradient for Rosen {}
impl BoxConstraints for Rosen {
    fn lower(&self) -> &DVector<f64> {
        &self.l
    }
    fn upper(&self) -> &DVector<f64> {
        &self.u
    }
}

#[test]
fn unbounded_rosenbrock_2d_converges() {
    let problem = Rosen {
        l: DVector::from_element(2, f64::NEG_INFINITY),
        u: DVector::from_element(2, f64::INFINITY),
    };
    let lower = problem.l.clone();
    let upper = problem.u.clone();
    let state = LbfgsState::new(DVector::from_vec(vec![-1.2, 1.0]), 5);

    let result = Executor::new(problem, LBFGSB::new(), state)
        .terminate_on(MaxIter(200))
        .terminate_on(ProjectedGradientTolerance::new(lower, upper, 1e-8))
        .run();

    assert!(result.cost() < 1e-10, "cost = {}", result.cost());
    assert!(
        (result.param()[0] - 1.0).abs() < 1e-4 && (result.param()[1] - 1.0).abs() < 1e-4,
        "x = ({}, {})",
        result.param()[0],
        result.param()[1]
    );
}

#[test]
fn booth_at_corner_converges() {
    let problem = BoothBoxed::<DVector<f64>>::new(
        DVector::from_vec(vec![-1.0, -1.0]),
        DVector::from_vec(vec![1.0, 1.0]),
    );
    let lower = DVector::from_vec(vec![-1.0, -1.0]);
    let upper = DVector::from_vec(vec![1.0, 1.0]);
    let state = LbfgsState::new(DVector::from_vec(vec![0.0, 0.0]), 5);

    let result = Executor::new(problem, LBFGSB::new(), state)
        .terminate_on(MaxIter(100))
        .terminate_on(ProjectedGradientTolerance::new(lower, upper, 1e-8))
        .run();

    assert!(
        (result.param()[0] - 1.0).abs() < 1e-5 && (result.param()[1] - 1.0).abs() < 1e-5,
        "x = ({}, {})",
        result.param()[0],
        result.param()[1]
    );
    assert!(
        (result.cost() - 20.0).abs() < 1e-8,
        "cost = {}",
        result.cost()
    );
}

#[test]
fn booth_slack_bounds_recover_unconstrained_minimum() {
    let problem = BoothBoxed::<DVector<f64>>::new(
        DVector::from_vec(vec![-5.0, -5.0]),
        DVector::from_vec(vec![5.0, 5.0]),
    );
    let lower = DVector::from_vec(vec![-5.0, -5.0]);
    let upper = DVector::from_vec(vec![5.0, 5.0]);
    let state = LbfgsState::new(DVector::from_vec(vec![0.0, 0.0]), 5);

    let result = Executor::new(problem, LBFGSB::new(), state)
        .terminate_on(MaxIter(100))
        .terminate_on(ProjectedGradientTolerance::new(lower, upper, 1e-10))
        .run();

    assert!(
        (result.param()[0] - 1.0).abs() < 1e-4 && (result.param()[1] - 3.0).abs() < 1e-4,
        "x = ({}, {})",
        result.param()[0],
        result.param()[1]
    );
    assert!(result.cost() < 1e-8, "cost = {}", result.cost());
}

/// On the unbounded Rosenbrock, L-BFGS-B with `m = 10` should
/// match BFGS-with-MoreThuente's iteration count within a small
/// constant (limited-memory ≈ full-memory once `m ≥ 2` here). Both
/// must reach machine-precision cost.
#[test]
fn lbfgsb_matches_bfgs_more_thuente_on_unbounded_rosenbrock() {
    let initial = DVector::from_vec(vec![-1.2, 1.0]);
    let l = DVector::from_element(2, f64::NEG_INFINITY);
    let u = DVector::from_element(2, f64::INFINITY);

    let bfgs_result = Executor::new(
        basin::problems::Rosenbrock::<DVector<f64>>::default(),
        BFGS::with_line_search(MoreThuente::new()),
        QuasiNewtonState::<DVector<f64>, DMatrix<f64>>::new(initial.clone()),
    )
    .max_iter(200)
    .terminate_on(basin::GradientTolerance(1e-8))
    .run();

    let lbfgsb_result = Executor::new(
        Rosen {
            l: l.clone(),
            u: u.clone(),
        },
        LBFGSB::new(),
        LbfgsState::new(initial, 10),
    )
    .max_iter(200)
    .terminate_on(ProjectedGradientTolerance::new(l, u, 1e-8))
    .run();

    assert!(
        bfgs_result.cost() < 1e-12,
        "BFGS cost = {}",
        bfgs_result.cost()
    );
    assert!(
        lbfgsb_result.cost() < 1e-10,
        "LBFGSB cost = {}",
        lbfgsb_result.cost()
    );
    // L-BFGS-B's per-iteration extra cost+grad eval (the
    // re-evaluation at the accepted step, see `next_iter`) inflates
    // its `cost_evals` slightly above BFGS's. The iteration count
    // should be comparable, though — a constant-factor slowdown
    // would indicate a wiring bug rather than the m → ∞ limit.
    assert!(
        lbfgsb_result.iter() < bfgs_result.iter() + 30,
        "LBFGSB iter {} vs BFGS iter {}",
        lbfgsb_result.iter(),
        bfgs_result.iter()
    );
}
