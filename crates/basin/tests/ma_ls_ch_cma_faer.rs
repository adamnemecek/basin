//! Faer-backend smoke tests for [`MaLsChCma`]. Convergence on Sphere
//! and Rastrigin to confirm the per-backend trait wiring works; the
//! deeper algorithmic invariants are covered by the nalgebra mirror
//! test (`tests/ma_ls_ch_cma_nalgebra.rs`).

#![cfg(feature = "faer")]

use basin::problems::RastriginBoxed;
use basin::{CostFunction, Executor, MaLsChCma, MaLsChState, MaxCostEvals};
use faer::{Col, Mat};

struct BoxedSphere {
    lower: Col<f64>,
    upper: Col<f64>,
}

impl BoxedSphere {
    fn new(n: usize, half_width: f64) -> Self {
        Self {
            lower: Col::from_fn(n, |_| -half_width),
            upper: Col::from_fn(n, |_| half_width),
        }
    }
}

impl CostFunction for BoxedSphere {
    type Param = Col<f64>;
    type Output = f64;
    fn cost(&self, x: &Col<f64>) -> f64 {
        x.iter().map(|v| v * v).sum()
    }
}

impl basin::BoxConstraints for BoxedSphere {
    fn lower(&self) -> &Col<f64> {
        &self.lower
    }
    fn upper(&self) -> &Col<f64> {
        &self.upper
    }
}

#[test]
fn converges_on_sphere_d10() {
    let problem = BoxedSphere::new(10, 5.0);
    let solver = MaLsChCma::<Col<f64>, Mat<f64>>::new(7).with_pop_size(20);
    let result = Executor::new(problem, solver, MaLsChState::new())
        .max_iter(u64::MAX)
        .terminate_on(MaxCostEvals(20_000))
        .run();

    assert!(
        result.cost() < 1e-6,
        "Sphere(D=10) faer cost {} should be < 1e-6 within 20k evals",
        result.cost()
    );
}

#[test]
fn converges_on_rastrigin_d10() {
    let problem = RastriginBoxed::<Col<f64>>::with_standard_bounds(10);
    let solver = MaLsChCma::<Col<f64>, Mat<f64>>::new(42).with_pop_size(30);
    let result = Executor::new(problem, solver, MaLsChState::new())
        .max_iter(u64::MAX)
        .terminate_on(MaxCostEvals(50_000))
        .run();

    assert!(
        result.cost() < 1.0,
        "Rastrigin(D=10) faer cost {} should be < 1.0 within 50k evals",
        result.cost()
    );
}
