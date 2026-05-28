//! Fused-trait opt-in tests for `CostAndGradient`,
//! `CostAndGradientAndHessian`, and `ResidualAndJacobian`.
//!
//! Three things to verify:
//! 1. The defaulted `cost_and_gradient` body produces the same `(cost,
//!    gradient)` pair as separate `cost(x)` + `gradient(x)` calls.
//! 2. A problem that *overrides* the fused method is actually called by
//!    a migrated solver (proven via a shared counter).
//! 3. `FiniteDiff` exposes the fused traits so a values-only problem
//!    still flows into the migrated solver bounds.

use std::cell::Cell;
use std::rc::Rc;

use basin::{
    BasicState, CostAndGradient, CostFunction, Executor, FiniteDiff, Gradient, GradientDescent,
    GradientTolerance, MaxIter,
};
#[cfg(feature = "nalgebra")]
use basin::{CostAndGradientAndHessian, Hessian};

// ---------------------------------------------------------------------
// 1. Defaulted body equals separate calls.
// ---------------------------------------------------------------------

struct Sphere;
impl CostFunction for Sphere {
    type Param = Vec<f64>;
    type Output = f64;
    fn cost(&self, x: &Vec<f64>) -> f64 {
        x.iter().map(|xi| xi * xi).sum()
    }
}
impl Gradient for Sphere {
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;
    fn gradient(&self, x: &Vec<f64>) -> Vec<f64> {
        x.iter().map(|xi| 2.0 * xi).collect()
    }
}
impl CostAndGradient for Sphere {}

#[test]
fn default_fused_matches_separate_calls() {
    let p = Sphere;
    let x = vec![1.5, -2.25, 0.5];
    let (fused_cost, fused_grad) = p.cost_and_gradient(&x);
    assert_eq!(fused_cost, p.cost(&x));
    assert_eq!(fused_grad, p.gradient(&x));
}

// ---------------------------------------------------------------------
// 2. Solver actually calls the overridden fused method.
// ---------------------------------------------------------------------

struct Counted {
    fused_calls: Rc<Cell<usize>>,
}
impl CostFunction for Counted {
    type Param = Vec<f64>;
    type Output = f64;
    fn cost(&self, x: &Vec<f64>) -> f64 {
        x.iter().map(|xi| xi * xi).sum()
    }
}
impl Gradient for Counted {
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;
    fn gradient(&self, x: &Vec<f64>) -> Vec<f64> {
        x.iter().map(|xi| 2.0 * xi).collect()
    }
}
impl CostAndGradient for Counted {
    fn cost_and_gradient(&self, x: &Vec<f64>) -> (f64, Vec<f64>) {
        self.fused_calls.set(self.fused_calls.get() + 1);
        let c = x.iter().map(|xi| xi * xi).sum();
        let g = x.iter().map(|xi| 2.0 * xi).collect();
        (c, g)
    }
}

#[test]
fn migrated_solver_calls_fused_override() {
    let counter = Rc::new(Cell::new(0_usize));
    let problem = Counted {
        fused_calls: counter.clone(),
    };
    let solver = GradientDescent::new(0.1);
    let state = BasicState::new(vec![1.0, 1.0]);
    let result = Executor::new(problem, solver, state)
        .terminate_on(MaxIter(200))
        .terminate_on(GradientTolerance(1e-10))
        .run();

    // Each GD iteration (init + next_iter) goes through the fused call,
    // so the counter is at least `iter + 1`.
    let n = result.iter() as usize + 1;
    let calls = counter.get();
    assert!(calls >= n, "expected at least {n} fused calls, got {calls}");
    assert!(result.cost() < 1e-8, "got cost {}", result.cost());
}

// ---------------------------------------------------------------------
// 3. CostAndGradientAndHessian default body equals separate calls.
// ---------------------------------------------------------------------

#[cfg(feature = "nalgebra")]
mod hessian {
    use super::*;
    use nalgebra::{DMatrix, DVector};

    struct SphereN;
    impl CostFunction for SphereN {
        type Param = DVector<f64>;
        type Output = f64;
        fn cost(&self, x: &DVector<f64>) -> f64 {
            x.dot(x)
        }
    }
    impl Gradient for SphereN {
        type Param = DVector<f64>;
        type Gradient = DVector<f64>;
        fn gradient(&self, x: &DVector<f64>) -> DVector<f64> {
            2.0 * x
        }
    }
    impl Hessian for SphereN {
        type Param = DVector<f64>;
        type Output = DMatrix<f64>;
        fn hessian(&self, x: &DVector<f64>) -> DMatrix<f64> {
            2.0 * DMatrix::identity(x.len(), x.len())
        }
    }
    impl CostAndGradientAndHessian for SphereN {}

    #[test]
    fn default_triple_matches_separate_calls() {
        let p = SphereN;
        let x = DVector::from_vec(vec![1.0, -2.0, 3.0]);
        let (c, g, h) = p.cost_and_gradient_and_hessian(&x);
        assert_eq!(c, p.cost(&x));
        assert_eq!(g, p.gradient(&x));
        assert_eq!(h, p.hessian(&x));
    }

    /// Verify a problem can override the fused triple. No Newton solver
    /// ships yet, so exercise the override directly.
    struct CountedTriple {
        fused_calls: Rc<Cell<usize>>,
    }
    impl CostFunction for CountedTriple {
        type Param = DVector<f64>;
        type Output = f64;
        fn cost(&self, x: &DVector<f64>) -> f64 {
            x.dot(x)
        }
    }
    impl Gradient for CountedTriple {
        type Param = DVector<f64>;
        type Gradient = DVector<f64>;
        fn gradient(&self, x: &DVector<f64>) -> DVector<f64> {
            2.0 * x
        }
    }
    impl Hessian for CountedTriple {
        type Param = DVector<f64>;
        type Output = DMatrix<f64>;
        fn hessian(&self, x: &DVector<f64>) -> DMatrix<f64> {
            2.0 * DMatrix::identity(x.len(), x.len())
        }
    }
    impl CostAndGradientAndHessian for CountedTriple {
        fn cost_and_gradient_and_hessian(
            &self,
            x: &DVector<f64>,
        ) -> (f64, DVector<f64>, DMatrix<f64>) {
            self.fused_calls.set(self.fused_calls.get() + 1);
            let c = x.dot(x);
            let g = 2.0 * x;
            let h = 2.0 * DMatrix::identity(x.len(), x.len());
            (c, g, h)
        }
    }

    #[test]
    fn override_triple_is_called() {
        let counter = Rc::new(Cell::new(0_usize));
        let p = CountedTriple {
            fused_calls: counter.clone(),
        };
        let x = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let _ = p.cost_and_gradient_and_hessian(&x);
        let _ = p.cost_and_gradient_and_hessian(&x);
        assert_eq!(counter.get(), 2);
    }
}

// ---------------------------------------------------------------------
// 4. ResidualAndJacobian default body equals separate calls, and LM
//    actually calls it.
// ---------------------------------------------------------------------

#[cfg(feature = "nalgebra")]
mod lsq {
    use super::*;
    use basin::{
        BasicState, Executor, Jacobian, LevenbergMarquardt, MaxIter, Residual, ResidualAndJacobian,
    };
    use nalgebra::{DMatrix, DVector};

    /// `r(x) = (x₀ − 1, x₁ − 2)`, J = I. Minimum at (1, 2).
    struct Affine {
        fused_calls: Rc<Cell<usize>>,
    }
    impl CostFunction for Affine {
        type Param = DVector<f64>;
        type Output = f64;
        fn cost(&self, x: &DVector<f64>) -> f64 {
            0.5 * ((x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2))
        }
    }
    impl Residual for Affine {
        type Param = DVector<f64>;
        type Output = DVector<f64>;
        fn residual(&self, x: &DVector<f64>) -> DVector<f64> {
            DVector::from_vec(vec![x[0] - 1.0, x[1] - 2.0])
        }
    }
    impl Jacobian for Affine {
        type Param = DVector<f64>;
        type Output = DMatrix<f64>;
        fn jacobian(&self, _x: &DVector<f64>) -> DMatrix<f64> {
            DMatrix::identity(2, 2)
        }
    }
    impl ResidualAndJacobian for Affine {
        fn residual_and_jacobian(&self, x: &DVector<f64>) -> (DVector<f64>, DMatrix<f64>) {
            self.fused_calls.set(self.fused_calls.get() + 1);
            (
                DVector::from_vec(vec![x[0] - 1.0, x[1] - 2.0]),
                DMatrix::identity(2, 2),
            )
        }
    }

    #[test]
    fn lm_calls_fused_residual_and_jacobian() {
        let counter = Rc::new(Cell::new(0_usize));
        let problem = Affine {
            fused_calls: counter.clone(),
        };
        let state = BasicState::new(DVector::from_vec(vec![0.0, 0.0]));
        let solver: LevenbergMarquardt<DVector<f64>, DMatrix<f64>> = LevenbergMarquardt::new();
        let result = Executor::new(problem, solver, state)
            .terminate_on(MaxIter(10))
            .run();

        // LM init does one fused call; subsequent iters reuse caches.
        assert!(counter.get() >= 1);
        let x = result.param();
        assert!((x[0] - 1.0).abs() < 1e-8);
        assert!((x[1] - 2.0).abs() < 1e-8);
    }
}

// ---------------------------------------------------------------------
// 5. `FiniteDiff` is usable with the migrated solver bound.
// ---------------------------------------------------------------------

/// Cost-only problem (no `Gradient` impl).
struct CostOnly;
impl CostFunction for CostOnly {
    type Param = Vec<f64>;
    type Output = f64;
    fn cost(&self, x: &Vec<f64>) -> f64 {
        x.iter().map(|xi| (xi - 1.0).powi(2)).sum()
    }
}

#[test]
fn finite_diff_satisfies_cost_and_gradient_bound() {
    let problem = FiniteDiff::new(CostOnly);
    let solver = GradientDescent::new(0.1);
    let state = BasicState::new(vec![0.0, 0.0]);
    let result = Executor::new(problem, solver, state)
        .terminate_on(MaxIter(200))
        .terminate_on(GradientTolerance(1e-8))
        .run();

    let x = result.param();
    assert!((x[0] - 1.0).abs() < 1e-3);
    assert!((x[1] - 1.0).abs() < 1e-3);
}
