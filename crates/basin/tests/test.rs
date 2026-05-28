//! A small, self-contained tour of the linear-inequality / log-barrier API.
//!
//! Run with: `cargo test --features nalgebra --test test -- --nocapture`
//!
//! It minimises `f(x) = ‖x − (2,2)‖²` subject to the linear inequality
//! `x₀ + x₁ ≤ 2`. The unconstrained minimum `(2,2)` is infeasible, so the
//! constrained optimum is the projection onto the line `x₀ + x₁ = 2` nearest
//! `(2,2)` — namely `(1, 1)`.
#![cfg(feature = "nalgebra")]

use basin::{
    Backtracking, BarrierMethod, BasicState, CostFunction, Executor, Gradient, GradientDescent,
    LinearInequalityConstraints, TerminationReason,
};
use nalgebra::{DMatrix, DVector};

// 1. Define the problem. The objective and its gradient are the usual
//    `CostFunction` / `Gradient` traits; the constraints are the new
//    `LinearInequalityConstraints` trait, which just hands the solver the
//    matrix `A` and vector `b` of `A x ≤ b`.
struct MyProblem {
    a: DMatrix<f64>,
    b: DVector<f64>,
}

// 1a. The objective value, f(x) = (x₀−2)² + (x₁−2)².
impl CostFunction for MyProblem {
    type Param = DVector<f64>;
    type Output = f64;
    fn cost(&self, x: &DVector<f64>) -> f64 {
        (x[0] - 2.0).powi(2) + (x[1] - 2.0).powi(2)
    }
}

// 1b. Its gradient, ∇f(x) = 2(x − 2).
impl Gradient for MyProblem {
    type Param = DVector<f64>;
    type Gradient = DVector<f64>;
    fn gradient(&self, x: &DVector<f64>) -> DVector<f64> {
        DVector::from_vec(vec![2.0 * (x[0] - 2.0), 2.0 * (x[1] - 2.0)])
    }
}
impl basin::CostAndGradient for MyProblem {}

// 1c. The constraints, in standard form `A x ≤ b`. Implementing *this* trait
//     (and not a box-bounds trait) is what routes the problem to the barrier
//     method at compile time. `b` lives in ℝᵐ (one entry per constraint row).
impl LinearInequalityConstraints for MyProblem {
    type Matrix = DMatrix<f64>;
    fn a(&self) -> &DMatrix<f64> {
        &self.a
    }
    fn b(&self) -> &DVector<f64> {
        &self.b
    }
}

#[test]
fn barrier_method_tour() {
    let problem = MyProblem {
        a: DMatrix::from_row_slice(1, 2, &[1.0, 1.0]), // one row: x₀ + x₁
        b: DVector::from_vec(vec![2.0]),               // ... ≤ 2
    };

    // 2. Choose an inner unconstrained solver. The barrier objective is +∞
    //    outside the feasible set, so the inner line search must reject such
    //    steps — Armijo backtracking does. (Strong-Wolfe searches or momentum
    //    can breach the barrier; see the `BarrierMethod` docs.)
    let inner = GradientDescent::with_line_search(Backtracking::new());

    // 3. Wrap it in the barrier method. The builder calls below are all
    //    defaults, shown explicitly for the tour.
    let solver = BarrierMethod::new(inner)
        .mu0(1.0) // initial barrier weight μ
        .reduction(10.0) // μ ← μ / 10 each outer iteration
        .tol(1e-8) // stop once the duality gap m·μ ≤ tol
        .inner_max_iter(50); // budget per inner barrier solve (the cost lever)

    // 4. The starting point must be *strictly feasible* (`A x₀ < b`). Here
    //    (0,0) gives slack 2 > 0. An infeasible start returns `SolverFailed`.
    let x0 = BasicState::new(DVector::from_vec(vec![0.0, 0.0]));

    // 5. Drive it with the usual `Executor`. `max_iter` is only an outer
    //    safety net — convergence comes from the gap test inside the solver.
    let result = Executor::new(problem, solver, x0).max_iter(50).run();

    // 6. Inspect the result.
    let x = result.param();
    println!(
        "reason = {:?}, optimum ≈ ({:.5}, {:.5}), f = {:.5}",
        result.reason,
        x[0],
        x[1],
        result.cost(),
    );

    assert_eq!(result.reason, TerminationReason::SolverConverged);
    assert!((x[0] - 1.0).abs() < 1e-4 && (x[1] - 1.0).abs() < 1e-4);
}
