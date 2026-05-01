use basin::{Backtracking, BasicState, CostFunction, Executor, Gradient, GradientDescent};
use nalgebra::DVector;

struct Rosenbrock;

impl CostFunction for Rosenbrock {
    type Param = DVector<f64>;
    type Output = f64;

    fn cost(&self, x: &DVector<f64>) -> f64 {
        (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2)
    }
}

impl Gradient for Rosenbrock {
    type Param = DVector<f64>;
    type Gradient = DVector<f64>;

    fn gradient(&self, x: &DVector<f64>) -> DVector<f64> {
        DVector::from_vec(vec![
            -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0].powi(2)),
            200.0 * (x[1] - x[0].powi(2)),
        ])
    }
}

#[test]
fn gradient_descent_with_nalgebra_dvector() {
    let problem = Rosenbrock;
    let initial = DVector::from_vec(vec![-1.2, 1.0]);
    let initial_cost = problem.cost(&initial);

    let result = Executor::new(
        problem,
        GradientDescent::new(0.001),
        BasicState::new(initial),
    )
    .max_iter(10_000)
    .run();

    assert!(
        result.cost() < initial_cost * 0.1,
        "expected cost to drop by >10x: initial={}, final={}",
        initial_cost,
        result.cost()
    );
}

#[test]
fn gradient_descent_with_nalgebra_dvector_and_backtracking() {
    let problem = Rosenbrock;
    let initial = DVector::from_vec(vec![-1.2, 1.0]);
    let initial_cost = problem.cost(&initial);

    let result = Executor::new(
        problem,
        GradientDescent::with_step_size(Backtracking::new()),
        BasicState::new(initial),
    )
    .max_iter(10_000)
    .run();

    assert!(
        result.cost() < initial_cost * 0.1,
        "expected cost to drop by >10x: initial={}, final={}",
        initial_cost,
        result.cost()
    );
}
