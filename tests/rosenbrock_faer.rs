#![cfg(feature = "faer")]

use basin::{
    Backtracking, BasicSimplexState, BasicState, CostFunction, Executor, Gradient, GradientDescent,
    NelderMead,
};
use faer::Col;

struct Rosenbrock;

impl CostFunction for Rosenbrock {
    type Param = Col<f64>;
    type Output = f64;

    fn cost(&self, x: &Col<f64>) -> f64 {
        (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2)
    }
}

impl Gradient for Rosenbrock {
    type Param = Col<f64>;
    type Gradient = Col<f64>;

    fn gradient(&self, x: &Col<f64>) -> Col<f64> {
        Col::from_fn(2, |i| match i {
            0 => -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0].powi(2)),
            1 => 200.0 * (x[1] - x[0].powi(2)),
            _ => unreachable!(),
        })
    }
}

#[test]
fn gradient_descent_with_faer_col() {
    let problem = Rosenbrock;
    let initial = Col::from_fn(2, |i| if i == 0 { -1.2 } else { 1.0 });
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
fn gradient_descent_with_faer_col_and_backtracking() {
    let problem = Rosenbrock;
    let initial = Col::from_fn(2, |i| if i == 0 { -1.2 } else { 1.0 });
    let initial_cost = problem.cost(&initial);

    let result = Executor::new(
        problem,
        GradientDescent::with_line_search(Backtracking::new()),
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
fn nelder_mead_with_faer_col() {
    let problem = Rosenbrock;
    let initial = Col::from_fn(2, |i| if i == 0 { -1.2 } else { 1.0 });

    let result = Executor::new(
        problem,
        NelderMead::adaptive(),
        BasicSimplexState::new(initial),
    )
    .max_iter(2_000)
    .run();

    assert!(result.cost() < 1e-6, "cost = {}", result.cost());
}
