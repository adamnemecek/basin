use basin::problems::Rosenbrock;
use basin::{Backtracking, BasicState, CostFunction, Executor, GradientDescent, TerminationReason};

#[test]
fn gradient_descent_decreases_rosenbrock_cost() {
    let problem = Rosenbrock::<Vec<f64>>::default();
    let initial = vec![-1.2, 1.0];
    let initial_cost = problem.cost(&initial);

    let result = Executor::new(
        problem,
        GradientDescent::new(0.001),
        BasicState::new(initial),
    )
    .max_iter(10_000)
    .run();

    assert_eq!(result.iter(), 10_000, "should hit max_iter");
    assert_eq!(result.reason, TerminationReason::MaxIter);
    assert!(
        result.cost() < initial_cost * 0.1,
        "expected cost to drop by >10x: initial={}, final={}",
        initial_cost,
        result.cost()
    );
}

#[test]
fn gradient_descent_with_backtracking_decreases_rosenbrock_cost() {
    let problem = Rosenbrock::<Vec<f64>>::default();
    let initial = vec![-1.2, 1.0];
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
