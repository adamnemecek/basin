use basin::{CostFunction, Executor, NelderMead, SimplexState, TerminationReason};

struct Rosenbrock;

impl CostFunction for Rosenbrock {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, x: &Vec<f64>) -> f64 {
        (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2)
    }
}

#[test]
fn nelder_mead_standard_minimises_rosenbrock() {
    let problem = Rosenbrock;
    let initial_cost = problem.cost(&vec![-1.2, 1.0]);

    let result = Executor::new(
        problem,
        NelderMead::standard(),
        SimplexState::new(vec![-1.2, 1.0]),
    )
    .max_iter(2_000)
    .run();

    assert!(
        result.cost() < 1e-6,
        "expected near-zero cost, got {} (initial {})",
        result.cost(),
        initial_cost
    );
    let best = result.param();
    assert!((best[0] - 1.0).abs() < 1e-3, "x[0] = {}", best[0]);
    assert!((best[1] - 1.0).abs() < 1e-3, "x[1] = {}", best[1]);
}

#[test]
fn nelder_mead_adaptive_minimises_rosenbrock() {
    let problem = Rosenbrock;

    let result = Executor::new(
        problem,
        NelderMead::adaptive(),
        SimplexState::new(vec![-1.2, 1.0]),
    )
    .max_iter(2_000)
    .run();

    // For n = 2, adaptive collapses to standard, so the bar is the same.
    assert!(result.cost() < 1e-6, "cost = {}", result.cost());
}

#[test]
fn nelder_mead_hits_max_iter_when_too_few() {
    let problem = Rosenbrock;

    let result = Executor::new(
        problem,
        NelderMead::standard(),
        SimplexState::new(vec![-1.2, 1.0]),
    )
    .max_iter(5)
    .run();

    assert_eq!(result.reason, TerminationReason::MaxIter);
    assert_eq!(result.iter(), 5);
}

#[test]
fn nelder_mead_keeps_best_first_after_each_iter() {
    // Quick invariant: after run completion, costs[] is ascending.
    let problem = Rosenbrock;
    let result = Executor::new(
        problem,
        NelderMead::standard(),
        SimplexState::new(vec![-1.2, 1.0]),
    )
    .max_iter(100)
    .run();

    for w in result.state.costs.windows(2) {
        assert!(w[0] <= w[1], "simplex not sorted: {:?}", result.state.costs);
    }
}

/// Sphere function: derivative-free path on a trivial convex problem.
struct Sphere;
impl CostFunction for Sphere {
    type Param = Vec<f64>;
    type Output = f64;
    fn cost(&self, x: &Vec<f64>) -> f64 {
        x.iter().map(|v| v * v).sum()
    }
}

#[test]
fn nelder_mead_adaptive_sphere_5d() {
    let problem = Sphere;

    let result = Executor::new(
        problem,
        NelderMead::adaptive(),
        SimplexState::new(vec![1.0; 5]),
    )
    .max_iter(2_000)
    .run();

    assert!(result.cost() < 1e-8, "cost = {}", result.cost());
}

#[test]
fn nelder_mead_from_simplex_accepts_custom_geometry() {
    // Verifies the escape hatch still works for users who want a custom
    // initial simplex (regular simplex with edge length 1, not FMINSEARCH-style).
    let problem = Rosenbrock;
    let simplex = vec![vec![-1.2, 1.0], vec![-0.2, 1.0], vec![-1.2, 2.0]];

    let result = Executor::new(
        problem,
        NelderMead::standard(),
        SimplexState::from_simplex(simplex),
    )
    .max_iter(2_000)
    .run();

    assert!(result.cost() < 1e-6);
}
