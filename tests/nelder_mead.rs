use basin::{CostFunction, Executor, NelderMead, SimplexState, State, TerminationReason};

struct Rosenbrock;

impl CostFunction for Rosenbrock {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, x: &Vec<f64>) -> f64 {
        (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2)
    }
}

/// Build an FMINSEARCH-style initial simplex around `x0`: each non-zero
/// coordinate `i` gets perturbed to `1.05 * x0[i]`, zero coordinates get
/// `0.00025`.
fn fminsearch_simplex(x0: &[f64]) -> Vec<Vec<f64>> {
    let n = x0.len();
    let mut simplex = vec![x0.to_vec()];
    for i in 0..n {
        let mut v = x0.to_vec();
        v[i] = if x0[i] != 0.0 { 1.05 * x0[i] } else { 0.00025 };
        simplex.push(v);
    }
    simplex
}

#[test]
fn nelder_mead_standard_minimises_rosenbrock() {
    let problem = Rosenbrock;
    let x0 = vec![-1.2, 1.0];
    let initial_cost = problem.cost(&x0);

    let (result, _reason) = Executor::new(
        problem,
        NelderMead::standard(),
        SimplexState::new(fminsearch_simplex(&x0)),
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
    let x0 = vec![-1.2, 1.0];

    let (result, _reason) = Executor::new(
        problem,
        NelderMead::adaptive(2),
        SimplexState::new(fminsearch_simplex(&x0)),
    )
    .max_iter(2_000)
    .run();

    // For n = 2, adaptive collapses to standard, so the bar is the same.
    assert!(result.cost() < 1e-6, "cost = {}", result.cost());
}

#[test]
fn nelder_mead_hits_max_iter_when_too_few() {
    let problem = Rosenbrock;
    let x0 = vec![-1.2, 1.0];

    let (result, reason) = Executor::new(
        problem,
        NelderMead::standard(),
        SimplexState::new(fminsearch_simplex(&x0)),
    )
    .max_iter(5)
    .run();

    assert_eq!(reason, TerminationReason::MaxIter);
    assert_eq!(result.iter, 5);
}

#[test]
fn nelder_mead_keeps_best_first_after_each_iter() {
    // Quick invariant: after run completion, costs[] is ascending.
    let problem = Rosenbrock;
    let x0 = vec![-1.2, 1.0];
    let (result, _reason) = Executor::new(
        problem,
        NelderMead::standard(),
        SimplexState::new(fminsearch_simplex(&x0)),
    )
    .max_iter(100)
    .run();

    for w in result.costs.windows(2) {
        assert!(w[0] <= w[1], "simplex not sorted: {:?}", result.costs);
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
    let n = 5;
    let problem = Sphere;
    let x0 = vec![1.0; n];

    let (result, _reason) = Executor::new(
        problem,
        NelderMead::adaptive(n),
        SimplexState::new(fminsearch_simplex(&x0)),
    )
    .max_iter(2_000)
    .run();

    assert!(result.cost() < 1e-8, "cost = {}", result.cost());
}
