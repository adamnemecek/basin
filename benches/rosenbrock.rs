use basin::{BasicState, CostFunction, Executor, Gradient, GradientDescent, Solver};
use std::hint::black_box;
use std::time::{Duration, Instant};

struct Rosenbrock;

impl CostFunction for Rosenbrock {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, x: &Vec<f64>) -> f64 {
        (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2)
    }
}

impl Gradient for Rosenbrock {
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;

    fn gradient(&self, x: &Vec<f64>) -> Vec<f64> {
        vec![
            -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0].powi(2)),
            200.0 * (x[1] - x[0].powi(2)),
        ]
    }
}

fn bench<S, R>(name: &str, iters: u32, mut setup: impl FnMut() -> S, mut run: impl FnMut(S) -> R) {
    for _ in 0..3 {
        let _ = black_box(run(setup()));
    }
    let mut total = Duration::ZERO;
    for _ in 0..iters {
        let s = setup();
        let t0 = Instant::now();
        let _ = black_box(run(s));
        total += t0.elapsed();
    }
    let per = total / iters;
    println!("{name:40}  {per:>10.3?} / iter  ({iters} iters)");
}

fn main() {
    bench(
        "gradient_descent_step",
        100_000,
        || {
            let mut solver = GradientDescent::new(0.001);
            // `Solver::init` populates cost + gradient at the initial param,
            // matching the contract `next_iter` expects (gradient cached
            // from the previous iter or from init).
            let state = solver.init(&Rosenbrock, BasicState::new(vec![-1.2, 1.0]));
            (solver, state)
        },
        |(mut solver, state)| solver.next_iter(&Rosenbrock, state),
    );

    bench(
        "gradient_descent_full_run_10k",
        50,
        || (),
        |_| {
            Executor::new(
                Rosenbrock,
                GradientDescent::new(0.001),
                BasicState::new(vec![-1.2, 1.0]),
            )
            .max_iter(10_000)
            .run()
        },
    );
}
