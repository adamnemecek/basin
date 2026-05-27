//! Backend axis (axis 1 of the bench plan): fix the solver + problem, vary
//! only the linear-algebra backend, isolating the cost of the LA layer.
//!
//! Runs the three solvers that work on *all four* backends — gradient
//! descent (More-Thuente line search), Nelder-Mead (standard simplex), and
//! unconstrained L-BFGS — on Rosenbrock at `n ∈ {2, 10, 100}`, on each of
//! `Vec<f64>`, nalgebra (`DVector<f64>`), ndarray (`Array1<f64>`), and faer
//! (`Col<f64>`). LM / Gauss-Newton (nalgebra + faer only) and BFGS / CMA-ES
//! (no ndarray) are out of scope here; the LM backend pair lives in
//! `lm_backends.rs`.
//!
//! A *fixed* iteration budget with no tolerance criterion (`MAX_ITERS`) means
//! every backend does the identical algorithmic work, so the ratio across a
//! group is pure per-iteration backend cost. Problem + start-state
//! construction is charged to `iter_batched` setup, not the timed routine.
//!
//! Run with
//! `cargo bench --features nalgebra,ndarray,faer --bench solver_backends`.

use std::hint::black_box;

use basin::problems::Rosenbrock;
use basin::{
    BasicSimplexState, BasicState, Executor, GradientDescent, LbfgsState, MoreThuente, NelderMead,
    LBFGSB,
};
use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};

use faer::Col;
use nalgebra::DVector;
use ndarray::Array1;

/// Fixed budget, no early stop — identical work across backends.
const MAX_ITERS: u64 = 200;

/// Rosenbrock dimensions. Larger `n` surfaces the backends' vectorization
/// differences; `n = 2` is the classic 2-D valley.
const DIMS: [usize; 3] = [2, 10, 100];

/// Classic Rosenbrock start, extended to `n` dims by repeating `(−1.2, 1.0)`
/// (matches `competitor-bench`'s `gd_nm.rs`).
fn rosenbrock_start(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| if i % 2 == 0 { -1.2 } else { 1.0 })
        .collect()
}

/// Emit one criterion contestant: a full solve to `MAX_ITERS` on `$prob`
/// (the backend-specific `Rosenbrock<V>`), building the start state from
/// `$setup` and wrapping it in `$state`. `$solver` is reconstructed each
/// iteration (cheap, and identical across backends).
macro_rules! contestant {
    ($g:expr, $label:literal, $setup:expr, $prob:ty, $solver:expr, $state:expr $(,)?) => {
        $g.bench_function(BenchmarkId::from_parameter($label), |b| {
            b.iter_batched(
                $setup,
                |x0| {
                    black_box(
                        Executor::new(<$prob>::default(), $solver, $state(x0))
                            .max_iter(MAX_ITERS)
                            .run(),
                    )
                },
                BatchSize::SmallInput,
            )
        });
    };
}

/// Run `contestant!` once per backend for a given solver + state, using the
/// canonical per-backend start-vector constructors.
macro_rules! all_backends {
    ($g:expr, $start:expr, $n:expr, $solver:expr, $state:expr $(,)?) => {{
        let start = $start;
        contestant!(
            $g,
            "vec",
            || start.clone(),
            Rosenbrock<Vec<f64>>,
            $solver,
            $state
        );
        contestant!(
            $g,
            "nalgebra",
            || DVector::from_vec(start.clone()),
            Rosenbrock<DVector<f64>>,
            $solver,
            $state,
        );
        contestant!(
            $g,
            "ndarray",
            || Array1::from_vec(start.clone()),
            Rosenbrock<Array1<f64>>,
            $solver,
            $state,
        );
        contestant!(
            $g,
            "faer",
            || Col::<f64>::from_fn($n, |i| start[i]),
            Rosenbrock<Col<f64>>,
            $solver,
            $state,
        );
    }};
}

fn bench_gd(c: &mut Criterion) {
    for n in DIMS {
        let mut g = c.benchmark_group(format!("gd_rosenbrock_n{n}"));
        all_backends!(
            g,
            rosenbrock_start(n),
            n,
            GradientDescent::with_line_search(MoreThuente::new()),
            BasicState::new,
        );
        g.finish();
    }
}

fn bench_nm(c: &mut Criterion) {
    for n in DIMS {
        let mut g = c.benchmark_group(format!("nm_rosenbrock_n{n}"));
        all_backends!(
            g,
            rosenbrock_start(n),
            n,
            NelderMead::standard(),
            BasicSimplexState::new,
        );
        g.finish();
    }
}

fn bench_lbfgs(c: &mut Criterion) {
    for n in DIMS {
        let mut g = c.benchmark_group(format!("lbfgs_rosenbrock_n{n}"));
        all_backends!(g, rosenbrock_start(n), n, LBFGSB::new().unbounded(), |x0| {
            LbfgsState::new(x0, 10)
        },);
        g.finish();
    }
}

criterion_group!(benches, bench_gd, bench_nm, bench_lbfgs);
criterion_main!(benches);
