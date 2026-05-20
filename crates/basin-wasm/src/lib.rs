//! WebAssembly bindings for the `basin` optimization library.
//!
//! Exposes a small, JS-friendly surface for the `web/` visualizer:
//!
//! - [`ProblemKind`] / [`SolverKind`] — plain enums marshaled across the
//!   wasm boundary as JS-side enums.
//! - [`eval_grid`] — sample a problem's cost on a uniform `nx × ny` grid
//!   for heatmap rendering. Free function so the heatmap can be rendered
//!   without constructing a [`Run`].
//! - [`Run`] — opaque handle that owns a [`basin::Stepper`] for the
//!   chosen `(problem, solver)` plus an in-wasm log of per-iteration
//!   `(x, y)` and cost. Step it with [`Run::step_many`]; pull the typed
//!   arrays out with [`Run::trajectory_xy`] and [`Run::costs`].
//!
//! The visualizer monomorphizes its concerns — 2D problems, `Vec<f64>`
//! params, no nalgebra/ndarray/faer — so the inner stepper is a single
//! concrete type per solver. That keeps the wasm bundle small and avoids
//! `dyn`-incompatible plumbing on the `Solver` trait.

use basin::problems::{beale, beale_gradient, booth, booth_gradient};
use basin::problems::{goldstein_price, goldstein_price_gradient};
use basin::problems::{matyas, matyas_gradient, mccormick, mccormick_gradient};
use basin::problems::{rosenbrock, rosenbrock_gradient, sphere, sphere_gradient};
use basin::{
    Backtracking, BasicSimplexState, BasicState, Constant, CostFunction, Executor, Gradient,
    GradientDescent, NelderMead, State, StepOutcome, Stepper, TerminationReason,
};
use serde::Serialize;
use wasm_bindgen::prelude::*;

/// Set up nicer panic messages in dev. Called automatically the first
/// time `Run::new` runs; idempotent.
fn install_panic_hook() {
    #[cfg(feature = "console_error_panic_hook")]
    {
        use std::sync::Once;
        static INIT: Once = Once::new();
        INIT.call_once(console_error_panic_hook::set_once);
    }
}

#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProblemKind {
    Sphere = 0,
    Rosenbrock = 1,
    Beale = 2,
    Booth = 3,
    Matyas = 4,
    McCormick = 5,
    GoldsteinPrice = 6,
}

#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolverKind {
    GradientDescentConstant = 0,
    GradientDescentBacktracking = 1,
    NelderMead = 2,
}

/// 2D problem dispatcher. Implements `CostFunction` + `Gradient` once
/// for `Vec<f64>`, delegating to basin's raw functions. Lets the inner
/// stepper be a single concrete type per solver instead of a forest of
/// monomorphizations.
#[derive(Clone, Copy)]
struct Problem2D(ProblemKind);

impl CostFunction for Problem2D {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, x: &Vec<f64>) -> f64 {
        match self.0 {
            ProblemKind::Sphere => sphere(x),
            ProblemKind::Rosenbrock => rosenbrock(x),
            ProblemKind::Beale => beale(x),
            ProblemKind::Booth => booth(x),
            ProblemKind::Matyas => matyas(x),
            ProblemKind::McCormick => mccormick(x),
            ProblemKind::GoldsteinPrice => goldstein_price(x),
        }
    }
}

impl Gradient for Problem2D {
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;

    fn gradient(&self, x: &Vec<f64>) -> Vec<f64> {
        let mut out = vec![0.0; x.len()];
        match self.0 {
            ProblemKind::Sphere => sphere_gradient(x, &mut out),
            ProblemKind::Rosenbrock => rosenbrock_gradient(x, &mut out),
            ProblemKind::Beale => beale_gradient(x, &mut out),
            ProblemKind::Booth => booth_gradient(x, &mut out),
            ProblemKind::Matyas => matyas_gradient(x, &mut out),
            ProblemKind::McCormick => mccormick_gradient(x, &mut out),
            ProblemKind::GoldsteinPrice => goldstein_price_gradient(x, &mut out),
        }
        out
    }
}

/// Sample `f(x, y)` on a uniform `nx × ny` grid spanning the rectangle
/// `[xmin, xmax] × [ymin, ymax]`.
///
/// Returns a flat row-major `Float64Array` of length `nx * ny` where
/// `row j` (y-coordinate index) has the `nx` x-samples laid out in
/// increasing x order. `j = 0` is `ymin`, `j = ny - 1` is `ymax`.
///
/// Cheap by design — JS calls this once per problem (or on resize) and
/// renders into a canvas. Intentionally returns a flat array, not a
/// `Vec<Vec<f64>>`, to avoid per-row JS object overhead.
#[wasm_bindgen(js_name = evalGrid)]
pub fn eval_grid(
    problem: ProblemKind,
    xmin: f64,
    xmax: f64,
    ymin: f64,
    ymax: f64,
    nx: u32,
    ny: u32,
) -> Vec<f64> {
    let p = Problem2D(problem);
    let nx = nx as usize;
    let ny = ny as usize;
    let mut out = vec![0.0; nx * ny];
    let dx = if nx > 1 {
        (xmax - xmin) / (nx as f64 - 1.0)
    } else {
        0.0
    };
    let dy = if ny > 1 {
        (ymax - ymin) / (ny as f64 - 1.0)
    } else {
        0.0
    };
    let mut xy = vec![0.0; 2];
    for j in 0..ny {
        xy[1] = ymin + dy * j as f64;
        let row = j * nx;
        for i in 0..nx {
            xy[0] = xmin + dx * i as f64;
            out[row + i] = p.cost(&xy);
        }
    }
    out
}

/// Inner enum dispatching by `(state shape, solver type)`. Each variant
/// is fully concrete so the resulting wasm is tight and no `dyn Solver`
/// gymnastics are needed.
enum Inner {
    GdConstant(Stepper<Problem2D, BasicState<Vec<f64>>, GradientDescent<Constant>>),
    GdBacktracking(Stepper<Problem2D, BasicState<Vec<f64>>, GradientDescent<Backtracking>>),
    NelderMead(Stepper<Problem2D, BasicSimplexState<Vec<f64>>, NelderMead>),
}

impl Inner {
    fn step(&mut self) -> StepOutcome {
        match self {
            Inner::GdConstant(s) => s.step(),
            Inner::GdBacktracking(s) => s.step(),
            Inner::NelderMead(s) => s.step(),
        }
    }

    fn xy(&self) -> (f64, f64) {
        let p: &Vec<f64> = match self {
            Inner::GdConstant(s) => s.state().param(),
            Inner::GdBacktracking(s) => s.state().param(),
            Inner::NelderMead(s) => s.state().param(),
        };
        (p[0], p[1])
    }

    fn cost(&self) -> f64 {
        match self {
            Inner::GdConstant(s) => s.state().cost(),
            Inner::GdBacktracking(s) => s.state().cost(),
            Inner::NelderMead(s) => s.state().cost(),
        }
    }
}

#[wasm_bindgen]
pub struct Run {
    inner: Inner,
    /// Flat (x, y) pairs, one per recorded iterate. Initial point is
    /// included at index 0 so JS doesn't need to track it separately.
    trajectory: Vec<f64>,
    costs: Vec<f64>,
    finished: Option<TerminationReason>,
}

/// Per-call result returned by `step_many`. Plain serializable shape so
/// JS receives `{ done, iters_added, reason? }` without manual JsValue
/// plumbing.
#[derive(Serialize)]
struct StepResult {
    /// True iff the run is finished (the stepper hit a termination
    /// criterion, including `MaxIter`). Once true, further `step_many`
    /// calls are no-ops.
    done: bool,
    /// Iterations actually completed by this call. May be less than the
    /// requested `n` if the run finished early or was already done.
    iters_added: u32,
    /// Termination reason as a stable string (see `reason_str`). `None`
    /// while still running.
    reason: Option<&'static str>,
}

#[wasm_bindgen]
impl Run {
    /// Construct a new run for the given `(problem, solver)` starting
    /// at `(x0, y0)`. `gd_alpha` is the constant step size for
    /// `GradientDescentConstant` (ignored for the other solvers; pass
    /// any value, e.g. `0.0`). `max_iter` caps the total number of
    /// iterations; subsequent `step_many` calls cumulatively count
    /// against this cap.
    #[wasm_bindgen(constructor)]
    pub fn new(
        problem: ProblemKind,
        solver: SolverKind,
        x0: f64,
        y0: f64,
        gd_alpha: f64,
        max_iter: u32,
    ) -> Run {
        install_panic_hook();
        let p = Problem2D(problem);
        let initial = vec![x0, y0];
        let initial_cost = p.cost(&initial);
        let inner = match solver {
            SolverKind::GradientDescentConstant => Inner::GdConstant(make_stepper(
                p,
                GradientDescent::new(gd_alpha),
                &initial,
                max_iter,
            )),
            SolverKind::GradientDescentBacktracking => Inner::GdBacktracking(make_stepper(
                p,
                GradientDescent::with_line_search(Backtracking::new()),
                &initial,
                max_iter,
            )),
            SolverKind::NelderMead => {
                let stepper = Executor::new(
                    p,
                    NelderMead::standard(),
                    BasicSimplexState::<Vec<f64>>::new(initial.clone()),
                )
                .max_iter(max_iter as u64)
                .into_stepper();
                Inner::NelderMead(stepper)
            }
        };
        Run {
            inner,
            trajectory: vec![x0, y0],
            costs: vec![initial_cost],
            finished: None,
        }
    }

    /// Advance up to `n` iterations, recording the `(x, y)` and cost
    /// after each. Returns `{ done, iters_added, reason? }` so JS can
    /// append only the new tail of the trajectory and stop the
    /// requestAnimationFrame loop when finished.
    #[wasm_bindgen(js_name = stepMany)]
    pub fn step_many(&mut self, n: u32) -> JsValue {
        let result = self.step_many_inner(n);
        serde_wasm_bindgen::to_value(&result).unwrap_or(JsValue::NULL)
    }

    /// Full trajectory as a flat `Float64Array` of `(x, y)` pairs.
    /// Length is `2 * (iter + 1)` (initial point + one per completed
    /// iteration).
    #[wasm_bindgen(js_name = trajectoryXy)]
    pub fn trajectory_xy(&self) -> Vec<f64> {
        self.trajectory.clone()
    }

    /// Per-iterate cost values, including the cost at the initial point
    /// (so `costs.length === trajectory.length / 2`).
    pub fn costs(&self) -> Vec<f64> {
        self.costs.clone()
    }

    /// Iteration counter (excludes the initial point).
    pub fn iter(&self) -> u32 {
        self.costs.len().saturating_sub(1) as u32
    }

    /// True iff the stepper has stopped.
    pub fn done(&self) -> bool {
        self.finished.is_some()
    }

    /// Termination reason string, or empty if still running.
    pub fn reason(&self) -> String {
        self.finished.map(reason_str).unwrap_or("").to_string()
    }
}

impl Run {
    /// Pure-Rust core of `step_many`, callable from native unit tests
    /// without going through `serde_wasm_bindgen` (which calls into JS
    /// APIs that panic on non-wasm targets).
    fn step_many_inner(&mut self, n: u32) -> StepResult {
        if self.finished.is_some() {
            return StepResult {
                done: true,
                iters_added: 0,
                reason: self.finished.map(reason_str),
            };
        }
        let mut iters_added = 0;
        for _ in 0..n {
            match self.inner.step() {
                StepOutcome::Continue => {
                    let (x, y) = self.inner.xy();
                    self.trajectory.push(x);
                    self.trajectory.push(y);
                    self.costs.push(self.inner.cost());
                    iters_added += 1;
                }
                StepOutcome::Stopped(reason) => {
                    self.finished = Some(reason);
                    break;
                }
            }
        }
        StepResult {
            done: self.finished.is_some(),
            iters_added,
            reason: self.finished.map(reason_str),
        }
    }
}

fn make_stepper<L>(
    problem: Problem2D,
    solver: GradientDescent<L>,
    initial: &[f64],
    max_iter: u32,
) -> Stepper<Problem2D, BasicState<Vec<f64>>, GradientDescent<L>>
where
    GradientDescent<L>: basin::Solver<Problem2D, BasicState<Vec<f64>>>,
{
    Executor::new(problem, solver, BasicState::new(initial.to_vec()))
        .max_iter(max_iter as u64)
        .into_stepper()
}

/// Stable, JS-friendly string for a `TerminationReason`. The wasm
/// boundary discards Rust enum nuance, so we serialize one short tag
/// per variant; the UI can branch on it.
fn reason_str(r: TerminationReason) -> &'static str {
    match r {
        TerminationReason::MaxIter => "max_iter",
        TerminationReason::MaxCostEvals => "max_cost_evals",
        TerminationReason::MaxGradientEvals => "max_gradient_evals",
        TerminationReason::GradientTolerance => "gradient_tolerance",
        TerminationReason::ProjectedGradientTolerance => "projected_gradient_tolerance",
        TerminationReason::ParamTolerance => "param_tolerance",
        TerminationReason::RelativeParamTolerance => "relative_param_tolerance",
        TerminationReason::CostTolerance => "cost_tolerance",
        TerminationReason::RelativeCostTolerance => "relative_cost_tolerance",
        TerminationReason::SimplexTolerance => "simplex_tolerance",
        TerminationReason::MaxTime => "max_time",
        TerminationReason::SolverConverged => "solver_converged",
        TerminationReason::SolverFailed => "solver_failed",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eval_grid_returns_expected_shape_and_values() {
        let g = eval_grid(ProblemKind::Sphere, -1.0, 1.0, -1.0, 1.0, 3, 3);
        assert_eq!(g.len(), 9);
        // Center sample is f(0, 0) = 0 for sphere.
        assert!(g[4].abs() < 1e-12);
        // Corners are f(±1, ±1) = 2.
        assert!((g[0] - 2.0).abs() < 1e-12);
        assert!((g[8] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn run_records_initial_point_and_progresses() {
        let mut run = Run::new(
            ProblemKind::Rosenbrock,
            SolverKind::GradientDescentConstant,
            -1.2,
            1.0,
            0.001,
            500,
        );
        assert_eq!(run.iter(), 0);
        assert_eq!(run.trajectory_xy(), vec![-1.2, 1.0]);
        let r = run.step_many_inner(50);
        assert_eq!(r.iters_added, 50);
        assert!(!r.done);
        assert_eq!(run.iter(), 50);
        assert_eq!(run.trajectory_xy().len(), 2 * 51);
        assert_eq!(run.costs().len(), 51);
    }

    #[test]
    fn run_terminates_on_max_iter() {
        let mut run = Run::new(
            ProblemKind::Sphere,
            SolverKind::GradientDescentConstant,
            1.0,
            1.0,
            0.5,
            5,
        );
        let r = run.step_many_inner(100);
        assert!(r.done);
        assert_eq!(r.reason, Some("max_iter"));
        assert!(run.done());
        assert_eq!(run.reason(), "max_iter");
        assert!(run.iter() <= 5);
    }
}
