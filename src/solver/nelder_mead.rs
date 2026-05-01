use crate::core::math::ScaledAdd;
use crate::core::problem::CostFunction;
use crate::core::solver::Solver;
use crate::core::state::SimplexState;

/// Nelder-Mead simplex method (derivative-free).
///
/// Implements the algorithm as stated in Lagarias et al. (1998) with the
/// adaptive parameter option of Gao & Han (2012). The four parameters are:
/// `α` (reflection), `β` (expansion), `γ` (contraction), `δ` (shrink), with
/// the constraints `α > 0`, `β > 1`, `0 < γ < 1`, `0 < δ < 1`.
pub struct NelderMead {
    alpha: f64,
    beta: f64,
    gamma: f64,
    delta: f64,
}

impl NelderMead {
    /// Standard parameters (Nelder & Mead 1965): α=1, β=2, γ=0.5, δ=0.5.
    pub fn standard() -> Self {
        Self {
            alpha: 1.0,
            beta: 2.0,
            gamma: 0.5,
            delta: 0.5,
        }
    }

    /// Adaptive parameters from Gao & Han (2012), eq. (4.1):
    /// α=1, β=1+2/n, γ=0.75−1/(2n), δ=1−1/n. Coincides with `standard()`
    /// when `n == 2`.
    pub fn adaptive(n: usize) -> Self {
        assert!(n >= 2, "NelderMead::adaptive requires n >= 2");
        let n = n as f64;
        Self {
            alpha: 1.0,
            beta: 1.0 + 2.0 / n,
            gamma: 0.75 - 1.0 / (2.0 * n),
            delta: 1.0 - 1.0 / n,
        }
    }

    pub fn with_params(alpha: f64, beta: f64, gamma: f64, delta: f64) -> Self {
        assert!(alpha > 0.0, "α must be > 0");
        assert!(beta > 1.0, "β must be > 1");
        assert!(gamma > 0.0 && gamma < 1.0, "γ must be in (0, 1)");
        assert!(delta > 0.0 && delta < 1.0, "δ must be in (0, 1)");
        Self {
            alpha,
            beta,
            gamma,
            delta,
        }
    }
}

/// Build `(1 - t) * a + t * b` from two vectors and a scalar interpolant.
/// Works for any `t ∈ ℝ` — values outside `[0, 1]` extrapolate, which is
/// what reflection needs.
fn affine<V: Clone + ScaledAdd<f64>>(a: &V, b: &V, t: f64) -> V {
    let mut out = a.clone();
    out.scaled_add(-t, a);
    out.scaled_add(t, b);
    out
}

/// Centroid of `vertices` (mean of all entries).
fn centroid<V: Clone + ScaledAdd<f64>>(vertices: &[V]) -> V {
    let inv = 1.0 / vertices.len() as f64;
    let mut c = vertices[0].clone();
    c.scaled_add(inv - 1.0, &vertices[0]);
    for v in &vertices[1..] {
        c.scaled_add(inv, v);
    }
    c
}

/// Sort `vertices` and `costs` jointly by ascending cost. NaN costs sort
/// last so a single bad evaluation can't drag itself to the front.
fn sort_simplex<V>(vertices: &mut [V], costs: &mut [f64]) {
    let n = vertices.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| {
        costs[i]
            .partial_cmp(&costs[j])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    apply_permutation(vertices, &idx);
    apply_permutation(costs, &idx);
}

fn apply_permutation<T>(slice: &mut [T], idx: &[usize]) {
    let mut visited = vec![false; slice.len()];
    for start in 0..slice.len() {
        if visited[start] || idx[start] == start {
            visited[start] = true;
            continue;
        }
        let mut current = start;
        loop {
            let next = idx[current];
            visited[current] = true;
            if next == start {
                break;
            }
            slice.swap(current, next);
            current = next;
        }
    }
}

impl<P, V> Solver<P, SimplexState<V>> for NelderMead
where
    P: CostFunction<Param = V, Output = f64>,
    V: Clone + ScaledAdd<f64>,
{
    fn init(&mut self, problem: &P, mut state: SimplexState<V>) -> SimplexState<V> {
        for (v, c) in state.vertices.iter().zip(state.costs.iter_mut()) {
            *c = problem.cost(v);
        }
        sort_simplex(&mut state.vertices, &mut state.costs);
        state
    }

    fn next_iter(&mut self, problem: &P, mut state: SimplexState<V>) -> SimplexState<V> {
        // Vertices are sorted (best at index 0) on entry; we restore that
        // invariant before returning. The simplex has n+1 vertices in n-D.
        let m = state.vertices.len();
        let n = m - 1;
        let worst = m - 1;

        let x_bar = centroid(&state.vertices[..n]);

        let f1 = state.costs[0];
        let fn_ = state.costs[n - 1];
        let fnp1 = state.costs[worst];

        // Reflection: x_r = x_bar + α(x_bar − x_{n+1}) = (1+α)·x_bar − α·x_{n+1}
        let x_r = affine(&x_bar, &state.vertices[worst], -self.alpha);
        let fr = problem.cost(&x_r);

        if f1 <= fr && fr < fn_ {
            // Accept reflection.
            state.vertices[worst] = x_r;
            state.costs[worst] = fr;
        } else if fr < f1 {
            // Try expansion: x_e = x_bar + β(x_r − x_bar).
            let x_e = affine(&x_bar, &x_r, self.beta);
            let fe = problem.cost(&x_e);
            if fe < fr {
                state.vertices[worst] = x_e;
                state.costs[worst] = fe;
            } else {
                state.vertices[worst] = x_r;
                state.costs[worst] = fr;
            }
        } else if fr < fnp1 {
            // fn ≤ fr < f_{n+1}: outside contraction.
            // x_oc = x_bar + γ(x_r − x_bar).
            let x_oc = affine(&x_bar, &x_r, self.gamma);
            let foc = problem.cost(&x_oc);
            if foc <= fr {
                state.vertices[worst] = x_oc;
                state.costs[worst] = foc;
            } else {
                self.shrink(problem, &mut state);
            }
        } else {
            // fr ≥ f_{n+1}: inside contraction.
            // x_ic = x_bar − γ(x_bar − x_{n+1}) = (1−γ)·x_bar + γ·x_{n+1}.
            let x_ic = affine(&x_bar, &state.vertices[worst], self.gamma);
            let fic = problem.cost(&x_ic);
            if fic < fnp1 {
                state.vertices[worst] = x_ic;
                state.costs[worst] = fic;
            } else {
                self.shrink(problem, &mut state);
            }
        }

        sort_simplex(&mut state.vertices, &mut state.costs);
        state
    }
}

impl NelderMead {
    fn shrink<P, V>(&self, problem: &P, state: &mut SimplexState<V>)
    where
        P: CostFunction<Param = V, Output = f64>,
        V: Clone + ScaledAdd<f64>,
    {
        // Best vertex is fixed at index 0; shrink every other vertex toward it.
        // Split-borrow lets us read x[0] while mutating x[i].
        let (best_slice, rest) = state.vertices.split_at_mut(1);
        let best = &best_slice[0];
        for (v, c) in rest.iter_mut().zip(&mut state.costs[1..]) {
            *v = affine(best, v, self.delta);
            *c = problem.cost(v);
        }
    }
}
