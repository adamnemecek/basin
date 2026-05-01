pub trait State {
    type Param;
    type Float;

    fn iter(&self) -> u64;
    fn increment_iter(&mut self);
    fn param(&self) -> &Self::Param;
    fn cost(&self) -> Self::Float;
}

/// States that carry a gradient at the current `param`.
///
/// Solvers that compute a gradient should populate `gradient()` so that
/// gradient-based termination criteria can read it without re-evaluating
/// the problem. `None` means "no gradient available at this iterate yet"
/// (e.g. before `Solver::init` has run).
pub trait GradientState: State {
    fn gradient(&self) -> Option<&Self::Param>;
}

pub struct BasicState<P> {
    pub param: P,
    pub cost: f64,
    pub gradient: Option<P>,
    pub iter: u64,
}

impl<P> BasicState<P> {
    pub fn new(param: P) -> Self {
        Self {
            param,
            cost: f64::INFINITY,
            gradient: None,
            iter: 0,
        }
    }
}

impl<P> State for BasicState<P> {
    type Param = P;
    type Float = f64;

    fn iter(&self) -> u64 {
        self.iter
    }

    fn increment_iter(&mut self) {
        self.iter += 1;
    }

    fn param(&self) -> &P {
        &self.param
    }

    fn cost(&self) -> f64 {
        self.cost
    }
}

impl<P> GradientState for BasicState<P> {
    fn gradient(&self) -> Option<&P> {
        self.gradient.as_ref()
    }
}

/// State for simplex-based solvers (Nelder-Mead, etc.).
///
/// Holds `n + 1` vertices and their costs in parallel vectors. The solver is
/// expected to keep them sorted by ascending cost at the start and end of
/// every `next_iter`, so `param()` and `cost()` always return the current
/// best vertex.
pub struct SimplexState<V> {
    pub vertices: Vec<V>,
    pub costs: Vec<f64>,
    pub iter: u64,
}

impl<V> SimplexState<V> {
    pub fn new(vertices: Vec<V>) -> Self {
        assert!(
            vertices.len() >= 2,
            "SimplexState requires at least 2 vertices (n+1 for an n-D problem)"
        );
        let n = vertices.len();
        Self {
            vertices,
            costs: vec![f64::INFINITY; n],
            iter: 0,
        }
    }
}

impl<V> State for SimplexState<V> {
    type Param = V;
    type Float = f64;

    fn iter(&self) -> u64 {
        self.iter
    }

    fn increment_iter(&mut self) {
        self.iter += 1;
    }

    fn param(&self) -> &V {
        &self.vertices[0]
    }

    fn cost(&self) -> f64 {
        self.costs[0]
    }
}
