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
    pub(crate) param: P,
    pub(crate) cost: Option<f64>,
    pub(crate) gradient: Option<P>,
    pub(crate) iter: u64,
}

impl<P> BasicState<P> {
    pub fn new(param: P) -> Self {
        Self {
            param,
            cost: None,
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

    /// Reads the cost cached at the current `param`. Panics if accessed
    /// before `Solver::init` has run — by contract, `Executor::run` calls
    /// `init` before any criterion check, so this is safe in practice.
    fn cost(&self) -> f64 {
        self.cost
            .expect("BasicState::cost read before Solver::init populated it")
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
    pub(crate) vertices: Vec<V>,
    pub(crate) costs: Vec<f64>,
    pub(crate) iter: u64,
}

impl<V> SimplexState<V> {
    /// Read-only access to the simplex vertices, sorted by ascending cost
    /// at the start and end of every iteration.
    pub fn vertices(&self) -> &[V] {
        &self.vertices
    }

    /// Read-only access to the per-vertex cached costs, parallel to
    /// `vertices()`.
    pub fn costs(&self) -> &[f64] {
        &self.costs
    }
}

impl<V> SimplexState<V> {
    /// Build from a pre-constructed simplex (advanced users / non-default
    /// initial geometries). For the common case of "I just have a starting
    /// point", prefer the backend-specific `SimplexState::new` constructors.
    pub fn from_simplex(vertices: Vec<V>) -> Self {
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

/// FMINSEARCH/SciPy-style initial simplex from a single starting point.
///
/// Implemented per backend (`Vec<f64>`, `nalgebra::DVector<f64>`, …) so a
/// single `SimplexState::new(x0)` constructor works uniformly across
/// backends. The default step is 5% on non-zero coordinates and an
/// absolute `0.00025` on zero coordinates.
pub trait IntoInitialSimplex<V> {
    fn into_initial_simplex(self, relative_step: f64) -> Vec<V>;
}

impl IntoInitialSimplex<Vec<f64>> for Vec<f64> {
    fn into_initial_simplex(self, relative_step: f64) -> Vec<Vec<f64>> {
        let n = self.len();
        let mut simplex = Vec::with_capacity(n + 1);
        simplex.push(self.clone());
        for i in 0..n {
            let mut v = self.clone();
            v[i] = if self[i] != 0.0 {
                (1.0 + relative_step) * self[i]
            } else {
                0.00025
            };
            simplex.push(v);
        }
        simplex
    }
}

#[cfg(feature = "nalgebra")]
impl IntoInitialSimplex<nalgebra::DVector<f64>> for nalgebra::DVector<f64> {
    fn into_initial_simplex(self, relative_step: f64) -> Vec<nalgebra::DVector<f64>> {
        let n = self.len();
        let mut simplex = Vec::with_capacity(n + 1);
        simplex.push(self.clone());
        for i in 0..n {
            let mut v = self.clone();
            v[i] = if self[i] != 0.0 {
                (1.0 + relative_step) * self[i]
            } else {
                0.00025
            };
            simplex.push(v);
        }
        simplex
    }
}

impl<V> SimplexState<V> {
    /// Build an FMINSEARCH/SciPy-style simplex around a starting point
    /// `x0`. Mirrors `BasicState::new` ergonomically — the solver infers
    /// dimension from the simplex during `init`.
    pub fn new<X: IntoInitialSimplex<V>>(x0: X) -> Self {
        Self::from_simplex(x0.into_initial_simplex(0.05))
    }

    /// Like `new`, but with a custom relative step (default is `0.05`).
    /// Zero coordinates still use the FMINSEARCH absolute step `0.00025`.
    pub fn with_step<X: IntoInitialSimplex<V>>(x0: X, relative_step: f64) -> Self {
        Self::from_simplex(x0.into_initial_simplex(relative_step))
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
