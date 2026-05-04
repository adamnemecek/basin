pub trait State {
    type Param;
    type Float;

    fn iter(&self) -> u64;
    fn increment_iter(&mut self);
    /// Cumulative count of cost-function evaluations performed so far.
    /// Diverges from `iter()` whenever a single iteration evaluates the
    /// cost more than once (line searches, Nelder-Mead shrinks, etc.) —
    /// this is what users actually budget against.
    fn cost_evals(&self) -> u64;
    fn increment_cost_evals(&mut self, by: u64);
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
    /// Cumulative count of gradient evaluations performed so far. Lives
    /// on `GradientState` rather than `State` so derivative-free states
    /// don't carry a counter they can never increment.
    fn gradient_evals(&self) -> u64;
    fn increment_gradient_evals(&mut self, by: u64);
}

/// States built around a simplex of `n + 1` vertices and parallel costs.
///
/// Mirrors `GradientState`: the trait exists so termination criteria
/// (e.g. simplex-collapse tests à la Lagarias et al. 1998 (T1)) can bound
/// on a richer view than `State::param()` / `cost()`, which only see the
/// best vertex. The vertex/cost arrays are sorted by ascending cost at the
/// start and end of every `Solver::next_iter`.
pub trait SimplexState: State {
    fn vertices(&self) -> &[Self::Param];
    fn costs(&self) -> &[Self::Float];
}

pub struct BasicState<P> {
    pub(crate) param: P,
    pub(crate) cost: Option<f64>,
    pub(crate) gradient: Option<P>,
    pub(crate) iter: u64,
    pub(crate) cost_evals: u64,
    pub(crate) gradient_evals: u64,
}

impl<P> BasicState<P> {
    pub fn new(param: P) -> Self {
        Self {
            param,
            cost: None,
            gradient: None,
            iter: 0,
            cost_evals: 0,
            gradient_evals: 0,
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

    fn cost_evals(&self) -> u64 {
        self.cost_evals
    }

    fn increment_cost_evals(&mut self, by: u64) {
        self.cost_evals += by;
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

    fn gradient_evals(&self) -> u64 {
        self.gradient_evals
    }

    fn increment_gradient_evals(&mut self, by: u64) {
        self.gradient_evals += by;
    }
}

/// Default `SimplexState` implementation: `n + 1` vertices and their costs
/// in parallel `Vec`s. The solver keeps both sorted by ascending cost at
/// the start and end of every `next_iter`, so `param()` / `cost()` always
/// return the current best vertex.
pub struct BasicSimplexState<V> {
    pub(crate) vertices: Vec<V>,
    pub(crate) costs: Vec<f64>,
    pub(crate) iter: u64,
    pub(crate) cost_evals: u64,
}

impl<V> BasicSimplexState<V> {
    /// Build from a pre-constructed simplex (advanced users / non-default
    /// initial geometries). For the common case of "I just have a starting
    /// point", prefer the backend-specific `BasicSimplexState::new`
    /// constructors.
    pub fn from_simplex(vertices: Vec<V>) -> Self {
        assert!(
            vertices.len() >= 2,
            "BasicSimplexState requires at least 2 vertices (n+1 for an n-D problem)"
        );
        let n = vertices.len();
        Self {
            vertices,
            costs: vec![f64::INFINITY; n],
            iter: 0,
            cost_evals: 0,
        }
    }
}

/// FMINSEARCH/SciPy-style initial simplex from a single starting point.
///
/// Implemented per backend (`Vec<f64>`, `nalgebra::DVector<f64>`, …) so a
/// single `BasicSimplexState::new(x0)` constructor works uniformly across
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

#[cfg(feature = "faer")]
impl IntoInitialSimplex<faer::Col<f64>> for faer::Col<f64> {
    fn into_initial_simplex(self, relative_step: f64) -> Vec<faer::Col<f64>> {
        let n = self.nrows();
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

#[cfg(feature = "ndarray")]
impl IntoInitialSimplex<ndarray::Array1<f64>> for ndarray::Array1<f64> {
    fn into_initial_simplex(self, relative_step: f64) -> Vec<ndarray::Array1<f64>> {
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

impl<V> BasicSimplexState<V> {
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

impl<V> State for BasicSimplexState<V> {
    type Param = V;
    type Float = f64;

    fn iter(&self) -> u64 {
        self.iter
    }

    fn increment_iter(&mut self) {
        self.iter += 1;
    }

    fn cost_evals(&self) -> u64 {
        self.cost_evals
    }

    fn increment_cost_evals(&mut self, by: u64) {
        self.cost_evals += by;
    }

    fn param(&self) -> &V {
        &self.vertices[0]
    }

    fn cost(&self) -> f64 {
        self.costs[0]
    }
}

impl<V> SimplexState for BasicSimplexState<V> {
    fn vertices(&self) -> &[V] {
        &self.vertices
    }

    fn costs(&self) -> &[f64] {
        &self.costs
    }
}

/// State for quasi-Newton solvers that maintain a dense inverse-Hessian
/// approximation `H ≈ ∇²f(x)⁻¹` (BFGS, DFP, SR1).
///
/// Hardcoded to nalgebra dynamic vectors/matrices for v1 (BFGS only
/// supports the nalgebra backend — see `solver::bfgs` doc). L-BFGS will
/// need a different state shape (history of `(s, y)` pairs) when it lands.
///
/// `initial_scaling_done` tracks whether we've applied the standard
/// `H₀ ← (sᵀy / yᵀy)·I` rescaling after the first accepted step (Nocedal
/// & Wright (6.20)). This makes the unit step well-scaled on poorly
/// conditioned problems where plain identity initialization stalls.
#[cfg(feature = "nalgebra")]
pub struct QuasiNewtonState<V, M> {
    pub(crate) param: V,
    pub(crate) cost: Option<f64>,
    pub(crate) gradient: Option<V>,
    pub(crate) inverse_hessian: M,
    pub(crate) initial_scaling_done: bool,
    pub(crate) iter: u64,
    pub(crate) cost_evals: u64,
    pub(crate) gradient_evals: u64,
}

#[cfg(feature = "nalgebra")]
impl QuasiNewtonState<nalgebra::DVector<f64>, nalgebra::DMatrix<f64>> {
    pub fn new(param: nalgebra::DVector<f64>) -> Self {
        let n = param.len();
        Self {
            param,
            cost: None,
            gradient: None,
            inverse_hessian: nalgebra::DMatrix::identity(n, n),
            initial_scaling_done: false,
            iter: 0,
            cost_evals: 0,
            gradient_evals: 0,
        }
    }
}

#[cfg(feature = "nalgebra")]
impl<V, M> State for QuasiNewtonState<V, M> {
    type Param = V;
    type Float = f64;

    fn iter(&self) -> u64 {
        self.iter
    }

    fn increment_iter(&mut self) {
        self.iter += 1;
    }

    fn cost_evals(&self) -> u64 {
        self.cost_evals
    }

    fn increment_cost_evals(&mut self, by: u64) {
        self.cost_evals += by;
    }

    fn param(&self) -> &V {
        &self.param
    }

    fn cost(&self) -> f64 {
        self.cost
            .expect("QuasiNewtonState::cost read before Solver::init populated it")
    }
}

#[cfg(feature = "nalgebra")]
impl<V, M> GradientState for QuasiNewtonState<V, M> {
    fn gradient(&self) -> Option<&V> {
        self.gradient.as_ref()
    }

    fn gradient_evals(&self) -> u64 {
        self.gradient_evals
    }

    fn increment_gradient_evals(&mut self, by: u64) {
        self.gradient_evals += by;
    }
}
