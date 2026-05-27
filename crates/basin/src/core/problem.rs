//! Problem traits the user implements about their objective. Solvers
//! bind on whichever subset they need (e.g. gradient descent requires
//! [`CostFunction`] *and* [`Gradient`]; Nelder-Mead only needs
//! [`CostFunction`]).

/// Scalar-valued objective `f(x): Param → Output`. The smallest
/// problem trait — every solver binds at least on this.
///
/// # Contract
///
/// - **Implementor must:** be a *pure* function of `param` —
///   evaluating at the same `param` twice must return the same
///   `Output`. Solvers cache costs across iterations, line searches
///   reuse evaluations, and termination criteria assume the cost they
///   read from the state matches what a fresh `cost(param)` would
///   return.
/// - **Implementor must not:** assume any particular call order or
///   frequency. Solvers may evaluate at exploratory points outside the
///   accepted iterate sequence (line-search probes, Nelder-Mead
///   reflections / contractions / shrinks, finite-difference probes).
///
/// # Examples
///
/// ```
/// use basin::CostFunction;
///
/// struct Sphere;
/// impl CostFunction for Sphere {
///     type Param = Vec<f64>;
///     type Output = f64;
///     fn cost(&self, x: &Vec<f64>) -> f64 {
///         x.iter().map(|xi| xi * xi).sum()
///     }
/// }
///
/// assert_eq!(Sphere.cost(&vec![3.0, 4.0]), 25.0);
/// ```
pub trait CostFunction {
    /// The parameter type the objective is defined over.
    type Param;
    /// Scalar cost type. In practice `f64` (see `AGENTS.md`'s
    /// provisional choices).
    type Output;

    /// Evaluate the objective at `param`.
    fn cost(&self, param: &Self::Param) -> Self::Output;
}

/// Analytic gradient `∇f(x): Param → Gradient`. Required by
/// first-order solvers (gradient descent, BFGS, …).
///
/// # Contract
///
/// - **Implementor must:** be a *pure* function of `param`, with the
///   same call-order independence as [`CostFunction::cost`].
/// - **Implementor must:** return a `Gradient` whose shape matches
///   `param` so solver math (`x ← x − α·∇f(x)`) lines up. Most
///   problems have `Gradient = Param`, which is what the shipped
///   solvers' bounds expect (e.g. `Gradient<Param = V, Gradient = V>`).
/// - The gradient must agree with [`CostFunction::cost`]: it is the
///   actual derivative, not a finite-difference approximation unless
///   the implementor is happy taking the loss in solver
///   convergence behavior.
///
/// # Examples
///
/// ```
/// use basin::Gradient;
///
/// struct Sphere;
/// impl Gradient for Sphere {
///     type Param = Vec<f64>;
///     type Gradient = Vec<f64>;
///     fn gradient(&self, x: &Vec<f64>) -> Vec<f64> {
///         x.iter().map(|xi| 2.0 * xi).collect()
///     }
/// }
///
/// assert_eq!(Sphere.gradient(&vec![1.0, 2.0]), vec![2.0, 4.0]);
/// ```
pub trait Gradient {
    /// The parameter type the gradient is defined over (matches
    /// [`CostFunction::Param`]).
    type Param;
    /// The gradient type. Typically the same as `Param`.
    type Gradient;

    /// Evaluate the gradient at `param`.
    fn gradient(&self, param: &Self::Param) -> Self::Gradient;
}

/// Vector-valued residual `r(x): Param → Output` for least-squares
/// problems. Required by Gauss-Newton, Levenberg-Marquardt, and any
/// solver that minimizes `½‖r(x)‖²`.
///
/// # Contract
///
/// - **Implementor must:** be a *pure* function of `param`, with the
///   same call-order independence as [`CostFunction::cost`].
/// - **Implementor must:** return an `Output` whose length `m` is fixed
///   for a given problem — `m` does not depend on the iterate. Solvers
///   may allocate workspace once based on the first call. `m` is
///   independent of `param.len() = n`.
/// - When [`CostFunction`] is also implemented, the cost must agree
///   with the residual under the convention `cost(x) = ½ Σ rᵢ(x)²`,
///   unless the problem documents an unscaled `Σ rᵢ²` form (see e.g.
///   the existing Rosenbrock cost, which is the published unscaled
///   form).
///
/// # Examples
///
/// ```
/// use basin::Residual;
///
/// // r(x) = (x₀ − 1, x₁ − 2); the least-squares optimum is (1, 2).
/// struct Affine;
/// impl Residual for Affine {
///     type Param = Vec<f64>;
///     type Output = Vec<f64>;
///     fn residual(&self, x: &Vec<f64>) -> Vec<f64> {
///         vec![x[0] - 1.0, x[1] - 2.0]
///     }
/// }
///
/// assert_eq!(Affine.residual(&vec![0.0, 0.0]), vec![-1.0, -2.0]);
/// ```
pub trait Residual {
    /// The parameter type the residual is defined over (matches
    /// [`CostFunction::Param`]).
    type Param;
    /// The residual vector type. Length is the number of residuals `m`,
    /// independent of `param.len() = n`.
    type Output;

    /// Evaluate the residual at `param`.
    fn residual(&self, param: &Self::Param) -> Self::Output;
}

/// Analytic Jacobian `J(x) = ∂r/∂x: Param → Output` for least-squares
/// solvers (Gauss-Newton, LM, TRF). The associated `Output` matrix
/// type is what lets solvers bound on the linear-algebra ops they need
/// ([`MatVec`](crate::core::math::MatVec),
/// [`LinearSolveSpd`](crate::core::math::LinearSolveSpd), …) without
/// baking in a specific backend or assuming density.
///
/// # Contract
///
/// - **Implementor must:** be a *pure* function of `param`, with the
///   same call-order independence as [`CostFunction::cost`].
/// - **Implementor must:** return a matrix of shape `m × n` where
///   `m = residual(param).len()` and `n = param.len()`. The `(i, j)`
///   entry is `∂rᵢ / ∂xⱼ`. Shape is fixed across iterates.
/// - The Jacobian must agree with [`Residual::residual`]: it is the
///   actual derivative, not a finite-difference approximation, unless
///   the implementor accepts the loss in solver convergence behavior.
///
/// # Backends
///
/// Wired up for the LA-heavy backends only:
///
/// - `Param = nalgebra::DVector<f64>` → `Output = nalgebra::DMatrix<f64>`
///   (dense) or `nalgebra_sparse::CscMatrix<f64>` (sparse). Both ride
///   on the `nalgebra` feature.
/// - `Param = faer::Col<f64>` → `Output = faer::Mat<f64>` (dense) or
///   `faer::sparse::SparseColMat<usize, f64>` (sparse). Both ride on
///   the `faer` feature.
///
/// `Vec<f64>` deliberately does not implement `Jacobian` — there is no
/// honest matrix type to pair with it. `ndarray::Array1<f64>` likewise
/// has no `Jacobian` impl: `ndarray-linalg` requires system BLAS/LAPACK
/// and breaks the wasm-default tenet, so there's no honest
/// [`LinearSolveSpd`](crate::core::math::LinearSolveSpd) to back it.
/// Per tenet 5 in `AGENTS.md`, missing backend coverage is a
/// compile-time error rather than a runtime surprise.
///
/// # Examples
///
/// ```
/// # #[cfg(feature = "nalgebra")] {
/// use basin::Jacobian;
/// use nalgebra::{DMatrix, DVector};
///
/// // For r(x) = (x₀ − 1, x₁ − 2) the Jacobian is the 2×2 identity.
/// struct Affine;
/// impl Jacobian for Affine {
///     type Param = DVector<f64>;
///     type Output = DMatrix<f64>;
///     fn jacobian(&self, _x: &DVector<f64>) -> DMatrix<f64> {
///         DMatrix::identity(2, 2)
///     }
/// }
///
/// let j = Affine.jacobian(&DVector::from_vec(vec![0.0, 0.0]));
/// assert_eq!(j[(0, 0)], 1.0);
/// # }
/// ```
pub trait Jacobian {
    /// The parameter type the Jacobian is defined over (matches
    /// [`Residual::Param`]).
    type Param;
    /// The Jacobian matrix type, shape `m × n`.
    type Output;

    /// Evaluate the Jacobian at `param`.
    fn jacobian(&self, param: &Self::Param) -> Self::Output;
}

/// Analytic Hessian `H(x) = ∇²f(x): Param → Output` for second-order
/// solvers (Newton, trust-region-Newton). Like [`Jacobian`], the
/// associated `Output` matrix type lets solvers bound on the
/// linear-algebra ops they need
/// ([`LinearSolveSpd`](crate::core::math::LinearSolveSpd),
/// [`SymmetricEigen`](crate::core::math::SymmetricEigen), …) without
/// baking in a backend.
///
/// # Contract
///
/// - **Implementor must:** be a *pure* function of `param`, with the
///   same call-order independence as [`CostFunction::cost`].
/// - **Implementor must:** return a **symmetric** `n × n` matrix where
///   `n = param.len()`. The `(i, j)` entry is `∂²f / ∂xᵢ∂xⱼ`. Shape is
///   fixed across iterates.
/// - The Hessian must agree with [`CostFunction::cost`] (and
///   [`Gradient::gradient`] when present): it is the actual second
///   derivative, not a finite-difference approximation, unless the
///   implementor accepts the loss in solver convergence behavior.
///
/// # Backends
///
/// Wired up for the LA-heavy backends only, mirroring [`Jacobian`]:
///
/// - `Param = nalgebra::DVector<f64>` → `Output = nalgebra::DMatrix<f64>`
///   (rides on the `nalgebra` feature).
/// - `Param = faer::Col<f64>` → `Output = faer::Mat<f64>` (rides on the
///   `faer` feature).
///
/// `Vec<f64>` and `ndarray::Array1<f64>` deliberately have no `Hessian`
/// impl — there's no honest dense matrix type to pair with them. Per
/// tenet 5 in `AGENTS.md`, missing backend coverage is a compile-time
/// error rather than a runtime surprise.
///
/// # Examples
///
/// ```
/// # #[cfg(feature = "nalgebra")] {
/// use basin::Hessian;
/// use nalgebra::{DMatrix, DVector};
///
/// // f(x) = x₀² + x₁² has constant Hessian 2·I.
/// struct Sphere;
/// impl Hessian for Sphere {
///     type Param = DVector<f64>;
///     type Output = DMatrix<f64>;
///     fn hessian(&self, _x: &DVector<f64>) -> DMatrix<f64> {
///         2.0 * DMatrix::identity(2, 2)
///     }
/// }
///
/// let h = Sphere.hessian(&DVector::from_vec(vec![1.0, 1.0]));
/// assert_eq!(h[(0, 0)], 2.0);
/// # }
/// ```
pub trait Hessian {
    /// The parameter type the Hessian is defined over (matches
    /// [`CostFunction::Param`]).
    type Param;
    /// The Hessian matrix type, shape `n × n` and symmetric.
    type Output;

    /// Evaluate the Hessian at `param`.
    fn hessian(&self, param: &Self::Param) -> Self::Output;
}
