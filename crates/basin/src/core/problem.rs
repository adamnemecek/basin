//! Problem traits the user implements about their objective. Solvers
//! bind on whichever subset they need (e.g. gradient descent requires
//! [`CostFunction`] *and* [`Gradient`]; Nelder-Mead only needs
//! [`CostFunction`]).
//!
//! # Soft reject vs hard abort
//!
//! Every problem trait method returns `Result<_, Self::Error>`. The two
//! ways to signal "something went wrong" are *deliberately* distinct:
//!
//! - **Soft reject (`Ok(f64::INFINITY)`)** — return `+∞` from
//!   [`CostFunction::cost`] to reject a single point. Line searches treat
//!   it as worse and retreat; population solvers treat it as worst
//!   fitness. This is the right channel for "this `x` is outside my
//!   domain, but the solve should continue."
//! - **Hard abort (`Err(_)`)** — return `Err` to terminate the entire
//!   solve. The error bubbles all the way out of
//!   [`Executor::run`](crate::Executor::run) typed as
//!   `Result<_, P::Error>`. Use this when the failure is *not* about a
//!   particular `x` — a downstream service vanished, the user pressed
//!   cancel, an early-stopping criterion in the problem's own state fired.
//!
//! Problems that never fail in this way pick
//! `type Error = std::convert::Infallible;` (or
//! [`!`](https://doc.rust-lang.org/std/primitive.never.html) on nightly).
//! Niche optimization collapses `Result<f64, Infallible>` to `f64` layout,
//! so the happy path stays zero-cost.

/// Scalar-valued objective `f(x): Param → Output`. The smallest
/// problem trait — every solver binds at least on this.
///
/// # Contract
///
/// - **Implementor must:** be a *pure* function of `param` —
///   evaluating at the same `param` twice must return the same
///   `Output` (or the same `Err`). Solvers cache costs across iterations,
///   line searches reuse evaluations, and termination criteria assume the
///   cost they read from the state matches what a fresh `cost(param)`
///   would return.
/// - **Implementor must not:** assume any particular call order or
///   frequency. Solvers may evaluate at exploratory points outside the
///   accepted iterate sequence (line-search probes, Nelder-Mead
///   reflections / contractions / shrinks, finite-difference probes).
///
/// # Soft reject vs hard abort
///
/// See the [module docs](self#soft-reject-vs-hard-abort). Return
/// `Ok(f64::INFINITY)` to *reject one point*; return `Err(_)` to abort
/// the entire solve.
///
/// # Examples
///
/// A never-fails problem uses [`Infallible`](std::convert::Infallible) as
/// its error:
///
/// ```
/// use basin::CostFunction;
///
/// struct Sphere;
/// impl CostFunction for Sphere {
///     type Param = Vec<f64>;
///     type Output = f64;
///     type Error = std::convert::Infallible;
///     fn cost(&self, x: &Vec<f64>) -> Result<f64, std::convert::Infallible> {
///         Ok(x.iter().map(|xi| xi * xi).sum())
///     }
/// }
///
/// assert_eq!(Sphere.cost(&vec![3.0, 4.0]).unwrap(), 25.0);
/// ```
pub trait CostFunction {
    /// The parameter type the objective is defined over.
    type Param;
    /// Scalar cost type. In practice `f64` (see `AGENTS.md`'s
    /// provisional choices).
    type Output;
    /// User-chosen hard-abort error. Pick
    /// [`std::convert::Infallible`] when the cost cannot fail — its
    /// niche optimization keeps `Result<f64, Infallible>` the same
    /// layout as bare `f64` on the happy path.
    type Error;

    /// Evaluate the objective at `param`.
    fn cost(&self, param: &Self::Param) -> Result<Self::Output, Self::Error>;
}

/// Analytic gradient `∇f(x): Param → Gradient`. Required by
/// first-order solvers (gradient descent, BFGS, …).
///
/// `Gradient` is a *subtrait* of [`CostFunction`]: a gradient is the
/// gradient *of* a cost, so the parameter and error types are inherited
/// and the two cannot disagree.
///
/// # Contract
///
/// - **Implementor must:** be a *pure* function of `param`, with the
///   same call-order independence as [`CostFunction::cost`].
/// - **Implementor must:** return a `Gradient` whose shape matches
///   `param` so solver math (`x ← x − α·∇f(x)`) lines up. Most
///   problems have `Gradient = Param`, which is what the shipped
///   solvers' bounds expect (e.g. `Gradient<Gradient = V>` paired with
///   `CostFunction<Param = V>`).
/// - The gradient must agree with [`CostFunction::cost`]: it is the
///   actual derivative, not a finite-difference approximation unless
///   the implementor is happy taking the loss in solver
///   convergence behavior.
///
/// # Fused evaluation
///
/// When a solver needs *both* `f(x)` and `∇f(x)` at the same point —
/// which it almost always does at the start of every iteration —
/// it calls [`cost_and_gradient`](Self::cost_and_gradient). The default
/// body simply calls [`CostFunction::cost`] and [`Gradient::gradient`]
/// in turn, which is the right answer for most problems and what
/// users get for free.
///
/// **Override `cost_and_gradient` when the two share substantial
/// intermediate work** — autodiff tapes, forward-mode adjoints,
/// neural-net activations, expensive simulation state. The default
/// then becomes a no-op and the solver picks up the fusion savings
/// without any further change.
///
/// Cost-only callers (line searches probing trial steps, cost-only
/// termination criteria, derivative-free solvers) keep calling
/// [`CostFunction::cost`] directly — no waste from the fused method.
///
/// # Examples
///
/// ```
/// use basin::{CostFunction, Gradient};
///
/// struct Sphere;
/// impl CostFunction for Sphere {
///     type Param = Vec<f64>;
///     type Output = f64;
///     type Error = std::convert::Infallible;
///     fn cost(&self, x: &Vec<f64>) -> Result<f64, std::convert::Infallible> {
///         Ok(x.iter().map(|xi| xi * xi).sum())
///     }
/// }
/// impl Gradient for Sphere {
///     type Gradient = Vec<f64>;
///     fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, std::convert::Infallible> {
///         Ok(x.iter().map(|xi| 2.0 * xi).collect())
///     }
/// }
///
/// assert_eq!(Sphere.gradient(&vec![1.0, 2.0]).unwrap(), vec![2.0, 4.0]);
/// ```
///
/// Fusion override (cost and gradient share `x * x`):
///
/// ```
/// use basin::{CostFunction, Gradient};
///
/// struct Sphere;
/// impl CostFunction for Sphere {
///     type Param = Vec<f64>;
///     type Output = f64;
///     type Error = std::convert::Infallible;
///     fn cost(&self, x: &Vec<f64>) -> Result<f64, std::convert::Infallible> {
///         Ok(x.iter().map(|xi| xi * xi).sum())
///     }
/// }
/// impl Gradient for Sphere {
///     type Gradient = Vec<f64>;
///     fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, std::convert::Infallible> {
///         Ok(x.iter().map(|xi| 2.0 * xi).collect())
///     }
///     fn cost_and_gradient(
///         &self,
///         x: &Vec<f64>,
///     ) -> Result<(f64, Vec<f64>), std::convert::Infallible> {
///         // Single pass over x; the per-element work is shared.
///         let mut cost = 0.0;
///         let grad = x
///             .iter()
///             .map(|xi| {
///                 cost += xi * xi;
///                 2.0 * xi
///             })
///             .collect();
///         Ok((cost, grad))
///     }
/// }
/// ```
pub trait Gradient: CostFunction {
    /// The gradient type. Typically the same as
    /// [`CostFunction::Param`].
    type Gradient;

    /// Evaluate the gradient at `param`.
    fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient, Self::Error>;

    /// Evaluate cost *and* gradient at `param` in one call. The default
    /// body delegates to [`CostFunction::cost`] and
    /// [`Gradient::gradient`]; override when shared intermediate work
    /// can be amortized across the two.
    ///
    /// **Contract.** The returned `(cost, gradient)` pair must equal
    /// what [`CostFunction::cost`] and [`Gradient::gradient`] would
    /// return separately at the same `param`. Solvers and the framework
    /// switch freely between the fused call and individual calls
    /// depending on what's needed at a given point; divergence breaks
    /// caching invariants.
    ///
    /// **Eval counting.** One fused call counts as one cost evaluation
    /// *and* one gradient evaluation: it produced both values, in the
    /// work of one fused evaluation.
    fn cost_and_gradient(
        &self,
        param: &Self::Param,
    ) -> Result<(Self::Output, Self::Gradient), Self::Error> {
        Ok((self.cost(param)?, self.gradient(param)?))
    }
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
/// # Soft reject vs hard abort
///
/// `Residual` carries its *own* [`Error`](Residual::Error) (independent
/// of [`CostFunction::Error`]); the soft/hard split from the
/// [module docs](self#soft-reject-vs-hard-abort) applies identically.
/// NLLS solvers `?`-propagate residual errors and treat any `Err` as a
/// hard abort.
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
///     type Error = std::convert::Infallible;
///     fn residual(&self, x: &Vec<f64>) -> Result<Vec<f64>, std::convert::Infallible> {
///         Ok(vec![x[0] - 1.0, x[1] - 2.0])
///     }
/// }
///
/// assert_eq!(
///     Affine.residual(&vec![0.0, 0.0]).unwrap(),
///     vec![-1.0, -2.0]
/// );
/// ```
pub trait Residual {
    /// The parameter type the residual is defined over (matches
    /// [`CostFunction::Param`]).
    type Param;
    /// The residual vector type. Length is the number of residuals `m`,
    /// independent of `param.len() = n`.
    type Output;
    /// User-chosen hard-abort error. Independent of
    /// [`CostFunction::Error`] — the trait families are orthogonal
    /// (NLLS solvers bind on `Residual` + [`Jacobian`]; first-order
    /// solvers bind on `CostFunction` + [`Gradient`]).
    type Error;

    /// Evaluate the residual at `param`.
    fn residual(&self, param: &Self::Param) -> Result<Self::Output, Self::Error>;
}

/// Analytic Jacobian `J(x) = ∂r/∂x: Param → Jacobian` for least-squares
/// solvers (Gauss-Newton, LM, TRF). The associated `Jacobian` matrix
/// type is what lets solvers bound on the linear-algebra ops they need
/// ([`MatVec`](crate::core::math::MatVec),
/// [`LinearSolveSpd`](crate::core::math::LinearSolveSpd), …) without
/// baking in a specific backend or assuming density.
///
/// `Jacobian` is a *subtrait* of [`Residual`]: a Jacobian is the
/// Jacobian *of* a residual, so the parameter and error types are
/// inherited.
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
/// # Fused evaluation
///
/// NLLS solvers (Gauss-Newton, LM, TRF) evaluate `r(x)` and `J(x)`
/// together at every accepted iterate — and `r(x)` is usually the
/// dominant cost, with `J(x)` reusing intermediate state (forward-mode
/// AD on the residual graph, FE assembly, simulation adjoints).
/// [`residual_and_jacobian`](Self::residual_and_jacobian) provides the
/// fused entry point. The default body calls [`Residual::residual`] and
/// [`Jacobian::jacobian`] in turn; override when work can be shared.
///
/// # Backends
///
/// Wired up for the LA-heavy backends only:
///
/// - `Param = nalgebra::DVector<f64>` → `Jacobian = nalgebra::DMatrix<f64>`
///   (dense) or `nalgebra_sparse::CscMatrix<f64>` (sparse). Both ride
///   on the `nalgebra` feature.
/// - `Param = faer::Col<f64>` → `Jacobian = faer::Mat<f64>` (dense) or
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
/// use basin::{Jacobian, Residual};
/// use nalgebra::{DMatrix, DVector};
///
/// struct Affine;
/// impl Residual for Affine {
///     type Param = DVector<f64>;
///     type Output = DVector<f64>;
///     type Error = std::convert::Infallible;
///     fn residual(
///         &self,
///         x: &DVector<f64>,
///     ) -> Result<DVector<f64>, std::convert::Infallible> {
///         Ok(DVector::from_vec(vec![x[0] - 1.0, x[1] - 2.0]))
///     }
/// }
/// impl Jacobian for Affine {
///     type Jacobian = DMatrix<f64>;
///     fn jacobian(
///         &self,
///         _x: &DVector<f64>,
///     ) -> Result<DMatrix<f64>, std::convert::Infallible> {
///         Ok(DMatrix::identity(2, 2))
///     }
/// }
///
/// let j = Affine.jacobian(&DVector::from_vec(vec![0.0, 0.0])).unwrap();
/// assert_eq!(j[(0, 0)], 1.0);
/// # }
/// ```
pub trait Jacobian: Residual {
    /// The Jacobian matrix type, shape `m × n`.
    type Jacobian;

    /// Evaluate the Jacobian at `param`.
    fn jacobian(&self, param: &Self::Param) -> Result<Self::Jacobian, <Self as Residual>::Error>;

    /// Evaluate residual *and* Jacobian at `param` in one call. The
    /// default body delegates to [`Residual::residual`] and
    /// [`Jacobian::jacobian`]; override when shared intermediate work
    /// can be amortized across the two — common in NLLS where `r(x)`
    /// reuses forward-mode AD state that `J(x)` continues from.
    ///
    /// **Contract.** The returned `(residual, jacobian)` pair must
    /// equal what [`Residual::residual`] and [`Jacobian::jacobian`]
    /// would return separately at the same `param`.
    ///
    /// **Eval counting.** NLLS solvers count one fused call as one
    /// `cost_evals` *and* one `gradient_evals` increment — the same
    /// convention solvers use for separate calls, because `½‖r‖²`
    /// plays the role of cost and `Jᵀr` the role of gradient.
    fn residual_and_jacobian(
        &self,
        param: &Self::Param,
    ) -> Result<(<Self as Residual>::Output, Self::Jacobian), <Self as Residual>::Error> {
        Ok((self.residual(param)?, self.jacobian(param)?))
    }
}

/// Analytic Hessian `H(x) = ∇²f(x): Param → Hessian` for second-order
/// solvers (Newton, trust-region-Newton). The associated `Hessian`
/// matrix type lets solvers bound on the linear-algebra ops they need
/// ([`LinearSolveSpd`](crate::core::math::LinearSolveSpd),
/// [`SymmetricEigen`](crate::core::math::SymmetricEigen), …) without
/// baking in a backend.
///
/// `Hessian` is a *subtrait* of [`Gradient`] (which is a subtrait of
/// [`CostFunction`]): a Hessian is the second derivative of a cost.
/// The error type is therefore [`CostFunction::Error`].
///
/// # Contract
///
/// - **Implementor must:** be a *pure* function of `param`, with the
///   same call-order independence as [`CostFunction::cost`].
/// - **Implementor must:** return a **symmetric** `n × n` matrix where
///   `n = param.len()`. The `(i, j)` entry is `∂²f / ∂xᵢ∂xⱼ`. Shape is
///   fixed across iterates.
/// - The Hessian must agree with [`CostFunction::cost`] and
///   [`Gradient::gradient`]: it is the actual second derivative, not a
///   finite-difference approximation, unless the implementor accepts
///   the loss in solver convergence behavior.
///
/// # Fused evaluation
///
/// Second-order solvers evaluate `f`, `∇f`, and `∇²f` together at
/// every accepted iterate. The
/// [`cost_and_gradient_and_hessian`](Self::cost_and_gradient_and_hessian)
/// method provides the fused entry point. The default body composes
/// [`Gradient::cost_and_gradient`] with [`Hessian::hessian`]; override
/// when all three share intermediate state.
///
/// # Backends
///
/// Wired up for the LA-heavy backends only, mirroring [`Jacobian`]:
///
/// - `Param = nalgebra::DVector<f64>` → `Hessian = nalgebra::DMatrix<f64>`
///   (rides on the `nalgebra` feature).
/// - `Param = faer::Col<f64>` → `Hessian = faer::Mat<f64>` (rides on
///   the `faer` feature).
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
/// use basin::{CostFunction, Gradient, Hessian};
/// use nalgebra::{DMatrix, DVector};
///
/// // f(x) = x₀² + x₁² has constant Hessian 2·I.
/// struct Sphere;
/// impl CostFunction for Sphere {
///     type Param = DVector<f64>;
///     type Output = f64;
///     type Error = std::convert::Infallible;
///     fn cost(&self, x: &DVector<f64>) -> Result<f64, std::convert::Infallible> {
///         Ok(x.dot(x))
///     }
/// }
/// impl Gradient for Sphere {
///     type Gradient = DVector<f64>;
///     fn gradient(
///         &self,
///         x: &DVector<f64>,
///     ) -> Result<DVector<f64>, std::convert::Infallible> {
///         Ok(2.0 * x)
///     }
/// }
/// impl Hessian for Sphere {
///     type Hessian = DMatrix<f64>;
///     fn hessian(
///         &self,
///         x: &DVector<f64>,
///     ) -> Result<DMatrix<f64>, std::convert::Infallible> {
///         Ok(2.0 * DMatrix::identity(x.len(), x.len()))
///     }
/// }
///
/// let h = Sphere.hessian(&DVector::from_vec(vec![1.0, 1.0])).unwrap();
/// assert_eq!(h[(0, 0)], 2.0);
/// # }
/// ```
pub trait Hessian: Gradient {
    /// The Hessian matrix type, shape `n × n` and symmetric.
    type Hessian;

    /// Evaluate the Hessian at `param`.
    fn hessian(&self, param: &Self::Param) -> Result<Self::Hessian, <Self as CostFunction>::Error>;

    /// Evaluate cost, gradient, *and* Hessian at `param` in one call.
    /// The default body delegates to [`Gradient::cost_and_gradient`]
    /// followed by [`Hessian::hessian`]; override when all three share
    /// intermediate work.
    ///
    /// **Contract.** The returned triple must equal what the three
    /// methods would return separately at the same `param`.
    #[allow(clippy::type_complexity)]
    fn cost_and_gradient_and_hessian(
        &self,
        param: &Self::Param,
    ) -> Result<
        (
            <Self as CostFunction>::Output,
            <Self as Gradient>::Gradient,
            Self::Hessian,
        ),
        <Self as CostFunction>::Error,
    > {
        let (cost, grad) = self.cost_and_gradient(param)?;
        Ok((cost, grad, self.hessian(param)?))
    }
}
