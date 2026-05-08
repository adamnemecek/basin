/// Brent's method (1D root / minimum bracketing).
pub mod brent;
/// Pure Gauss-Newton solver for nonlinear least squares.
pub mod gauss_newton;
/// Steepest-descent solver with a pluggable line search.
pub mod gradient_descent;
/// Levenberg-Marquardt solver for nonlinear least squares with
/// Nielsen 1999 damping update.
pub mod levenberg_marquardt;
/// Nelder-Mead derivative-free simplex solver.
pub mod nelder_mead;

/// BFGS quasi-Newton solver (nalgebra-only).
#[cfg(feature = "nalgebra")]
pub mod bfgs;

#[cfg(feature = "nalgebra")]
pub use bfgs::BFGS;
pub use brent::Brent;
pub use gauss_newton::GaussNewton;
pub use gradient_descent::GradientDescent;
pub use levenberg_marquardt::LevenbergMarquardt;
pub use nelder_mead::NelderMead;
