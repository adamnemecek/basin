pub mod brent;
pub mod gradient_descent;
pub mod nelder_mead;

#[cfg(feature = "nalgebra")]
pub mod bfgs;

#[cfg(feature = "nalgebra")]
pub use bfgs::BFGS;
pub use brent::Brent;
pub use gradient_descent::GradientDescent;
pub use nelder_mead::NelderMead;
