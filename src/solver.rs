pub mod gradient_descent;
pub mod step_size;

pub use gradient_descent::GradientDescent;
pub use step_size::{Backtracking, Constant, StepSize};
