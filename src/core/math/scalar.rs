use super::{NormInfinity, NormSquared, ScaledAdd};

impl ScaledAdd<f64> for f64 {
    fn scaled_add(&mut self, scalar: f64, other: &Self) {
        *self += scalar * other;
    }
}

impl NormSquared for f64 {
    fn norm_squared(&self) -> f64 {
        self * self
    }
}

impl NormInfinity for f64 {
    fn norm_infinity(&self) -> f64 {
        self.abs()
    }
}
