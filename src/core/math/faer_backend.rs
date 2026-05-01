use faer::Col;

use super::{NormInfinity, NormSquared, ScaledAdd};

impl ScaledAdd<f64> for Col<f64> {
    fn scaled_add(&mut self, scalar: f64, other: &Self) {
        assert_eq!(self.nrows(), other.nrows(), "scaled_add: shape mismatch");
        faer::zip!(self.as_mut(), other.as_ref()).for_each(|faer::unzip!(x, y)| *x += scalar * *y);
    }
}

impl NormSquared for Col<f64> {
    fn norm_squared(&self) -> f64 {
        self.iter().map(|x| x * x).sum()
    }
}

impl NormInfinity for Col<f64> {
    fn norm_infinity(&self) -> f64 {
        self.iter().map(|x| x.abs()).fold(0.0, f64::max)
    }
}
