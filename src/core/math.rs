pub trait ScaledAdd<S> {
    fn scaled_add(&mut self, scalar: S, other: &Self);
}

impl ScaledAdd<f64> for Vec<f64> {
    fn scaled_add(&mut self, scalar: f64, other: &Self) {
        assert_eq!(self.len(), other.len(), "scaled_add: length mismatch");
        for (x, y) in self.iter_mut().zip(other.iter()) {
            *x += scalar * y;
        }
    }
}

pub trait NormSquared {
    fn norm_squared(&self) -> f64;
}

impl NormSquared for Vec<f64> {
    fn norm_squared(&self) -> f64 {
        self.iter().map(|x| x * x).sum()
    }
}
