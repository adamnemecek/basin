use ndarray::{ArrayBase, Data, DataMut, Dimension};

use super::{NormInfinity, NormSquared, ScaledAdd};

impl<S, D> ScaledAdd<f64> for ArrayBase<S, D>
where
    S: DataMut<Elem = f64>,
    D: Dimension,
{
    fn scaled_add(&mut self, scalar: f64, other: &Self) {
        assert_eq!(self.shape(), other.shape(), "scaled_add: shape mismatch");
        self.zip_mut_with(other, |x, y| *x += scalar * y);
    }
}

impl<S, D> NormSquared for ArrayBase<S, D>
where
    S: Data<Elem = f64>,
    D: Dimension,
{
    fn norm_squared(&self) -> f64 {
        self.iter().map(|x| x * x).sum()
    }
}

impl<S, D> NormInfinity for ArrayBase<S, D>
where
    S: Data<Elem = f64>,
    D: Dimension,
{
    fn norm_infinity(&self) -> f64 {
        self.iter().map(|x| x.abs()).fold(0.0, f64::max)
    }
}
