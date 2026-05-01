use nalgebra::{Dim, Matrix, Storage, StorageMut};

use super::{NormInfinity, NormSquared, ScaledAdd};

impl<R, C, S> ScaledAdd<f64> for Matrix<f64, R, C, S>
where
    R: Dim,
    C: Dim,
    S: StorageMut<f64, R, C>,
{
    fn scaled_add(&mut self, scalar: f64, other: &Self) {
        assert_eq!(self.shape(), other.shape(), "scaled_add: shape mismatch");
        self.zip_apply(other, |x, y| *x += scalar * y);
    }
}

impl<R, C, S> NormSquared for Matrix<f64, R, C, S>
where
    R: Dim,
    C: Dim,
    S: Storage<f64, R, C>,
{
    fn norm_squared(&self) -> f64 {
        self.iter().map(|x| x * x).sum()
    }
}

impl<R, C, S> NormInfinity for Matrix<f64, R, C, S>
where
    R: Dim,
    C: Dim,
    S: Storage<f64, R, C>,
{
    fn norm_infinity(&self) -> f64 {
        self.iter().map(|x| x.abs()).fold(0.0, f64::max)
    }
}
