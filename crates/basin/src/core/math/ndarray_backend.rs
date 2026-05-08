use ndarray::{ArrayBase, Data, DataMut, Dimension};

use super::{ClampInPlace, Dot, NegInPlace, NormInfinity, NormSquared, ScaledAdd};

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

impl<S, D> Dot for ArrayBase<S, D>
where
    S: Data<Elem = f64>,
    D: Dimension,
{
    fn dot(&self, other: &Self) -> f64 {
        assert_eq!(self.shape(), other.shape(), "dot: shape mismatch");
        self.iter().zip(other.iter()).map(|(a, b)| a * b).sum()
    }
}

impl<S, D> NegInPlace for ArrayBase<S, D>
where
    S: DataMut<Elem = f64>,
    D: Dimension,
{
    fn neg_in_place(&mut self) {
        self.map_inplace(|x| *x = -*x);
    }
}

impl<S, D> ClampInPlace for ArrayBase<S, D>
where
    S: DataMut<Elem = f64>,
    D: Dimension,
{
    fn clamp_in_place(&mut self, lower: &Self, upper: &Self) {
        assert_eq!(
            self.shape(),
            lower.shape(),
            "clamp_in_place: lower shape mismatch"
        );
        assert_eq!(
            self.shape(),
            upper.shape(),
            "clamp_in_place: upper shape mismatch"
        );
        ndarray::Zip::from(self)
            .and(lower)
            .and(upper)
            .for_each(|x, &lo, &hi| *x = x.clamp(lo, hi));
    }
}
