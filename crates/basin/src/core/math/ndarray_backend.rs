use ndarray::{Array1, ArrayBase, Data, DataMut, Dimension};
use rand::{Rng, RngExt};
use rand_distr::{Distribution, StandardNormal};

use super::cl_scaling::{
    cl_scaling_pair, max_feasible_step_component, project_strictly_inside_component,
    BoxAffineScaling,
};
use super::sample::{SampleStandardNormal, SampleUniformBox};
use super::{
    ClampInPlace, ComponentDivAssign, ComponentMaxAssign, ComponentMulAssign, Dot,
    FloorZerosInPlace, NegInPlace, NormInfinity, NormSquared, ScaleInPlace, ScaledAdd, VectorLen,
};

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

impl SampleUniformBox for Array1<f64> {
    fn sample_uniform_box<R: Rng + ?Sized>(lower: &Self, upper: &Self, rng: &mut R) -> Self {
        assert_eq!(
            lower.len(),
            upper.len(),
            "sample_uniform_box: bounds length mismatch"
        );
        Array1::from_shape_fn(lower.len(), |i| rng.random_range(lower[i]..=upper[i]))
    }
}

impl VectorLen for Array1<f64> {
    fn vec_len(&self) -> usize {
        self.len()
    }
}

impl SampleStandardNormal for Array1<f64> {
    fn sample_standard_normal<R: Rng + ?Sized>(template: &Self, rng: &mut R) -> Self {
        Array1::from_shape_fn(template.len(), |_| StandardNormal.sample(rng))
    }
}

impl<S, D> ScaleInPlace for ArrayBase<S, D>
where
    S: DataMut<Elem = f64>,
    D: Dimension,
{
    fn scale_in_place(&mut self, scalar: f64) {
        self.map_inplace(|x| *x *= scalar);
    }
}

impl<S, D> ComponentMulAssign for ArrayBase<S, D>
where
    S: DataMut<Elem = f64>,
    D: Dimension,
{
    fn component_mul_assign(&mut self, other: &Self) {
        assert_eq!(
            self.shape(),
            other.shape(),
            "component_mul_assign: shape mismatch"
        );
        self.zip_mut_with(other, |x, y| *x *= y);
    }
}

impl<S, D> ComponentMaxAssign for ArrayBase<S, D>
where
    S: DataMut<Elem = f64>,
    D: Dimension,
{
    fn component_max_assign(&mut self, other: &Self) {
        assert_eq!(
            self.shape(),
            other.shape(),
            "component_max_assign: shape mismatch"
        );
        self.zip_mut_with(other, |x, y| *x = x.max(*y));
    }
}

impl<S, D> FloorZerosInPlace for ArrayBase<S, D>
where
    S: DataMut<Elem = f64>,
    D: Dimension,
{
    fn floor_zeros_in_place(&mut self, value: f64) {
        self.map_inplace(|x| {
            if *x <= 0.0 {
                *x = value;
            }
        });
    }
}

impl<S, D> ComponentDivAssign for ArrayBase<S, D>
where
    S: DataMut<Elem = f64>,
    D: Dimension,
{
    fn component_div_assign(&mut self, other: &Self) {
        assert_eq!(
            self.shape(),
            other.shape(),
            "component_div_assign: shape mismatch"
        );
        self.zip_mut_with(other, |x, y| *x /= y);
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

impl<S, D> BoxAffineScaling for ArrayBase<S, D>
where
    S: DataMut<Elem = f64>,
    D: Dimension,
{
    fn compute_cl_scaling(
        &self,
        gradient: &Self,
        lower: &Self,
        upper: &Self,
        d_sq: &mut Self,
        c_diag: &mut Self,
    ) {
        assert_eq!(
            self.shape(),
            gradient.shape(),
            "compute_cl_scaling: gradient shape mismatch"
        );
        assert_eq!(
            self.shape(),
            lower.shape(),
            "compute_cl_scaling: lower shape mismatch"
        );
        assert_eq!(
            self.shape(),
            upper.shape(),
            "compute_cl_scaling: upper shape mismatch"
        );
        assert_eq!(
            self.shape(),
            d_sq.shape(),
            "compute_cl_scaling: d_sq shape mismatch"
        );
        assert_eq!(
            self.shape(),
            c_diag.shape(),
            "compute_cl_scaling: c_diag shape mismatch"
        );
        ndarray::Zip::from(d_sq)
            .and(c_diag)
            .and(self)
            .and(gradient)
            .and(lower)
            .and(upper)
            .for_each(|d, c, &x, &g, &l, &u| {
                let (d_sq_i, c_i) = cl_scaling_pair(x, g, l, u);
                *d = d_sq_i;
                *c = c_i;
            });
    }

    fn max_feasible_step(&self, step: &Self, lower: &Self, upper: &Self) -> f64 {
        assert_eq!(
            self.shape(),
            step.shape(),
            "max_feasible_step: step shape mismatch"
        );
        assert_eq!(
            self.shape(),
            lower.shape(),
            "max_feasible_step: lower shape mismatch"
        );
        assert_eq!(
            self.shape(),
            upper.shape(),
            "max_feasible_step: upper shape mismatch"
        );
        let mut tau = f64::INFINITY;
        ndarray::Zip::from(self)
            .and(step)
            .and(lower)
            .and(upper)
            .for_each(|&x, &s, &l, &u| {
                let t = max_feasible_step_component(x, s, l, u);
                if t < tau {
                    tau = t;
                }
            });
        tau
    }

    fn cl_kkt_inf_norm(&self, d_sq: &Self) -> f64 {
        assert_eq!(
            self.shape(),
            d_sq.shape(),
            "cl_kkt_inf_norm: shape mismatch"
        );
        let mut best = 0.0_f64;
        ndarray::Zip::from(self).and(d_sq).for_each(|&v, &d| {
            let candidate = v.abs() / d;
            if candidate > best {
                best = candidate;
            }
        });
        best
    }

    fn weighted_norm_squared(&self, weights: &Self) -> f64 {
        assert_eq!(
            self.shape(),
            weights.shape(),
            "weighted_norm_squared: shape mismatch"
        );
        let mut sum = 0.0_f64;
        ndarray::Zip::from(self)
            .and(weights)
            .for_each(|&v, &w| sum += v * v * w);
        sum
    }

    fn project_strictly_inside(&mut self, lower: &Self, upper: &Self, rstep: f64) {
        assert_eq!(
            self.shape(),
            lower.shape(),
            "project_strictly_inside: lower shape mismatch"
        );
        assert_eq!(
            self.shape(),
            upper.shape(),
            "project_strictly_inside: upper shape mismatch"
        );
        ndarray::Zip::from(self)
            .and(lower)
            .and(upper)
            .for_each(|x, &l, &u| {
                *x = project_strictly_inside_component(*x, l, u, rstep);
            });
    }
}
