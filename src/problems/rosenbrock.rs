//! N-dimensional Rosenbrock function.
//!
//! `f(x) = Σ_{i=0}^{n-2} [100·(x_{i+1} − x_i²)² + (1 − x_i)²]`
//!
//! Global minimum at `x = (1, …, 1)` with `f = 0`. The classical 2D start is
//! `(−1.2, 1.0)`. The 1D case (`n = 1`) is degenerate and returns 0.

use core::marker::PhantomData;

use crate::{CostFunction, Gradient};

/// Evaluates Rosenbrock's function at `x`.
pub fn rosenbrock(x: &[f64]) -> f64 {
    let mut s = 0.0;
    for i in 0..x.len().saturating_sub(1) {
        let a = x[i + 1] - x[i] * x[i];
        let b = 1.0 - x[i];
        s += 100.0 * a * a + b * b;
    }
    s
}

/// Writes the Rosenbrock gradient at `x` into `out`. Lengths must match.
pub fn rosenbrock_gradient(x: &[f64], out: &mut [f64]) {
    debug_assert_eq!(x.len(), out.len());
    for v in out.iter_mut() {
        *v = 0.0;
    }
    for i in 0..x.len().saturating_sub(1) {
        let a = x[i + 1] - x[i] * x[i];
        out[i] += -400.0 * x[i] * a - 2.0 * (1.0 - x[i]);
        out[i + 1] += 200.0 * a;
    }
}

/// Pre-wrapped Rosenbrock problem. Generic over the parameter backend `P`;
/// the default `P = Vec<f64>` lets you write `Rosenbrock::default()` for the
/// common case. Backend impls (`nalgebra::DVector<f64>`, `ndarray::Array1<f64>`,
/// `faer::Col<f64>`) are gated behind their respective features.
pub struct Rosenbrock<P = Vec<f64>>(PhantomData<fn() -> P>);

impl<P> Rosenbrock<P> {
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<P> Default for Rosenbrock<P> {
    fn default() -> Self {
        Self::new()
    }
}

impl CostFunction for Rosenbrock<Vec<f64>> {
    type Param = Vec<f64>;
    type Output = f64;
    fn cost(&self, x: &Vec<f64>) -> f64 {
        rosenbrock(x)
    }
}

impl Gradient for Rosenbrock<Vec<f64>> {
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;
    fn gradient(&self, x: &Vec<f64>) -> Vec<f64> {
        let mut out = vec![0.0; x.len()];
        rosenbrock_gradient(x, &mut out);
        out
    }
}

#[cfg(feature = "nalgebra")]
mod nalgebra_impl {
    use super::{rosenbrock, rosenbrock_gradient, Rosenbrock};
    use crate::{CostFunction, Gradient};
    use nalgebra::DVector;

    impl CostFunction for Rosenbrock<DVector<f64>> {
        type Param = DVector<f64>;
        type Output = f64;
        fn cost(&self, x: &DVector<f64>) -> f64 {
            rosenbrock(x.as_slice())
        }
    }

    impl Gradient for Rosenbrock<DVector<f64>> {
        type Param = DVector<f64>;
        type Gradient = DVector<f64>;
        fn gradient(&self, x: &DVector<f64>) -> DVector<f64> {
            let mut out = DVector::zeros(x.len());
            rosenbrock_gradient(x.as_slice(), out.as_mut_slice());
            out
        }
    }
}

#[cfg(feature = "ndarray")]
mod ndarray_impl {
    use super::{rosenbrock, rosenbrock_gradient, Rosenbrock};
    use crate::{CostFunction, Gradient};
    use ndarray::Array1;

    // Array1 owns a contiguous buffer, so the `as_slice` calls always succeed.
    impl CostFunction for Rosenbrock<Array1<f64>> {
        type Param = Array1<f64>;
        type Output = f64;
        fn cost(&self, x: &Array1<f64>) -> f64 {
            rosenbrock(x.as_slice().expect("Array1 is contiguous"))
        }
    }

    impl Gradient for Rosenbrock<Array1<f64>> {
        type Param = Array1<f64>;
        type Gradient = Array1<f64>;
        fn gradient(&self, x: &Array1<f64>) -> Array1<f64> {
            let mut out = Array1::zeros(x.len());
            rosenbrock_gradient(
                x.as_slice().expect("Array1 is contiguous"),
                out.as_slice_mut().expect("Array1 is contiguous"),
            );
            out
        }
    }
}

#[cfg(feature = "faer")]
mod faer_impl {
    use super::Rosenbrock;
    use crate::{CostFunction, Gradient};
    use faer::Col;

    // faer's `Col` doesn't expose a `&[f64]` directly across all 0.24 APIs we
    // care about, so we evaluate elementwise here rather than routing through
    // the slice-based primitives.
    impl CostFunction for Rosenbrock<Col<f64>> {
        type Param = Col<f64>;
        type Output = f64;
        fn cost(&self, x: &Col<f64>) -> f64 {
            let n = x.nrows();
            let mut s = 0.0;
            for i in 0..n.saturating_sub(1) {
                let a = x[i + 1] - x[i] * x[i];
                let b = 1.0 - x[i];
                s += 100.0 * a * a + b * b;
            }
            s
        }
    }

    impl Gradient for Rosenbrock<Col<f64>> {
        type Param = Col<f64>;
        type Gradient = Col<f64>;
        fn gradient(&self, x: &Col<f64>) -> Col<f64> {
            let n = x.nrows();
            let mut out = Col::<f64>::zeros(n);
            for i in 0..n.saturating_sub(1) {
                let a = x[i + 1] - x[i] * x[i];
                out[i] += -400.0 * x[i] * a - 2.0 * (1.0 - x[i]);
                out[i + 1] += 200.0 * a;
            }
            out
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rosenbrock_minimum_is_zero_at_ones() {
        assert_eq!(rosenbrock(&[1.0, 1.0]), 0.0);
        assert_eq!(rosenbrock(&[1.0, 1.0, 1.0, 1.0]), 0.0);
    }

    #[test]
    fn rosenbrock_known_value_at_classical_start() {
        // f(-1.2, 1.0) = (1 - (-1.2))^2 + 100*(1 - 1.44)^2 = 4.84 + 19.36 = 24.2
        assert!((rosenbrock(&[-1.2, 1.0]) - 24.2).abs() < 1e-12);
    }

    #[test]
    fn gradient_zero_at_minimum() {
        let mut g = vec![0.0; 4];
        rosenbrock_gradient(&[1.0, 1.0, 1.0, 1.0], &mut g);
        for v in g {
            assert!(v.abs() < 1e-12);
        }
    }

    #[test]
    fn gradient_matches_finite_difference() {
        let x = [-1.2, 1.0, 0.7, 0.4];
        let mut g = vec![0.0; x.len()];
        rosenbrock_gradient(&x, &mut g);

        let h = 1e-6;
        for i in 0..x.len() {
            let mut xp = x;
            let mut xm = x;
            xp[i] += h;
            xm[i] -= h;
            let fd = (rosenbrock(&xp) - rosenbrock(&xm)) / (2.0 * h);
            assert!((g[i] - fd).abs() < 1e-5, "i={i}, g={}, fd={fd}", g[i]);
        }
    }
}
