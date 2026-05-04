//! 2D Booth function.
//!
//! `f(x, y) = (x + 2y − 7)² + (2x + y − 5)²`
//!
//! Smooth, convex 2D quadratic. Global minimum at `(x, y) = (1, 3)` with
//! `f = 0`. Usual search domain is `x, y ∈ [-10, 10]`. The cost is the squared
//! residual of the linear system `[[1, 2], [2, 1]] · [x, y]ᵀ = [7, 5]ᵀ`, so
//! the Hessian is constant SPD and any reasonable first-order method should
//! converge in a handful of steps. Useful as an easy convex 2D gradient sanity
//! check distinct from the separable Sphere.

use core::marker::PhantomData;

use super::spec::{Dimensionality, HasSpec, ProblemSpec, Properties, Reference};
use crate::{CostFunction, Gradient};

/// Evaluates the Booth function at `x`. Requires `x.len() == 2`.
pub fn booth(x: &[f64]) -> f64 {
    debug_assert_eq!(x.len(), 2);
    let (a, b) = (x[0], x[1]);
    let t1 = a + 2.0 * b - 7.0;
    let t2 = 2.0 * a + b - 5.0;
    t1 * t1 + t2 * t2
}

/// Writes the Booth gradient at `x` into `out`. Both slices must have length 2.
pub fn booth_gradient(x: &[f64], out: &mut [f64]) {
    debug_assert_eq!(x.len(), 2);
    debug_assert_eq!(out.len(), 2);
    let (a, b) = (x[0], x[1]);
    let t1 = a + 2.0 * b - 7.0;
    let t2 = 2.0 * a + b - 5.0;
    // ∂f/∂x = 2·t1 + 4·t2
    out[0] = 2.0 * t1 + 4.0 * t2;
    // ∂f/∂y = 4·t1 + 2·t2
    out[1] = 4.0 * t1 + 2.0 * t2;
}

/// Pre-wrapped Booth problem (fixed 2D). Generic over the parameter backend
/// `P`; the default `P = Vec<f64>` lets you write `Booth::default()` for the
/// common case. Backend impls (`nalgebra::DVector<f64>`, `ndarray::Array1<f64>`,
/// `faer::Col<f64>`) are gated behind their respective features.
pub struct Booth<P = Vec<f64>>(PhantomData<fn() -> P>);

impl<P> Booth<P> {
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<P> Default for Booth<P> {
    fn default() -> Self {
        Self::new()
    }
}

pub static BOOTH_SPEC: ProblemSpec = ProblemSpec {
    name: "Booth",
    dim: Dimensionality::Fixed(2),
    properties: Properties {
        smooth: true,
        differentiable: true,
        // Sum of squared affine forms in (x, y); the Hessian [[10, 8], [8, 10]]
        // is constant and positive definite, so f is strictly convex on R².
        convex: true,
        unimodal: true,
        // Cross term 8xy in the expansion prevents per-coordinate decomposition.
        separable: false,
        scalable: false,
    },
    references: &[Reference {
        citation: "Jamil & Yang (2013)",
        title: "A Literature Survey of Benchmark Functions For Global Optimisation Problems",
        source: "International Journal of Mathematical Modelling and Numerical Optimisation, 4(2), 150–194",
        doi: Some("10.1504/IJMMNO.2013.055204"),
        url: Some("https://arxiv.org/abs/1308.4008"),
    }],
    description: "Smooth convex 2D quadratic: f(x, y) = (x + 2y − 7)² + \
                  (2x + y − 5)². Global minimum at (x, y) = (1, 3), value 0. \
                  Usual search domain is x, y ∈ [-10, 10]. The Hessian is \
                  constant and positive definite, so first-order methods \
                  converge in a handful of steps.",
};

impl<P> HasSpec for Booth<P> {
    const SPEC: &'static ProblemSpec = &BOOTH_SPEC;
}

impl CostFunction for Booth<Vec<f64>> {
    type Param = Vec<f64>;
    type Output = f64;
    fn cost(&self, x: &Vec<f64>) -> f64 {
        booth(x)
    }
}

impl Gradient for Booth<Vec<f64>> {
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;
    fn gradient(&self, x: &Vec<f64>) -> Vec<f64> {
        let mut out = vec![0.0; x.len()];
        booth_gradient(x, &mut out);
        out
    }
}

#[cfg(feature = "nalgebra")]
mod nalgebra_impl {
    use super::{booth, booth_gradient, Booth};
    use crate::{CostFunction, Gradient};
    use nalgebra::DVector;

    impl CostFunction for Booth<DVector<f64>> {
        type Param = DVector<f64>;
        type Output = f64;
        fn cost(&self, x: &DVector<f64>) -> f64 {
            booth(x.as_slice())
        }
    }

    impl Gradient for Booth<DVector<f64>> {
        type Param = DVector<f64>;
        type Gradient = DVector<f64>;
        fn gradient(&self, x: &DVector<f64>) -> DVector<f64> {
            let mut out = DVector::zeros(x.len());
            booth_gradient(x.as_slice(), out.as_mut_slice());
            out
        }
    }
}

#[cfg(feature = "ndarray")]
mod ndarray_impl {
    use super::{booth, booth_gradient, Booth};
    use crate::{CostFunction, Gradient};
    use ndarray::Array1;

    impl CostFunction for Booth<Array1<f64>> {
        type Param = Array1<f64>;
        type Output = f64;
        fn cost(&self, x: &Array1<f64>) -> f64 {
            booth(x.as_slice().expect("Array1 is contiguous"))
        }
    }

    impl Gradient for Booth<Array1<f64>> {
        type Param = Array1<f64>;
        type Gradient = Array1<f64>;
        fn gradient(&self, x: &Array1<f64>) -> Array1<f64> {
            let mut out = Array1::zeros(x.len());
            booth_gradient(
                x.as_slice().expect("Array1 is contiguous"),
                out.as_slice_mut().expect("Array1 is contiguous"),
            );
            out
        }
    }
}

#[cfg(feature = "faer")]
mod faer_impl {
    use super::Booth;
    use crate::{CostFunction, Gradient};
    use faer::Col;

    // faer's `Col` doesn't expose a `&[f64]` directly across all 0.24 APIs we
    // care about, so we evaluate elementwise here rather than routing through
    // the slice-based primitives.
    impl CostFunction for Booth<Col<f64>> {
        type Param = Col<f64>;
        type Output = f64;
        fn cost(&self, x: &Col<f64>) -> f64 {
            debug_assert_eq!(x.nrows(), 2);
            let (a, b) = (x[0], x[1]);
            let t1 = a + 2.0 * b - 7.0;
            let t2 = 2.0 * a + b - 5.0;
            t1 * t1 + t2 * t2
        }
    }

    impl Gradient for Booth<Col<f64>> {
        type Param = Col<f64>;
        type Gradient = Col<f64>;
        fn gradient(&self, x: &Col<f64>) -> Col<f64> {
            debug_assert_eq!(x.nrows(), 2);
            let (a, b) = (x[0], x[1]);
            let t1 = a + 2.0 * b - 7.0;
            let t2 = 2.0 * a + b - 5.0;
            let g0 = 2.0 * t1 + 4.0 * t2;
            let g1 = 4.0 * t1 + 2.0 * t2;
            Col::<f64>::from_fn(2, |i| if i == 0 { g0 } else { g1 })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn booth_minimum_is_zero_at_known_optimum() {
        assert!(booth(&[1.0, 3.0]).abs() < 1e-12);
    }

    #[test]
    fn booth_known_value_at_origin() {
        // f(0, 0) = (-7)² + (-5)² = 49 + 25 = 74.
        assert!((booth(&[0.0, 0.0]) - 74.0).abs() < 1e-12);
    }

    #[test]
    fn gradient_zero_at_minimum() {
        let mut g = vec![0.0; 2];
        booth_gradient(&[1.0, 3.0], &mut g);
        for v in g {
            assert!(v.abs() < 1e-12);
        }
    }

    #[test]
    fn gradient_matches_finite_difference() {
        let x = [-1.2, 0.7];
        let mut g = vec![0.0; x.len()];
        booth_gradient(&x, &mut g);

        let h = 1e-6;
        for i in 0..x.len() {
            let mut xp = x;
            let mut xm = x;
            xp[i] += h;
            xm[i] -= h;
            let fd = (booth(&xp) - booth(&xm)) / (2.0 * h);
            assert!((g[i] - fd).abs() < 1e-5, "i={i}, g={}, fd={fd}", g[i]);
        }
    }

    #[test]
    fn spec_is_wired_up_via_has_spec_trait() {
        let spec = <Booth<Vec<f64>> as HasSpec>::SPEC;
        assert_eq!(spec.name, "Booth");
        assert!(spec.properties.smooth);
        assert!(spec.properties.differentiable);
        assert!(spec.properties.convex);
        assert!(spec.properties.unimodal);
        assert!(matches!(spec.dim, Dimensionality::Fixed(2)));
        assert!(!spec.references.is_empty());
    }
}
