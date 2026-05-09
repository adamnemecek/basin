use faer::linalg::matmul::matmul;
use faer::linalg::solvers::{Llt, Solve};
use faer::{Accum, Col, Mat, Par, Side};
use rand::Rng;

use super::cl_scaling::{
    cl_scaling_pair, max_feasible_step_component, project_strictly_inside_component,
    BoxAffineScaling,
};
use super::linalg::{
    AddDiagonalInPlace, AddDiagonalVectorInPlace, GramMatrix, LinearSolveError, LinearSolveSpd,
    MatTransposeVec, MatVec, MaxDiagonal,
};
use super::sample::SampleUniformBox;
use super::{ClampInPlace, Dot, NegInPlace, NormInfinity, NormSquared, ScaledAdd};

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

impl Dot for Col<f64> {
    fn dot(&self, other: &Self) -> f64 {
        assert_eq!(self.nrows(), other.nrows(), "dot: shape mismatch");
        self.iter().zip(other.iter()).map(|(a, b)| a * b).sum()
    }
}

impl NegInPlace for Col<f64> {
    fn neg_in_place(&mut self) {
        faer::zip!(self.as_mut()).for_each(|faer::unzip!(x)| *x = -*x);
    }
}

impl SampleUniformBox for Col<f64> {
    fn sample_uniform_box<R: Rng + ?Sized>(lower: &Self, upper: &Self, rng: &mut R) -> Self {
        assert_eq!(
            lower.nrows(),
            upper.nrows(),
            "sample_uniform_box: bounds length mismatch"
        );
        Col::<f64>::from_fn(lower.nrows(), |i| rng.random_range(lower[i]..=upper[i]))
    }
}

impl ClampInPlace for Col<f64> {
    fn clamp_in_place(&mut self, lower: &Self, upper: &Self) {
        assert_eq!(
            self.nrows(),
            lower.nrows(),
            "clamp_in_place: lower shape mismatch"
        );
        assert_eq!(
            self.nrows(),
            upper.nrows(),
            "clamp_in_place: upper shape mismatch"
        );
        faer::zip!(self.as_mut(), lower.as_ref(), upper.as_ref())
            .for_each(|faer::unzip!(x, lo, hi)| *x = x.clamp(*lo, *hi));
    }
}

impl BoxAffineScaling for Col<f64> {
    fn compute_cl_scaling(
        &self,
        gradient: &Self,
        lower: &Self,
        upper: &Self,
        d_sq: &mut Self,
        c_diag: &mut Self,
    ) {
        let n = self.nrows();
        assert_eq!(
            n,
            gradient.nrows(),
            "compute_cl_scaling: gradient shape mismatch"
        );
        assert_eq!(n, lower.nrows(), "compute_cl_scaling: lower shape mismatch");
        assert_eq!(n, upper.nrows(), "compute_cl_scaling: upper shape mismatch");
        assert_eq!(n, d_sq.nrows(), "compute_cl_scaling: d_sq shape mismatch");
        assert_eq!(
            n,
            c_diag.nrows(),
            "compute_cl_scaling: c_diag shape mismatch"
        );
        // Faer's `zip!` macro caps at four operands; do an indexed loop.
        for i in 0..n {
            let (d_sq_i, c_i) = cl_scaling_pair(self[i], gradient[i], lower[i], upper[i]);
            d_sq[i] = d_sq_i;
            c_diag[i] = c_i;
        }
    }

    fn max_feasible_step(&self, step: &Self, lower: &Self, upper: &Self) -> f64 {
        let n = self.nrows();
        assert_eq!(n, step.nrows(), "max_feasible_step: step shape mismatch");
        assert_eq!(n, lower.nrows(), "max_feasible_step: lower shape mismatch");
        assert_eq!(n, upper.nrows(), "max_feasible_step: upper shape mismatch");
        let mut tau = f64::INFINITY;
        for i in 0..n {
            let t = max_feasible_step_component(self[i], step[i], lower[i], upper[i]);
            if t < tau {
                tau = t;
            }
        }
        tau
    }

    fn cl_kkt_inf_norm(&self, d_sq: &Self) -> f64 {
        assert_eq!(
            self.nrows(),
            d_sq.nrows(),
            "cl_kkt_inf_norm: shape mismatch"
        );
        self.iter()
            .zip(d_sq.iter())
            .map(|(&v, &d)| v.abs() / d)
            .fold(0.0, f64::max)
    }

    fn weighted_norm_squared(&self, weights: &Self) -> f64 {
        assert_eq!(
            self.nrows(),
            weights.nrows(),
            "weighted_norm_squared: shape mismatch"
        );
        self.iter()
            .zip(weights.iter())
            .map(|(&v, &w)| v * v * w)
            .sum()
    }

    fn project_strictly_inside(&mut self, lower: &Self, upper: &Self, rstep: f64) {
        let n = self.nrows();
        assert_eq!(
            n,
            lower.nrows(),
            "project_strictly_inside: lower shape mismatch"
        );
        assert_eq!(
            n,
            upper.nrows(),
            "project_strictly_inside: upper shape mismatch"
        );
        for i in 0..n {
            self[i] = project_strictly_inside_component(self[i], lower[i], upper[i], rstep);
        }
    }
}

// ----------------------------------------------------------------------
// linalg tier — dense ops on Mat<f64> with V = Col<f64>.
// faer 0.24 has no `*` operator on Mat/Col — go through `matmul` directly.
// ----------------------------------------------------------------------

impl MatVec<Col<f64>> for Mat<f64> {
    fn matvec(&self, x: &Col<f64>) -> Col<f64> {
        assert_eq!(
            self.ncols(),
            x.nrows(),
            "matvec: A.ncols ({}) != x.nrows ({})",
            self.ncols(),
            x.nrows()
        );
        let mut y = Col::<f64>::zeros(self.nrows());
        matmul(
            y.as_mut(),
            Accum::Replace,
            self.as_ref(),
            x.as_ref(),
            1.0,
            Par::Seq,
        );
        y
    }
}

impl MatTransposeVec<Col<f64>> for Mat<f64> {
    fn mat_transpose_vec(&self, x: &Col<f64>) -> Col<f64> {
        assert_eq!(
            self.nrows(),
            x.nrows(),
            "mat_transpose_vec: A.nrows ({}) != x.nrows ({})",
            self.nrows(),
            x.nrows()
        );
        let mut y = Col::<f64>::zeros(self.ncols());
        matmul(
            y.as_mut(),
            Accum::Replace,
            self.transpose(),
            x.as_ref(),
            1.0,
            Par::Seq,
        );
        y
    }
}

impl GramMatrix for Mat<f64> {
    fn gram(&self) -> Self {
        let n = self.ncols();
        let mut g = Mat::<f64>::zeros(n, n);
        matmul(
            g.as_mut(),
            Accum::Replace,
            self.transpose(),
            self.as_ref(),
            1.0,
            Par::Seq,
        );
        g
    }
}

impl MaxDiagonal for Mat<f64> {
    fn max_diagonal(&self) -> f64 {
        assert_eq!(
            self.nrows(),
            self.ncols(),
            "max_diagonal: matrix must be square, got {}x{}",
            self.nrows(),
            self.ncols()
        );
        (0..self.nrows())
            .map(|i| self[(i, i)])
            .fold(f64::NEG_INFINITY, f64::max)
    }
}

impl AddDiagonalInPlace for Mat<f64> {
    fn add_diagonal_in_place(&mut self, scalar: f64) {
        assert_eq!(
            self.nrows(),
            self.ncols(),
            "add_diagonal_in_place: matrix must be square, got {}x{}",
            self.nrows(),
            self.ncols()
        );
        for i in 0..self.nrows() {
            self[(i, i)] += scalar;
        }
    }
}

impl AddDiagonalVectorInPlace<Col<f64>> for Mat<f64> {
    fn add_diagonal_vector_in_place(&mut self, diag: &Col<f64>) {
        assert_eq!(
            self.nrows(),
            self.ncols(),
            "add_diagonal_vector_in_place: matrix must be square, got {}x{}",
            self.nrows(),
            self.ncols()
        );
        assert_eq!(
            self.nrows(),
            diag.nrows(),
            "add_diagonal_vector_in_place: matrix is {}x{} but diag has length {}",
            self.nrows(),
            self.ncols(),
            diag.nrows()
        );
        for i in 0..self.nrows() {
            self[(i, i)] += diag[i];
        }
    }
}

impl LinearSolveSpd<Col<f64>> for Mat<f64> {
    fn solve_spd(&self, b: &Col<f64>) -> Result<Col<f64>, LinearSolveError> {
        assert_eq!(
            self.nrows(),
            self.ncols(),
            "solve_spd: matrix must be square, got {}x{}",
            self.nrows(),
            self.ncols()
        );
        assert_eq!(
            self.nrows(),
            b.nrows(),
            "solve_spd: A.nrows ({}) != b.nrows ({})",
            self.nrows(),
            b.nrows()
        );
        let llt = Llt::new(self.as_ref(), Side::Lower)
            .map_err(|_| LinearSolveError::NotPositiveDefinite)?;
        let mut x = b.clone();
        llt.solve_in_place(&mut x);
        Ok(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    fn mat2(row0: [f64; 2], row1: [f64; 2]) -> Mat<f64> {
        let rows = [row0, row1];
        Mat::from_fn(2, 2, |i, j| rows[i][j])
    }

    #[test]
    fn matvec_known_values() {
        let a = mat2([1.0, 2.0], [3.0, 4.0]);
        let x = Col::<f64>::from_fn(2, |i| [5.0, 6.0][i]);
        let y = a.matvec(&x);
        assert_eq!(y.nrows(), 2);
        assert!(approx_eq(y[0], 17.0, 1e-12));
        assert!(approx_eq(y[1], 39.0, 1e-12));
    }

    #[test]
    fn mat_transpose_vec_known_values() {
        let a = mat2([1.0, 2.0], [3.0, 4.0]);
        let x = Col::<f64>::from_fn(2, |i| [5.0, 6.0][i]);
        let y = a.mat_transpose_vec(&x);
        assert_eq!(y.nrows(), 2);
        // Aᵀ x = [1·5 + 3·6, 2·5 + 4·6] = [23, 34]
        assert!(approx_eq(y[0], 23.0, 1e-12));
        assert!(approx_eq(y[1], 34.0, 1e-12));
    }

    #[test]
    fn gram_known_values() {
        let a = mat2([1.0, 2.0], [3.0, 4.0]);
        let g = a.gram();
        // AᵀA = [[10, 14], [14, 20]]
        assert_eq!(g.nrows(), 2);
        assert_eq!(g.ncols(), 2);
        assert!(approx_eq(g[(0, 0)], 10.0, 1e-12));
        assert!(approx_eq(g[(0, 1)], 14.0, 1e-12));
        assert!(approx_eq(g[(1, 0)], 14.0, 1e-12));
        assert!(approx_eq(g[(1, 1)], 20.0, 1e-12));
    }

    #[test]
    fn solve_spd_happy_path() {
        let a = mat2([4.0, 1.0], [1.0, 3.0]);
        let b = Col::<f64>::from_fn(2, |i| [1.0, 2.0][i]);
        let x = a.solve_spd(&b).expect("SPD system must solve");
        // Same hand-computed answer as the nalgebra test: x = [1/11, 7/11].
        assert!(approx_eq(x[0], 1.0 / 11.0, 1e-12));
        assert!(approx_eq(x[1], 7.0 / 11.0, 1e-12));
    }

    #[test]
    fn solve_spd_indefinite_returns_error() {
        let a = mat2([1.0, 2.0], [2.0, 1.0]);
        let b = Col::<f64>::from_fn(2, |i| [1.0, 1.0][i]);
        let err = a.solve_spd(&b).expect_err("indefinite must fail");
        assert_eq!(err, LinearSolveError::NotPositiveDefinite);
    }

    #[test]
    fn gram_of_rank_deficient_is_singular() {
        let a = mat2([1.0, 2.0], [2.0, 4.0]);
        let g = a.gram();
        let b = Col::<f64>::from_fn(2, |i| [1.0, 1.0][i]);
        let err = g.solve_spd(&b).expect_err("rank-deficient gram must fail");
        assert_eq!(err, LinearSolveError::NotPositiveDefinite);
    }

    #[test]
    fn add_diagonal_in_place_adds_to_diagonal_only() {
        let mut a = Mat::<f64>::from_fn(3, 3, |i, j| (i * 3 + j + 1) as f64);
        a.add_diagonal_in_place(0.5);
        // Diagonal: 1+0.5=1.5, 5+0.5=5.5, 9+0.5=9.5; off-diagonal untouched.
        assert!(approx_eq(a[(0, 0)], 1.5, 1e-12));
        assert!(approx_eq(a[(1, 1)], 5.5, 1e-12));
        assert!(approx_eq(a[(2, 2)], 9.5, 1e-12));
        assert!(approx_eq(a[(0, 1)], 2.0, 1e-12));
        assert!(approx_eq(a[(1, 0)], 4.0, 1e-12));
        assert!(approx_eq(a[(2, 1)], 8.0, 1e-12));
    }

    #[test]
    fn add_diagonal_regularizes_singular_gram() {
        let a = mat2([1.0, 2.0], [2.0, 4.0]);
        let mut g = a.gram();
        let b = Col::<f64>::from_fn(2, |i| [1.0, 1.0][i]);
        assert!(g.clone().solve_spd(&b).is_err());
        g.add_diagonal_in_place(1e-3);
        let x = g.solve_spd(&b).expect("damped gram must be SPD");
        assert_eq!(x.nrows(), 2);
    }

    #[test]
    fn add_diagonal_vector_in_place_adds_per_index() {
        let mut a = Mat::<f64>::from_fn(3, 3, |i, j| (i * 3 + j + 1) as f64);
        a.add_diagonal_vector_in_place(&Col::<f64>::from_fn(3, |i| [10.0, 100.0, 1000.0][i]));
        // Diagonal: 1+10=11, 5+100=105, 9+1000=1009; off-diagonal untouched.
        assert!(approx_eq(a[(0, 0)], 11.0, 1e-12));
        assert!(approx_eq(a[(1, 1)], 105.0, 1e-12));
        assert!(approx_eq(a[(2, 2)], 1009.0, 1e-12));
        assert!(approx_eq(a[(0, 1)], 2.0, 1e-12));
        assert!(approx_eq(a[(2, 1)], 8.0, 1e-12));
    }
}
