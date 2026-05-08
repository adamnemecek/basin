use nalgebra::{DMatrix, DVector, Dim, Matrix, Storage, StorageMut};

use super::cl_scaling::{
    cl_scaling_pair, max_feasible_step_component, project_strictly_inside_component,
    BoxAffineScaling,
};
use super::linalg::{
    AddDiagonalInPlace, AddDiagonalVectorInPlace, GramMatrix, LinearSolveError, LinearSolveSpd,
    MatTransposeVec, MatVec, MaxDiagonal,
};
use super::{ClampInPlace, Dot, NegInPlace, NormInfinity, NormSquared, ScaledAdd};

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

impl<R, C, S> Dot for Matrix<f64, R, C, S>
where
    R: Dim,
    C: Dim,
    S: Storage<f64, R, C>,
{
    fn dot(&self, other: &Self) -> f64 {
        assert_eq!(self.shape(), other.shape(), "dot: shape mismatch");
        self.iter().zip(other.iter()).map(|(a, b)| a * b).sum()
    }
}

impl<R, C, S> NegInPlace for Matrix<f64, R, C, S>
where
    R: Dim,
    C: Dim,
    S: StorageMut<f64, R, C>,
{
    fn neg_in_place(&mut self) {
        self.apply(|x| *x = -*x);
    }
}

impl<R, C, S> ClampInPlace for Matrix<f64, R, C, S>
where
    R: Dim,
    C: Dim,
    S: StorageMut<f64, R, C>,
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
        // `iter_mut` and `iter` both traverse in column-major order, so
        // zipping is consistent across self / lower / upper.
        for ((x, &lo), &hi) in self.iter_mut().zip(lower.iter()).zip(upper.iter()) {
            *x = x.clamp(lo, hi);
        }
    }
}

impl<R, C, S> BoxAffineScaling for Matrix<f64, R, C, S>
where
    R: Dim,
    C: Dim,
    S: StorageMut<f64, R, C>,
{
    fn compute_cl_scaling(
        &self,
        gradient: &Self,
        lower: &Self,
        upper: &Self,
        d_sq: &mut Self,
        c_diag: &mut Self,
    ) {
        let shape = self.shape();
        assert_eq!(
            shape,
            gradient.shape(),
            "compute_cl_scaling: gradient shape mismatch"
        );
        assert_eq!(
            shape,
            lower.shape(),
            "compute_cl_scaling: lower shape mismatch"
        );
        assert_eq!(
            shape,
            upper.shape(),
            "compute_cl_scaling: upper shape mismatch"
        );
        assert_eq!(
            shape,
            d_sq.shape(),
            "compute_cl_scaling: d_sq shape mismatch"
        );
        assert_eq!(
            shape,
            c_diag.shape(),
            "compute_cl_scaling: c_diag shape mismatch"
        );
        // Column-major iteration order matches the ClampInPlace impl;
        // consistent across all six operands.
        for (((((&x, &g), &l), &u), d), c) in self
            .iter()
            .zip(gradient.iter())
            .zip(lower.iter())
            .zip(upper.iter())
            .zip(d_sq.iter_mut())
            .zip(c_diag.iter_mut())
        {
            let (d_sq_i, c_i) = cl_scaling_pair(x, g, l, u);
            *d = d_sq_i;
            *c = c_i;
        }
    }

    fn max_feasible_step(&self, step: &Self, lower: &Self, upper: &Self) -> f64 {
        let shape = self.shape();
        assert_eq!(
            shape,
            step.shape(),
            "max_feasible_step: step shape mismatch"
        );
        assert_eq!(
            shape,
            lower.shape(),
            "max_feasible_step: lower shape mismatch"
        );
        assert_eq!(
            shape,
            upper.shape(),
            "max_feasible_step: upper shape mismatch"
        );
        let mut tau = f64::INFINITY;
        for (((&x, &s), &l), &u) in self
            .iter()
            .zip(step.iter())
            .zip(lower.iter())
            .zip(upper.iter())
        {
            let t = max_feasible_step_component(x, s, l, u);
            if t < tau {
                tau = t;
            }
        }
        tau
    }

    fn cl_kkt_inf_norm(&self, d_sq: &Self) -> f64 {
        assert_eq!(
            self.shape(),
            d_sq.shape(),
            "cl_kkt_inf_norm: shape mismatch"
        );
        self.iter()
            .zip(d_sq.iter())
            .map(|(&v, &d)| v.abs() / d)
            .fold(0.0, f64::max)
    }

    fn weighted_norm_squared(&self, weights: &Self) -> f64 {
        assert_eq!(
            self.shape(),
            weights.shape(),
            "weighted_norm_squared: shape mismatch"
        );
        self.iter()
            .zip(weights.iter())
            .map(|(&v, &w)| v * v * w)
            .sum()
    }

    fn project_strictly_inside(&mut self, lower: &Self, upper: &Self, rstep: f64) {
        let shape = self.shape();
        assert_eq!(
            shape,
            lower.shape(),
            "project_strictly_inside: lower shape mismatch"
        );
        assert_eq!(
            shape,
            upper.shape(),
            "project_strictly_inside: upper shape mismatch"
        );
        for ((x, &l), &u) in self.iter_mut().zip(lower.iter()).zip(upper.iter()) {
            *x = project_strictly_inside_component(*x, l, u, rstep);
        }
    }
}

// ----------------------------------------------------------------------
// linalg tier — dense ops on DMatrix<f64> with V = DVector<f64>.
// Per tenet 5, this is dense-only; sparse comes in S2b.
// ----------------------------------------------------------------------

impl MatVec<DVector<f64>> for DMatrix<f64> {
    fn matvec(&self, x: &DVector<f64>) -> DVector<f64> {
        assert_eq!(
            self.ncols(),
            x.len(),
            "matvec: A.ncols ({}) != x.len ({})",
            self.ncols(),
            x.len()
        );
        self * x
    }
}

impl MatTransposeVec<DVector<f64>> for DMatrix<f64> {
    fn mat_transpose_vec(&self, x: &DVector<f64>) -> DVector<f64> {
        assert_eq!(
            self.nrows(),
            x.len(),
            "mat_transpose_vec: A.nrows ({}) != x.len ({})",
            self.nrows(),
            x.len()
        );
        self.tr_mul(x)
    }
}

impl GramMatrix for DMatrix<f64> {
    fn gram(&self) -> Self {
        // tr_mul(self) computes Aᵀ A in one pass without an explicit transpose.
        self.tr_mul(self)
    }
}

impl MaxDiagonal for DMatrix<f64> {
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

impl AddDiagonalInPlace for DMatrix<f64> {
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

impl AddDiagonalVectorInPlace<DVector<f64>> for DMatrix<f64> {
    fn add_diagonal_vector_in_place(&mut self, diag: &DVector<f64>) {
        assert_eq!(
            self.nrows(),
            self.ncols(),
            "add_diagonal_vector_in_place: matrix must be square, got {}x{}",
            self.nrows(),
            self.ncols()
        );
        assert_eq!(
            self.nrows(),
            diag.len(),
            "add_diagonal_vector_in_place: matrix is {}x{} but diag has length {}",
            self.nrows(),
            self.ncols(),
            diag.len()
        );
        for i in 0..self.nrows() {
            self[(i, i)] += diag[i];
        }
    }
}

impl LinearSolveSpd<DVector<f64>> for DMatrix<f64> {
    fn solve_spd(&self, b: &DVector<f64>) -> Result<DVector<f64>, LinearSolveError> {
        assert_eq!(
            self.nrows(),
            self.ncols(),
            "solve_spd: matrix must be square, got {}x{}",
            self.nrows(),
            self.ncols()
        );
        assert_eq!(
            self.nrows(),
            b.len(),
            "solve_spd: A.nrows ({}) != b.len ({})",
            self.nrows(),
            b.len()
        );
        // nalgebra's `cholesky` consumes the matrix — clone is unavoidable
        // without a separate factorize/solve split.
        self.clone()
            .cholesky()
            .ok_or(LinearSolveError::NotPositiveDefinite)
            .map(|chol| chol.solve(b))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn matvec_known_values() {
        let a = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let x = DVector::from_vec(vec![5.0, 6.0]);
        let y = a.matvec(&x);
        assert_eq!(y.len(), 2);
        assert!(approx_eq(y[0], 17.0, 1e-12));
        assert!(approx_eq(y[1], 39.0, 1e-12));
    }

    #[test]
    fn mat_transpose_vec_known_values() {
        let a = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let x = DVector::from_vec(vec![5.0, 6.0]);
        let y = a.mat_transpose_vec(&x);
        assert_eq!(y.len(), 2);
        // Aᵀ x = [1·5 + 3·6, 2·5 + 4·6] = [23, 34]
        assert!(approx_eq(y[0], 23.0, 1e-12));
        assert!(approx_eq(y[1], 34.0, 1e-12));
    }

    #[test]
    fn gram_known_values() {
        let a = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let g = a.gram();
        // AᵀA = [[1·1+3·3, 1·2+3·4], [2·1+4·3, 2·2+4·4]] = [[10, 14], [14, 20]]
        assert_eq!(g.shape(), (2, 2));
        assert!(approx_eq(g[(0, 0)], 10.0, 1e-12));
        assert!(approx_eq(g[(0, 1)], 14.0, 1e-12));
        assert!(approx_eq(g[(1, 0)], 14.0, 1e-12));
        assert!(approx_eq(g[(1, 1)], 20.0, 1e-12));
    }

    #[test]
    fn solve_spd_happy_path() {
        // A = [[4, 1], [1, 3]], b = [1, 2].
        // det = 11, x = (1/11) [3·1 − 1·2, −1·1 + 4·2] = [1/11, 7/11].
        let a = DMatrix::from_row_slice(2, 2, &[4.0, 1.0, 1.0, 3.0]);
        let b = DVector::from_vec(vec![1.0, 2.0]);
        let x = a.solve_spd(&b).expect("SPD system must solve");
        assert!(approx_eq(x[0], 1.0 / 11.0, 1e-12));
        assert!(approx_eq(x[1], 7.0 / 11.0, 1e-12));
    }

    #[test]
    fn solve_spd_indefinite_returns_error() {
        // A = [[1, 2], [2, 1]] is symmetric but indefinite (det = −3).
        let a = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 2.0, 1.0]);
        let b = DVector::from_vec(vec![1.0, 1.0]);
        let err = a.solve_spd(&b).expect_err("indefinite must fail");
        assert_eq!(err, LinearSolveError::NotPositiveDefinite);
    }

    #[test]
    fn gram_of_rank_deficient_is_singular() {
        // Rank-1 matrix → AᵀA is rank-1, singular, fails Cholesky.
        let a = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 2.0, 4.0]);
        let g = a.gram();
        let b = DVector::from_vec(vec![1.0, 1.0]);
        let err = g.solve_spd(&b).expect_err("rank-deficient gram must fail");
        assert_eq!(err, LinearSolveError::NotPositiveDefinite);
    }

    #[test]
    fn add_diagonal_in_place_adds_to_diagonal_only() {
        let mut a = DMatrix::from_row_slice(3, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        a.add_diagonal_in_place(0.5);
        // Diagonal: 1.5, 5.5, 9.5; off-diagonal untouched.
        assert!(approx_eq(a[(0, 0)], 1.5, 1e-12));
        assert!(approx_eq(a[(1, 1)], 5.5, 1e-12));
        assert!(approx_eq(a[(2, 2)], 9.5, 1e-12));
        assert!(approx_eq(a[(0, 1)], 2.0, 1e-12));
        assert!(approx_eq(a[(1, 0)], 4.0, 1e-12));
        assert!(approx_eq(a[(2, 1)], 8.0, 1e-12));
    }

    #[test]
    fn add_diagonal_vector_in_place_adds_per_index() {
        let mut a = DMatrix::from_row_slice(3, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        a.add_diagonal_vector_in_place(&DVector::from_vec(vec![10.0, 100.0, 1000.0]));
        // Diagonal: 11, 105, 1009; off-diagonal untouched.
        assert!(approx_eq(a[(0, 0)], 11.0, 1e-12));
        assert!(approx_eq(a[(1, 1)], 105.0, 1e-12));
        assert!(approx_eq(a[(2, 2)], 1009.0, 1e-12));
        assert!(approx_eq(a[(0, 1)], 2.0, 1e-12));
        assert!(approx_eq(a[(2, 1)], 8.0, 1e-12));
    }

    #[test]
    fn add_diagonal_regularizes_singular_gram() {
        // The "why LM works" property: a rank-deficient Gram becomes
        // SPD once you add μI, and Cholesky succeeds.
        let a = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 2.0, 4.0]);
        let mut g = a.gram();
        assert!(g
            .clone()
            .solve_spd(&DVector::from_vec(vec![1.0, 1.0]))
            .is_err());
        g.add_diagonal_in_place(1e-3);
        let x = g
            .solve_spd(&DVector::from_vec(vec![1.0, 1.0]))
            .expect("damped gram must be SPD");
        assert_eq!(x.len(), 2);
    }
}
