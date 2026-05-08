use nalgebra::{DMatrix, DVector, Dim, Matrix, Storage, StorageMut};

use super::linalg::{
    AddDiagonalInPlace, GramMatrix, LinearSolveError, LinearSolveSpd, MatTransposeVec, MatVec,
    MaxDiagonal,
};
use super::{Dot, NegInPlace, NormInfinity, NormSquared, ScaledAdd};

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
