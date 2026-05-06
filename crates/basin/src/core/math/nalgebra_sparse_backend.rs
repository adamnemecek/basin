//! Sparse impls of the `linalg` tier for
//! [`nalgebra_sparse::CscMatrix<f64>`] (CSC) over
//! [`nalgebra::DVector<f64>`]. Lands in S2b alongside the dense
//! nalgebra backend.
//!
//! nalgebra-sparse covers four of the five `linalg` traits: SpMV /
//! Aᵀ-SpMV via [`spmm_csc_dense`], Gram via the `&CscMatrix *
//! &CscMatrix` operator overload (composing transpose + spmm under
//! the hood), and SPD solve via
//! [`CscCholesky`](nalgebra_sparse::factorization::CscCholesky).
//! The fifth — `LinearSolveLstsq` — is **deliberately not
//! implemented** here: nalgebra-sparse 0.10 doesn't ship a sparse QR.
//! Reach for the faer-sparse backend if you need least-squares on
//! sparse `J`.

use nalgebra::{DMatrix, DVector};
use nalgebra_sparse::factorization::CscCholesky;
use nalgebra_sparse::ops::serial::spmm_csc_dense;
use nalgebra_sparse::ops::Op;
use nalgebra_sparse::CscMatrix;

use super::linalg::{GramMatrix, LinearSolveError, LinearSolveSpd, MatTransposeVec, MatVec};

impl MatVec<DVector<f64>> for CscMatrix<f64> {
    fn matvec(&self, x: &DVector<f64>) -> DVector<f64> {
        assert_eq!(
            self.ncols(),
            x.len(),
            "matvec: A.ncols ({}) != x.len ({})",
            self.ncols(),
            x.len()
        );
        // The `Mul` impl on `&CscMatrix * &DVector` forwards to
        // spmm_csc_dense and returns an OMatrix<f64, Dyn, U1>, which
        // is exactly DVector<f64>.
        self * x
    }
}

impl MatTransposeVec<DVector<f64>> for CscMatrix<f64> {
    fn mat_transpose_vec(&self, x: &DVector<f64>) -> DVector<f64> {
        assert_eq!(
            self.nrows(),
            x.len(),
            "mat_transpose_vec: A.nrows ({}) != x.len ({})",
            self.nrows(),
            x.len()
        );
        // spmm_csc_dense with Op::Transpose lets us avoid materializing
        // Aᵀ. Output dimension is `ncols(self) × 1`; the helper takes
        // dense RHS as a `DMatrixView`, so we wrap `x` as a 1-column
        // DMatrix.
        let mut y = DMatrix::<f64>::zeros(self.ncols(), 1);
        let x_mat = DMatrix::from_column_slice(x.len(), 1, x.as_slice());
        spmm_csc_dense(0.0, &mut y, 1.0, Op::Transpose(self), Op::NoOp(&x_mat));
        DVector::from_column_slice(y.column(0).as_slice())
    }
}

impl GramMatrix for CscMatrix<f64> {
    fn gram(&self) -> Self {
        // The `&CscMatrix * &CscMatrix` operator overload composes
        // pattern construction + spmm. Aᵀ A → CSC of shape
        // `(ncols, ncols)`; transpose() materializes Aᵀ as CSC.
        &self.transpose() * self
    }
}

impl LinearSolveSpd<DVector<f64>> for CscMatrix<f64> {
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
        // CscCholesky::solve takes a `DMatrixView` and returns a
        // `DMatrix`, so we round-trip the DVector through a 1-column
        // dense matrix. One small allocation per solve; the Cholesky
        // factorization itself dominates cost.
        let chol = CscCholesky::factor(self).map_err(|_| LinearSolveError::NotPositiveDefinite)?;
        let b_mat = DMatrix::from_column_slice(b.len(), 1, b.as_slice());
        let x_mat = chol.solve(&b_mat);
        Ok(DVector::from_column_slice(x_mat.column(0).as_slice()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra_sparse::CooMatrix;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    /// 2×2 dense matrix expressed as CSC via COO triplets.
    fn csc2(row0: [f64; 2], row1: [f64; 2]) -> CscMatrix<f64> {
        let mut coo = CooMatrix::<f64>::new(2, 2);
        coo.push(0, 0, row0[0]);
        coo.push(0, 1, row0[1]);
        coo.push(1, 0, row1[0]);
        coo.push(1, 1, row1[1]);
        CscMatrix::from(&coo)
    }

    #[test]
    fn matvec_known_values() {
        let a = csc2([1.0, 2.0], [3.0, 4.0]);
        let x = DVector::from_vec(vec![5.0, 6.0]);
        let y = a.matvec(&x);
        assert_eq!(y.len(), 2);
        assert!(approx_eq(y[0], 17.0, 1e-12));
        assert!(approx_eq(y[1], 39.0, 1e-12));
    }

    #[test]
    fn mat_transpose_vec_known_values() {
        let a = csc2([1.0, 2.0], [3.0, 4.0]);
        let x = DVector::from_vec(vec![5.0, 6.0]);
        let y = a.mat_transpose_vec(&x);
        assert_eq!(y.len(), 2);
        // Aᵀ x = [1·5 + 3·6, 2·5 + 4·6] = [23, 34]
        assert!(approx_eq(y[0], 23.0, 1e-12));
        assert!(approx_eq(y[1], 34.0, 1e-12));
    }

    #[test]
    fn gram_known_values() {
        let a = csc2([1.0, 2.0], [3.0, 4.0]);
        let g = a.gram();
        // AᵀA = [[10, 14], [14, 20]]
        assert_eq!(g.nrows(), 2);
        assert_eq!(g.ncols(), 2);
        let e0 = DVector::from_vec(vec![1.0, 0.0]);
        let e1 = DVector::from_vec(vec![0.0, 1.0]);
        let col0 = g.matvec(&e0);
        let col1 = g.matvec(&e1);
        assert!(approx_eq(col0[0], 10.0, 1e-12));
        assert!(approx_eq(col0[1], 14.0, 1e-12));
        assert!(approx_eq(col1[0], 14.0, 1e-12));
        assert!(approx_eq(col1[1], 20.0, 1e-12));
    }

    #[test]
    fn solve_spd_happy_path() {
        let a = csc2([4.0, 1.0], [1.0, 3.0]);
        let b = DVector::from_vec(vec![1.0, 2.0]);
        let x = a.solve_spd(&b).expect("SPD system must solve");
        // Same hand-computed answer as dense: x = [1/11, 7/11].
        assert!(approx_eq(x[0], 1.0 / 11.0, 1e-12));
        assert!(approx_eq(x[1], 7.0 / 11.0, 1e-12));
    }

    #[test]
    fn solve_spd_indefinite_returns_error() {
        let a = csc2([1.0, 2.0], [2.0, 1.0]);
        let b = DVector::from_vec(vec![1.0, 1.0]);
        let err = a.solve_spd(&b).expect_err("indefinite must fail");
        assert_eq!(err, LinearSolveError::NotPositiveDefinite);
    }

    #[test]
    fn gram_of_rank_deficient_is_singular() {
        let a = csc2([1.0, 2.0], [2.0, 4.0]);
        let g = a.gram();
        let b = DVector::from_vec(vec![1.0, 1.0]);
        let err = g.solve_spd(&b).expect_err("rank-deficient gram must fail");
        assert_eq!(err, LinearSolveError::NotPositiveDefinite);
    }
}
