//! Hand-rolled dense `f64` matrix for the `Vec<f64>` backend.
//!
//! basin's default param backend is `Vec<f64>` (no external crate). The
//! matrix-capable backends (nalgebra, faer) bring their own dense matrix
//! types, but `Vec<f64>` had none — so the linear-constraint solvers
//! ([`BarrierMethod`](crate::solver::BarrierMethod),
//! [`AugmentedLagrangianMethod`](crate::solver::AugmentedLagrangianMethod)),
//! which only need `A x` and `Aᵀ v`, were a compile-time error on the default
//! backend (tenet 5).
//!
//! [`DenseMatrix`] closes that gap with exactly the two matvec ops and nothing
//! more: no Cholesky / QR / eigen ([`LinearSolveSpd`](super::LinearSolveSpd),
//! [`GramMatrix`](super::GramMatrix), [`SymmetricEigen`](super::SymmetricEigen),
//! …) — those stay nalgebra/faer-only, so LA-heavy solvers (Newton,
//! Gauss-Newton, Levenberg-Marquardt, CMA-ES) remain a compile-time error on
//! `Vec<f64>` by design.

use super::{MatTransposeVec, MatVec};

/// Row-major dense `f64` matrix — the matrix companion to `Vec<f64>` as the
/// param vector.
///
/// Storage is row-major (`data[i * cols + j] = A[i, j]`), the natural layout
/// for a linear-constraint matrix where each row is one constraint
/// `aᵢᵀ x ≤ bᵢ`. This is also what makes [`from_row_slice`](Self::from_row_slice)
/// a transparent mirror of `nalgebra::DMatrix::from_row_slice`.
///
/// The type implements only [`MatVec`] (`A x`) and [`MatTransposeVec`]
/// (`Aᵀ v`); see the module docs for why the factorization ops are
/// deliberately absent.
#[derive(Clone, Debug, PartialEq)]
pub struct DenseMatrix {
    /// Row-major entries: `data[i * cols + j] = A[i, j]`.
    data: Vec<f64>,
    rows: usize,
    cols: usize,
}

impl DenseMatrix {
    /// Build a `rows × cols` matrix from a **row-major** slice (`data[i * cols
    /// + j] = A[i, j]`). Mirrors `nalgebra::DMatrix::from_row_slice`.
    ///
    /// # Panics
    ///
    /// Panics if `data.len() != rows * cols`.
    pub fn from_row_slice(rows: usize, cols: usize, data: &[f64]) -> Self {
        assert_eq!(
            data.len(),
            rows * cols,
            "DenseMatrix::from_row_slice: expected {} entries for a {}×{} matrix, got {}",
            rows * cols,
            rows,
            cols,
            data.len()
        );
        Self {
            data: data.to_vec(),
            rows,
            cols,
        }
    }

    /// Build a `rows × cols` matrix from a per-entry closure `(i, j) -> A[i,
    /// j]`. Mirrors `faer::Mat::from_fn`. The closure is called once per entry
    /// in row-major order.
    pub fn from_fn<F: FnMut(usize, usize) -> f64>(rows: usize, cols: usize, mut f: F) -> Self {
        let mut data = Vec::with_capacity(rows * cols);
        for i in 0..rows {
            for j in 0..cols {
                data.push(f(i, j));
            }
        }
        Self { data, rows, cols }
    }

    /// Number of rows.
    pub fn nrows(&self) -> usize {
        self.rows
    }

    /// Number of columns.
    pub fn ncols(&self) -> usize {
        self.cols
    }

    /// Read entry `A[i, j]`.
    ///
    /// # Panics
    ///
    /// Panics if `i >= nrows()` or `j >= ncols()`.
    pub fn get(&self, i: usize, j: usize) -> f64 {
        assert!(
            i < self.rows && j < self.cols,
            "DenseMatrix::get: index ({i}, {j}) out of bounds for a {}×{} matrix",
            self.rows,
            self.cols
        );
        self.data[i * self.cols + j]
    }
}

impl MatVec<Vec<f64>> for DenseMatrix {
    fn matvec(&self, x: &Vec<f64>) -> Vec<f64> {
        assert_eq!(
            x.len(),
            self.cols,
            "matvec: x has length {} but the matrix has {} columns",
            x.len(),
            self.cols
        );
        let mut y = vec![0.0; self.rows];
        for (i, yi) in y.iter_mut().enumerate() {
            let row = &self.data[i * self.cols..(i + 1) * self.cols];
            *yi = row.iter().zip(x.iter()).map(|(a, xj)| a * xj).sum();
        }
        y
    }
}

impl MatTransposeVec<Vec<f64>> for DenseMatrix {
    fn mat_transpose_vec(&self, x: &Vec<f64>) -> Vec<f64> {
        assert_eq!(
            x.len(),
            self.rows,
            "mat_transpose_vec: x has length {} but the matrix has {} rows",
            x.len(),
            self.rows
        );
        let mut y = vec![0.0; self.cols];
        for (i, &xi) in x.iter().enumerate() {
            let row = &self.data[i * self.cols..(i + 1) * self.cols];
            for (yj, a) in y.iter_mut().zip(row.iter()) {
                *yj += a * xi;
            }
        }
        y
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A non-square `2×3` matrix exercises both ops with distinct input and
    /// output lengths and a non-trivial transpose.
    ///
    /// ```text
    /// A = [1 2 3]   x = (1, 1, 1)ᵀ   ⇒ A x = (6, 15)ᵀ
    ///     [4 5 6]
    /// ```
    fn fixture() -> DenseMatrix {
        DenseMatrix::from_row_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    }

    #[test]
    fn shape_and_entry_access() {
        let a = fixture();
        assert_eq!(a.nrows(), 2);
        assert_eq!(a.ncols(), 3);
        assert_eq!(a.get(0, 2), 3.0);
        assert_eq!(a.get(1, 0), 4.0);
    }

    #[test]
    fn from_fn_matches_from_row_slice() {
        let by_fn = DenseMatrix::from_fn(2, 3, |i, j| (i * 3 + j + 1) as f64);
        assert_eq!(by_fn, fixture());
    }

    #[test]
    fn matvec_computes_a_times_x() {
        let a = fixture();
        let y = a.matvec(&vec![1.0, 1.0, 1.0]);
        assert_eq!(y, vec![6.0, 15.0]);
    }

    #[test]
    fn mat_transpose_vec_computes_a_transpose_times_x() {
        let a = fixture();
        // Aᵀ v with v = (1, 1)ᵀ sums the two rows column-wise: (5, 7, 9).
        let y = a.mat_transpose_vec(&vec![1.0, 1.0]);
        assert_eq!(y, vec![5.0, 7.0, 9.0]);
    }

    /// The two ops must agree on the implicit transpose: `(A x)·v = x·(Aᵀ v)`.
    #[test]
    fn matvec_and_transpose_are_consistent() {
        let a = fixture();
        let x = vec![0.5, -1.0, 2.0];
        let v = vec![3.0, -2.0];

        let ax = a.matvec(&x);
        let atv = a.mat_transpose_vec(&v);

        let lhs: f64 = ax.iter().zip(&v).map(|(p, q)| p * q).sum();
        let rhs: f64 = x.iter().zip(&atv).map(|(p, q)| p * q).sum();
        assert!((lhs - rhs).abs() < 1e-12, "lhs={lhs}, rhs={rhs}");
    }

    #[test]
    #[should_panic(expected = "from_row_slice")]
    fn from_row_slice_rejects_wrong_length() {
        let _ = DenseMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0]);
    }

    #[test]
    #[should_panic(expected = "matvec")]
    fn matvec_rejects_length_mismatch() {
        let a = fixture();
        let _ = a.matvec(&vec![1.0, 1.0]); // needs length 3 (ncols)
    }

    #[test]
    #[should_panic(expected = "mat_transpose_vec")]
    fn mat_transpose_vec_rejects_length_mismatch() {
        let a = fixture();
        let _ = a.mat_transpose_vec(&vec![1.0, 1.0, 1.0]); // needs length 2 (nrows)
    }
}
