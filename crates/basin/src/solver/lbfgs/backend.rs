//! Backend-agnostic slice views over the parameter vector.
//!
//! L-BFGS-B's inner numerics (cauchy, subsm, formk, compact-form
//! helpers) all operate on `&[f64]` / `&mut [f64]`. To stay generic
//! over the user-chosen parameter backend (`Vec<f64>`, nalgebra
//! `DVector<f64>`, faer `Col<f64>`), the top-level solver views each
//! vector as a contiguous f64 slice at iteration boundaries. The two
//! tiny traits below capture that view; impls are per-backend and
//! `pub(crate)` to keep them off the public API surface.
//!
//! This is the slice-extraction layer noted in the L-BFGS-B port plan
//! (Stage 6 in `~/.claude/plans/i-want-to-add-ticklish-breeze.md`).
//! It only matters at the solver / state boundary — once a slice has
//! been extracted, every cauchy / subsm / formk call already works on
//! `&[f64]`.

/// Read-only view of a parameter vector as a contiguous `&[f64]`.
pub(crate) trait AsFloatSlice {
    /// Borrow the underlying storage as a contiguous slice.
    fn as_float_slice(&self) -> &[f64];
}

/// Mutable companion to [`AsFloatSlice`].
pub(crate) trait AsFloatSliceMut: AsFloatSlice {
    /// Borrow the underlying storage as a contiguous mutable slice.
    fn as_float_slice_mut(&mut self) -> &mut [f64];
}

impl AsFloatSlice for Vec<f64> {
    fn as_float_slice(&self) -> &[f64] {
        self.as_slice()
    }
}
impl AsFloatSliceMut for Vec<f64> {
    fn as_float_slice_mut(&mut self) -> &mut [f64] {
        self.as_mut_slice()
    }
}

#[cfg(feature = "nalgebra")]
impl AsFloatSlice for nalgebra::DVector<f64> {
    fn as_float_slice(&self) -> &[f64] {
        self.as_slice()
    }
}
#[cfg(feature = "nalgebra")]
impl AsFloatSliceMut for nalgebra::DVector<f64> {
    fn as_float_slice_mut(&mut self) -> &mut [f64] {
        self.as_mut_slice()
    }
}

#[cfg(feature = "faer")]
impl AsFloatSlice for faer::Col<f64> {
    fn as_float_slice(&self) -> &[f64] {
        self.try_as_col_major()
            .expect("faer::Col<f64> backing storage must be col-major contiguous")
            .as_slice()
    }
}
#[cfg(feature = "faer")]
impl AsFloatSliceMut for faer::Col<f64> {
    fn as_float_slice_mut(&mut self) -> &mut [f64] {
        self.try_as_col_major_mut()
            .expect("faer::Col<f64> backing storage must be col-major contiguous")
            .as_slice_mut()
    }
}
