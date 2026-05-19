//! Compact-form helpers for L-BFGS-B.
//!
//! Mirrors the Fortran subroutines in `references/lbfgsb-v3.0/lbfgsb.f`:
//!
//! - [`formt`] — build `T = θ SᵀS + L D⁻¹ Lᵀ` from the Gram blocks
//!   stored on [`LbfgsState`][s], then Cholesky-factorize it in place
//!   so the upper triangle of `wt` holds `J` such that `T = Jᵀ J`
//!   (Fortran `formt`, `lbfgsb.f:2173`).
//! - [`bmv`] — apply the 2k × 2k middle-matrix inverse `M` to a vector,
//!   via the two block-triangular solves (`lbfgsb.f:1106`).
//! - [`cholesky_upper_in_place`], [`solve_upper_tri`],
//!   [`solve_upper_tri_transposed`] — pure-Rust replacements for the
//!   LINPACK `dpofa` / `dtrsl` calls. Row-major storage with stride
//!   `m_capacity` so the leading `col × col` block matches the
//!   Fortran layout of `wt(m, m)` with only the leading `col × col`
//!   live.
//!
//! All routines operate on `&[f64]` / `&mut [f64]`; the surrounding
//! solver is responsible for sourcing those slices from whichever
//! backend [`LbfgsState`][s] is parameterized on.
//!
//! [s]: crate::core::state::LbfgsState

/// Build the upper triangle of `T = θ · SᵀS + L D⁻¹ Lᵀ` from
/// `sy = SᵀY`, `ss = SᵀS`, and `theta`, then Cholesky-factorize so the
/// upper triangle of `wt` holds `J` with `T = Jᵀ J`.
///
/// Storage is row-major with stride `m`; the leading `col × col` block
/// is the live region. Lower-triangle entries of `wt` are left
/// undefined (Fortran does the same — `dpofa` only touches the upper
/// triangle).
///
/// Returns `Ok(())` on success, or `Err(FormtError::NotPositiveDefinite)`
/// matching Fortran's `info = -3`.
pub(crate) fn formt(
    theta: f64,
    sy: &[f64],
    ss: &[f64],
    col: usize,
    m: usize,
    wt: &mut [f64],
) -> Result<(), FormtError> {
    debug_assert!(col <= m, "col must be ≤ m");
    debug_assert!(sy.len() >= m * m && ss.len() >= m * m && wt.len() >= m * m);
    if col == 0 {
        return Ok(());
    }

    // Row 0 of T: T[0, j] = θ · SS[0, j] (the `L D⁻¹ Lᵀ` term is zero
    // here because L's first row has no strict-lower entries).
    for j in 0..col {
        wt[j] = theta * ss[j];
    }
    // Rows i = 1..col, columns j = i..col (upper triangle only).
    for i in 1..col {
        for j in i..col {
            // ddum = Σ_{k=0}^{min(i,j)-1} SY[i,k] · SY[j,k] / SY[k,k]
            let k1 = i.min(j);
            let mut ddum = 0.0;
            for k in 0..k1 {
                ddum += sy[i * m + k] * sy[j * m + k] / sy[k * m + k];
            }
            wt[i * m + j] = ddum + theta * ss[i * m + j];
        }
    }

    cholesky_upper_in_place(wt, col, m)
        .then_some(())
        .ok_or(FormtError::NotPositiveDefinite)
}

/// Reasons [`formt`] can fail. Matches Fortran's `info = -3` from
/// `dpofa` returning a non-PD diagonal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FormtError {
    /// The middle matrix `T = θ SᵀS + L D⁻¹ Lᵀ` failed Cholesky — a
    /// pivot was non-positive. Mirrors Fortran `info = -3`.
    NotPositiveDefinite,
}

/// In-place Cholesky factorization of the upper-triangular form: given
/// an SPD matrix `T` stored in the upper triangle of `t` (row-major,
/// stride `m`, leading `col × col` live), overwrite the upper triangle
/// with `J` such that `T = Jᵀ J` (so `J` is upper triangular).
///
/// Mirrors LINPACK's `dpofa`. Returns `false` if a pivot is
/// non-positive (matches `dpofa`'s `info > 0` exit; Fortran flags it
/// as `info = -3` in `formt`).
pub(crate) fn cholesky_upper_in_place(t: &mut [f64], col: usize, m: usize) -> bool {
    if col == 0 {
        return true;
    }
    for j in 0..col {
        // s = T[j, j] − Σ_{k<j} J[k, j]²
        let mut s = t[j * m + j];
        for k in 0..j {
            let jkj = t[k * m + j];
            s -= jkj * jkj;
        }
        if !s.is_finite() || s <= 0.0 {
            return false;
        }
        let djj = s.sqrt();
        t[j * m + j] = djj;
        // For i > j: J[j, i] = (T[j, i] − Σ_{k<j} J[k, j] · J[k, i]) / J[j, j]
        for i in (j + 1)..col {
            let mut s = t[j * m + i];
            for k in 0..j {
                s -= t[k * m + j] * t[k * m + i];
            }
            t[j * m + i] = s / djj;
        }
    }
    true
}

/// Solve `J x = b` in place on `b`, where `J` is the upper-triangular
/// Cholesky factor stored in the upper triangle of `j_upper`
/// (row-major, stride `m`, leading `col × col` live). Mirrors LINPACK's
/// `dtrsl(..., job=01, ...)`.
pub(crate) fn solve_upper_tri(j_upper: &[f64], col: usize, m: usize, b: &mut [f64]) {
    if col == 0 {
        return;
    }
    // Back-substitution: solve from the bottom row up.
    for i in (0..col).rev() {
        let mut s = b[i];
        for k in (i + 1)..col {
            s -= j_upper[i * m + k] * b[k];
        }
        b[i] = s / j_upper[i * m + i];
    }
}

/// Solve `Jᵀ x = b` in place on `b`. Mirrors LINPACK's `dtrsl(...,
/// job=11, ...)` — `J` is upper triangular, the transposed solve runs
/// top-down.
pub(crate) fn solve_upper_tri_transposed(j_upper: &[f64], col: usize, m: usize, b: &mut [f64]) {
    if col == 0 {
        return;
    }
    for i in 0..col {
        let mut s = b[i];
        for k in 0..i {
            s -= j_upper[k * m + i] * b[k];
        }
        b[i] = s / j_upper[i * m + i];
    }
}

/// Apply the 2k × 2k middle-matrix inverse `M` to a vector. Computes
/// `p = M v` via the structured block-triangular solve in
/// `references/lbfgsb-v3.0/lbfgsb.f:1106` (`bmv`), using:
///
/// - the diagonal `D` of `sy` (`D_ii = sᵢ · yᵢ`);
/// - the strict lower triangle `L` of `sy` (`L_ij = sᵢ · yⱼ`, `i > j`);
/// - the Cholesky factor `J` of `T = θ SᵀS + L D⁻¹ Lᵀ` in `wt`.
///
/// `v` and `p` must be at least `2*col` long; only the leading `2*col`
/// entries are read / written. Returns `Err(BmvError::SingularJ)` if
/// any pivot of `J` is zero (matches Fortran `info ≠ 0`).
///
/// `M_inv = [ -D       Lᵀ ]` and the structured factorization in
/// `[ L    θ SᵀS ]` Fortran `bmv` gives the two-stage solve below.
pub(crate) fn bmv(
    sy: &[f64],
    wt: &[f64],
    col: usize,
    m: usize,
    v: &[f64],
    p: &mut [f64],
) -> Result<(), BmvError> {
    if col == 0 {
        return Ok(());
    }
    debug_assert!(v.len() >= 2 * col && p.len() >= 2 * col);

    // Check J's diagonal upfront — the two triangular solves divide
    // by `wt[i*m+i]`, so a zero/NaN/Inf pivot would silently produce
    // NaNs instead of surfacing the singularity. Mirrors Fortran's
    // `info ≠ 0` exit from `dtrsl`.
    for i in 0..col {
        let d = wt[i * m + i];
        if d == 0.0 || !d.is_finite() {
            return Err(BmvError::SingularJ);
        }
    }

    // PART I — solve [  D^{1/2}      0  ] [ p1 ] = [ v1 ]
    //                [ −L D^{−1/2}   J  ] [ p2 ]   [ v2 ].
    //
    // Stage 1a: build `rhs = v2 + L D⁻¹ v1` into `p[col..]` then
    // overwrite with `(Jᵀ)⁻¹ · rhs`.  Fortran uses
    // `dtrsl(wt, ..., job=11)`, which on `wt` storing the upper-tri
    // Cholesky factor `J` solves `Jᵀ x = b`.
    p[col] = v[col];
    for i in 1..col {
        let mut sum = 0.0;
        for k in 0..i {
            sum += sy[i * m + k] * v[k] / sy[k * m + k];
        }
        p[col + i] = v[col + i] + sum;
    }
    solve_upper_tri_transposed(wt, col, m, &mut p[col..col + col]);

    // Stage 1b: p1 = v1 / D^{1/2}.
    for i in 0..col {
        p[i] = v[i] / sy[i * m + i].sqrt();
    }

    // PART II — solve [ −D^{1/2}   D^{−1/2} Lᵀ ] [ p1 ] = [ p1 ]
    //                 [    0          Jᵀ       ] [ p2 ]   [ p2 ].
    //
    // Stage 2a: solve `J p2 = p2`. Fortran uses
    // `dtrsl(wt, ..., job=01)` — the non-transposed solve on an
    // upper-tri factor.
    solve_upper_tri(wt, col, m, &mut p[col..col + col]);

    // Stage 2b: p1 = −D⁻¹/² p1 + D⁻¹ Lᵀ p2
    //         = −p1 / sqrt(D) + Σ_{k>i} sy[k,i] · p[col+k] / D[i,i].
    for i in 0..col {
        p[i] = -p[i] / sy[i * m + i].sqrt();
    }
    for i in 0..col {
        let mut sum = 0.0;
        for k in (i + 1)..col {
            sum += sy[k * m + i] * p[col + k] / sy[i * m + i];
        }
        p[i] += sum;
    }
    Ok(())
}

/// Reasons [`bmv`] can fail. Matches Fortran's `info ≠ 0` exit when
/// `dtrsl` encounters a zero pivot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BmvError {
    /// The Cholesky factor `J` of `T` has a zero pivot — the
    /// triangular solve cannot proceed. Mirrors Fortran `info ≠ 0`
    /// from `dtrsl`.
    SingularJ,
}

#[cfg(test)]
// Explicit `i * m + j` indexing (including `0 * m + 0`) mirrors the
// Fortran source's 2-D layout — load-bearing for readability when
// cross-checking against `lbfgsb.f`.
#[allow(clippy::identity_op, clippy::erasing_op)]
mod tests {
    use super::*;

    #[test]
    fn cholesky_round_trip_2x2() {
        // T = [[4, 6], [6, 13]] is SPD with J = [[2, 3], [0, 2]].
        let m = 3;
        let mut t = vec![0.0; m * m];
        t[0 * m + 0] = 4.0;
        t[0 * m + 1] = 6.0;
        t[1 * m + 1] = 13.0;
        let ok = cholesky_upper_in_place(&mut t, 2, m);
        assert!(ok);
        assert!((t[0 * m + 0] - 2.0).abs() < 1e-12);
        assert!((t[0 * m + 1] - 3.0).abs() < 1e-12);
        assert!((t[1 * m + 1] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn cholesky_rejects_non_pd() {
        // T = [[1, 2], [2, 1]] has det −3 — indefinite.
        let m = 2;
        let mut t = vec![0.0; m * m];
        t[0 * m + 0] = 1.0;
        t[0 * m + 1] = 2.0;
        t[1 * m + 1] = 1.0;
        assert!(!cholesky_upper_in_place(&mut t, 2, m));
    }

    #[test]
    fn solve_upper_tri_inverts_apply() {
        // J = [[2, 3], [0, 2]]; solve J x = b for b = (5, 4).
        // Expected x = (1, 2): 2·1 + 3·2 = 8? wait. Let me re-do.
        // J x = b → 2 x1 + 3 x2 = 5 and 2 x2 = 4. So x2 = 2, x1 = (5
        // − 6)/2 = −0.5.
        let m = 2;
        let mut j_upper = vec![0.0; m * m];
        j_upper[0 * m + 0] = 2.0;
        j_upper[0 * m + 1] = 3.0;
        j_upper[1 * m + 1] = 2.0;
        let mut b = vec![5.0, 4.0];
        solve_upper_tri(&j_upper, 2, m, &mut b);
        assert!((b[0] - (-0.5)).abs() < 1e-12);
        assert!((b[1] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn solve_upper_tri_transposed_matches_forward_sub() {
        // J = [[2, 3], [0, 2]] → Jᵀ x = b for b = (4, 11).
        // Jᵀ = [[2, 0], [3, 2]]. So 2 x1 = 4 → x1 = 2; 3·2 + 2 x2 =
        // 11 → x2 = 2.5.
        let m = 2;
        let mut j_upper = vec![0.0; m * m];
        j_upper[0 * m + 0] = 2.0;
        j_upper[0 * m + 1] = 3.0;
        j_upper[1 * m + 1] = 2.0;
        let mut b = vec![4.0, 11.0];
        solve_upper_tri_transposed(&j_upper, 2, m, &mut b);
        assert!((b[0] - 2.0).abs() < 1e-12);
        assert!((b[1] - 2.5).abs() < 1e-12);
    }

    #[test]
    fn formt_col_one_gives_theta_ss_then_sqrt() {
        // col = 1: T = θ · ss[0,0] (no L D⁻¹ Lᵀ contribution).
        // Cholesky of 1x1 just takes sqrt.
        let m = 3;
        let mut sy = vec![0.0; m * m];
        let mut ss = vec![0.0; m * m];
        sy[0] = 11.0; // s·y
        ss[0] = 5.0; // s·s
        let theta = 25.0 / 11.0;
        let mut wt = vec![0.0; m * m];
        formt(theta, &sy, &ss, 1, m, &mut wt).unwrap();
        assert!((wt[0] - (theta * 5.0).sqrt()).abs() < 1e-12);
    }

    #[test]
    fn bmv_returns_zero_for_col_zero() {
        // col = 0 → bmv is a no-op; p stays untouched.
        let sy = vec![0.0; 4];
        let wt = vec![0.0; 4];
        let v = vec![1.0, 2.0];
        let mut p = vec![99.0, 99.0];
        assert!(bmv(&sy, &wt, 0, 2, &v, &mut p).is_ok());
        assert_eq!(p, vec![99.0, 99.0]);
    }

    #[test]
    fn bmv_col_one_matches_2x2_inverse() {
        // For col = 1 the middle matrix is the 2 × 2 block
        //   M_inv = [ −D    Lᵀ ] = [ −d   0 ]
        //           [  L  θ SᵀS ]   [  0  θ·ss ]
        // (L is empty for col = 1.) Then
        //   M = [ −1/d        0           ]
        //       [   0     1/(θ · ss)      ]
        // and bmv computes M v.
        //
        // Picking d = sy[0,0] = 11, ss[0,0] = 5, θ = 25/11 (so the
        // theta from the (1, 2)/(3, 4) pair in the LbfgsState test).
        let m = 2;
        let mut sy = vec![0.0; m * m];
        let mut ss = vec![0.0; m * m];
        sy[0] = 11.0;
        ss[0] = 5.0;
        let theta = 25.0 / 11.0;
        let mut wt = vec![0.0; m * m];
        formt(theta, &sy, &ss, 1, m, &mut wt).unwrap();

        let v = vec![7.0, 9.0]; // (v1, v2)
        let mut p = vec![0.0; 2];
        bmv(&sy, &wt, 1, m, &v, &mut p).unwrap();

        let d = 11.0;
        let exp_p1 = -v[0] / d;
        let exp_p2 = v[1] / (theta * 5.0);
        assert!((p[0] - exp_p1).abs() < 1e-12, "p1 = {} vs {}", p[0], exp_p1);
        assert!((p[1] - exp_p2).abs() < 1e-12, "p2 = {} vs {}", p[1], exp_p2);
    }
}
