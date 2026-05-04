use crate::core::problem::CostFunction;

/// Box (interval) bounds on the parameter.
///
/// Lives on the *problem* side (tenet 4 in `AGENTS.md`): constraints
/// describe the problem, not the executor. Solvers that require box
/// bounds bind on this trait so handing them an unconstrained problem is
/// a compile error rather than a silent runtime issue.
///
/// `BoxConstrained` is a supertrait of `CostFunction` so the `Param` type
/// is shared automatically — solver bounds read
/// `P: BoxConstrained<Param = f64>` instead of repeating the parameter
/// type across two trait bounds.
///
/// For 1D problems `Param = f64` and bounds are scalars; for n-D box
/// constraints `Param` is a vector and bounds are vectors of the same
/// length.
pub trait BoxConstrained: CostFunction {
    fn lower(&self) -> &Self::Param;
    fn upper(&self) -> &Self::Param;
}
