pub trait State {
    type Param;
    type Float;

    fn iter(&self) -> u64;
    fn increment_iter(&mut self);
    fn param(&self) -> &Self::Param;
    fn cost(&self) -> Self::Float;
}

/// States that carry a gradient at the current `param`.
///
/// Solvers that compute a gradient should populate `gradient()` so that
/// gradient-based termination criteria can read it without re-evaluating
/// the problem. `None` means "no gradient available at this iterate yet"
/// (e.g. before `Solver::init` has run).
pub trait GradientState: State {
    fn gradient(&self) -> Option<&Self::Param>;
}

pub struct BasicState<P> {
    pub param: P,
    pub cost: f64,
    pub gradient: Option<P>,
    pub iter: u64,
}

impl<P> BasicState<P> {
    pub fn new(param: P) -> Self {
        Self {
            param,
            cost: f64::INFINITY,
            gradient: None,
            iter: 0,
        }
    }
}

impl<P> State for BasicState<P> {
    type Param = P;
    type Float = f64;

    fn iter(&self) -> u64 {
        self.iter
    }

    fn increment_iter(&mut self) {
        self.iter += 1;
    }

    fn param(&self) -> &P {
        &self.param
    }

    fn cost(&self) -> f64 {
        self.cost
    }
}

impl<P> GradientState for BasicState<P> {
    fn gradient(&self) -> Option<&P> {
        self.gradient.as_ref()
    }
}
