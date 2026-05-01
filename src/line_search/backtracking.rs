use crate::core::math::{Dot, ScaledAdd};
use crate::core::problem::CostFunction;
use crate::line_search::{LineSearch, LineSearchResult};

pub struct Backtracking {
    pub alpha_init: f64,
    pub rho: f64,
    pub c: f64,
    pub max_iter: u32,
}

impl Default for Backtracking {
    fn default() -> Self {
        Self {
            alpha_init: 1.0,
            rho: 0.5,
            c: 1e-4,
            max_iter: 50,
        }
    }
}

impl Backtracking {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn alpha_init(mut self, alpha_init: f64) -> Self {
        self.alpha_init = alpha_init;
        self
    }

    pub fn rho(mut self, rho: f64) -> Self {
        self.rho = rho;
        self
    }

    pub fn c(mut self, c: f64) -> Self {
        self.c = c;
        self
    }

    pub fn max_iter(mut self, max_iter: u32) -> Self {
        self.max_iter = max_iter;
        self
    }
}

impl<P, V> LineSearch<P, V> for Backtracking
where
    P: CostFunction<Param = V, Output = f64>,
    V: ScaledAdd<f64> + Dot + Clone,
{
    fn next(
        &mut self,
        problem: &P,
        param: &V,
        cost: f64,
        gradient: &V,
        direction: &V,
    ) -> LineSearchResult {
        // Armijo: f(x + α d) ≤ f(x) + c α (∇f · d). For a descent direction,
        // `g_dot_d` is negative, so the threshold drops with α.
        let g_dot_d = gradient.dot(direction);
        let mut alpha = self.alpha_init;
        let mut cost_evals = 0u64;
        for _ in 0..self.max_iter {
            let mut trial = param.clone();
            trial.scaled_add(alpha, direction);
            let trial_cost = problem.cost(&trial);
            cost_evals += 1;
            if trial_cost <= cost + self.c * alpha * g_dot_d {
                return LineSearchResult {
                    alpha,
                    cost_evals,
                    gradient_evals: 0,
                };
            }
            alpha *= self.rho;
        }
        LineSearchResult {
            alpha,
            cost_evals,
            gradient_evals: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 1D quadratic via Vec<f64>: f(x) = (x[0] − 3)². Min at x = 3,
    /// ∇f = 2(x − 3).
    struct Quadratic;

    impl CostFunction for Quadratic {
        type Param = Vec<f64>;
        type Output = f64;
        fn cost(&self, x: &Vec<f64>) -> f64 {
            (x[0] - 3.0).powi(2)
        }
    }

    fn run(ls: &mut Backtracking, x: &[f64], grad: &[f64], dir: &[f64]) -> LineSearchResult {
        let p = Quadratic;
        let x = x.to_vec();
        let f0 = p.cost(&x);
        let g = grad.to_vec();
        let d = dir.to_vec();
        ls.next(&p, &x, f0, &g, &d)
    }

    #[test]
    fn accepts_alpha_init_when_armijo_holds() {
        // Start at x=0, descent direction d=+6 (= −g for g = −6). Armijo
        // with the default α_init=1.0 lands at x=6, f(6)=9, threshold
        // f(0) + c·α·gᵀd = 0 + 1e-4·1·(−36) = −0.0036. 9 ≤ −0.0036 fails,
        // so we expect at least one backtrack. Use α_init=0.1 to get
        // accept-on-first-try: x=0.6, f=5.76, threshold 0 − 0.00036.
        // Still 5.76 > −0.00036 — so even α=0.1 backtracks.
        // For accept-first: pick d so the line minimum is past α_init.
        // Start at x=2, d=+1 (descent: g=−2, gᵀd=−2 < 0). α_init=0.5
        // → x=2.5, f=0.25, threshold 1 − 1e-4·0.5·2 = 0.9999. 0.25 ≤ 0.9999 ✓.
        let mut ls = Backtracking::new().alpha_init(0.5);
        let r = run(&mut ls, &[2.0], &[-2.0], &[1.0]);
        assert_eq!(r.alpha, 0.5, "expected α_init accepted on first try");
        assert_eq!(r.cost_evals, 1);
        assert_eq!(r.gradient_evals, 0);
    }

    #[test]
    fn backtracks_when_initial_alpha_overshoots() {
        // From x=0, g=−6, direction d=+6. α_init=1.0 lands at x=6, f=9
        // (way past minimum at x=3). Backtrack until Armijo holds.
        let mut ls = Backtracking::new(); // ρ=0.5, c=1e-4, max_iter=50
        let r = run(&mut ls, &[0.0], &[-6.0], &[6.0]);
        // Armijo at returned α: f(α·d) ≤ f(0) + c·α·gᵀd.
        let f0 = 9.0; // (0-3)^2
        let f_new = (r.alpha * 6.0 - 3.0).powi(2);
        let g_dot_d = (-6.0_f64) * 6.0;
        assert!(
            f_new <= f0 + 1e-4 * r.alpha * g_dot_d,
            "Armijo violated: f_new={f_new}, threshold={}",
            f0 + 1e-4 * r.alpha * g_dot_d,
        );
        // Should have shrunk at least once.
        assert!(r.alpha < 1.0, "expected backtrack, got α={}", r.alpha);
        assert!(r.cost_evals > 1);
        assert_eq!(r.gradient_evals, 0);
    }

    #[test]
    fn reports_cost_eval_count() {
        let mut ls = Backtracking::new().rho(0.5);
        let r = run(&mut ls, &[0.0], &[-6.0], &[6.0]);
        // Each backtrack iteration consumes exactly one cost eval.
        assert!(r.cost_evals >= 1);
        assert!(
            r.cost_evals <= ls.max_iter as u64,
            "cost_evals={} exceeds max_iter={}",
            r.cost_evals,
            ls.max_iter
        );
    }

    #[test]
    fn caps_at_max_iter_when_armijo_never_holds() {
        // Wrong-sign direction: gᵀd > 0, so f increases with α and Armijo
        // (with descent-direction assumption) is unsatisfiable. Backtrack
        // burns max_iter cost evals and returns the smallest α tried.
        let mut ls = Backtracking::new().max_iter(5);
        let r = run(&mut ls, &[0.0], &[-6.0], &[-6.0]);
        assert_eq!(r.cost_evals, 5);
        // α reduced 5 times by ρ=0.5 from 1.0 → 1/32.
        assert!(
            (r.alpha - 1.0 / 32.0).abs() < 1e-12,
            "expected α=1/32, got {}",
            r.alpha
        );
    }
}
