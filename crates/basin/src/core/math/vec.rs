use super::{ClampInPlace, Dot, NegInPlace, NormInfinity, NormSquared, ScaledAdd};

impl ScaledAdd<f64> for Vec<f64> {
    fn scaled_add(&mut self, scalar: f64, other: &Self) {
        assert_eq!(self.len(), other.len(), "scaled_add: length mismatch");
        for (x, y) in self.iter_mut().zip(other.iter()) {
            *x += scalar * y;
        }
    }
}

impl NormSquared for Vec<f64> {
    fn norm_squared(&self) -> f64 {
        self.iter().map(|x| x * x).sum()
    }
}

impl NormInfinity for Vec<f64> {
    fn norm_infinity(&self) -> f64 {
        self.iter().map(|x| x.abs()).fold(0.0, f64::max)
    }
}

impl Dot for Vec<f64> {
    fn dot(&self, other: &Self) -> f64 {
        assert_eq!(self.len(), other.len(), "dot: length mismatch");
        self.iter().zip(other.iter()).map(|(a, b)| a * b).sum()
    }
}

impl NegInPlace for Vec<f64> {
    fn neg_in_place(&mut self) {
        for x in self.iter_mut() {
            *x = -*x;
        }
    }
}

impl ClampInPlace for Vec<f64> {
    fn clamp_in_place(&mut self, lower: &Self, upper: &Self) {
        assert_eq!(
            self.len(),
            lower.len(),
            "clamp_in_place: lower length mismatch"
        );
        assert_eq!(
            self.len(),
            upper.len(),
            "clamp_in_place: upper length mismatch"
        );
        for ((x, &lo), &hi) in self.iter_mut().zip(lower.iter()).zip(upper.iter()) {
            *x = x.clamp(lo, hi);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clamp_inside_box_is_identity() {
        let mut x = vec![0.5, -0.5, 1.5];
        x.clamp_in_place(&vec![-1.0, -1.0, 0.0], &vec![1.0, 1.0, 2.0]);
        assert_eq!(x, vec![0.5, -0.5, 1.5]);
    }

    #[test]
    fn clamp_partially_outside_pins_only_offending_components() {
        let mut x = vec![-2.0, 0.5, 3.0];
        x.clamp_in_place(&vec![-1.0, -1.0, -1.0], &vec![1.0, 1.0, 1.0]);
        assert_eq!(x, vec![-1.0, 0.5, 1.0]);
    }

    #[test]
    fn clamp_entirely_outside_pins_to_nearest_face() {
        let mut x = vec![-10.0, 10.0];
        x.clamp_in_place(&vec![-1.0, -1.0], &vec![1.0, 1.0]);
        assert_eq!(x, vec![-1.0, 1.0]);
    }

    #[test]
    fn clamp_with_equal_bounds_pins_to_value() {
        let mut x = vec![5.0, -5.0];
        x.clamp_in_place(&vec![1.0, 2.0], &vec![1.0, 2.0]);
        assert_eq!(x, vec![1.0, 2.0]);
    }
}
