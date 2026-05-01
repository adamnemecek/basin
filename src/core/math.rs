pub trait ScaledAdd<S> {
    fn scaled_add(&mut self, scalar: S, other: &Self);
}

pub trait NormSquared {
    fn norm_squared(&self) -> f64;
}

mod vec;

#[cfg(feature = "nalgebra")]
mod nalgebra_backend;
