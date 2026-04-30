pub trait CostFunction {
    type Param;
    type Output;

    fn cost(&self, param: &Self::Param) -> Self::Output;
}

pub trait Gradient {
    type Param;
    type Gradient;

    fn gradient(&self, param: &Self::Param) -> Self::Gradient;
}
