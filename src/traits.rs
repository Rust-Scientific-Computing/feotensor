pub trait MatMul<Rhs = Self> {
    type Output;

    fn matmul(self, rhs: &Rhs) -> Self::Output;
}
