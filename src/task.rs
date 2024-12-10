use crate::types::Value;

/// Column-major matrix.
#[derive(Debug, Clone, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct Matrix<const N: usize>(Box<[Value]>);
impl<const N: usize> Matrix<N> {
    pub fn of(value: Value) -> Self {
        Self(vec![value; N * N].into_boxed_slice())
    }

    pub fn zeros() -> Self {
        Self::of(0.)
    }

    pub fn ones() -> Self {
        Self::of(1.)
    }

    pub fn into_inner(self) -> Box<[Value]> {
        self.0
    }

    pub const fn len(&self) -> usize {
        self.0.len()
    }
}
impl<const N: usize> TryFrom<Box<[Value]>> for Matrix<N> {
    type Error = ();

    fn try_from(value: Box<[Value]>) -> Result<Self, Self::Error> {
        if value.len() == N * N {
            Ok(Self(value))
        } else {
            Err(())
        }
    }
}
