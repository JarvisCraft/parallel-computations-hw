#[derive(Debug)]
#[repr(transparent)]
pub struct Matrix<const N: usize>(Box<[f64]>);
impl<const N: usize> Matrix<N> {
    pub fn of(value: f64) -> Self {
        Self(vec![value; N * N].into_boxed_slice())
    }

    pub fn zeros() -> Self {
        Self::of(0.)
    }

    pub fn ones() -> Self {
        Self::of(1.)
    }

    pub fn into_inner(self) -> Box<[f64]> {
        self.0
    }
}
impl<const N: usize> TryFrom<Box<[f64]>> for Matrix<N> {
    type Error = ();

    fn try_from(value: Box<[f64]>) -> Result<Self, Self::Error> {
        if value.len() == N * N {
            Ok(Self(value))
        } else {
            Err(())
        }
    }
}
