use std::ops::{Index, IndexMut};

use crate::types::{Value, ONE, ZERO};

/// Column-major matrix.
#[derive(Debug, Clone, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct Matrix<const N: usize>(Box<[Value]>);
impl<const N: usize> Matrix<N> {
    pub fn of(value: Value) -> Self {
        Self(vec![value; N * N].into_boxed_slice())
    }

    pub fn zeros() -> Self {
        Self::of(ZERO)
    }

    pub fn ones() -> Self {
        Self::of(ONE)
    }

    pub fn as_slice(&self) -> &[Value] {
        self.0.as_ref()
    }

    pub const fn len(&self) -> usize {
        self.0.len()
    }

    pub fn from_vec(vec: Vec<Value>) -> Option<Self> {
        if vec.len() == N * N {
            Some(Self(vec.into_boxed_slice()))
        } else {
            None
        }
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
impl<const N: usize> TryFrom<Vec<Value>> for Matrix<N> {
    type Error = ();

    fn try_from(value: Vec<Value>) -> Result<Self, Self::Error> {
        Self::from_vec(value).ok_or(())
    }
}
impl<const N: usize> Index<usize> for Matrix<N> {
    type Output = Value;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}
impl<const N: usize> IndexMut<usize> for Matrix<N> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct Task<const N: usize>(pub Vec<Matrix<N>>);

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct Solution<const N: usize>(pub Vec<Matrix<N>>);
