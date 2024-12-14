use std::ops::{Index, IndexMut};

use crate::{types::Value, util::sqrt};

/// Column-major matrix.
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct Matrix {
    n: usize,
    values: Box<[Value]>,
}
impl Matrix {
    pub const fn n(&self) -> usize {
        self.n
    }

    pub fn as_slice(&self) -> &[Value] {
        self.values.as_ref()
    }

    pub const fn len(&self) -> usize {
        self.values.len()
    }

    pub fn from_vec(vec: Vec<Value>) -> Option<Self> {
        let n = sqrt(vec.len());
        if vec.len() == n * n {
            Some(Self {
                n,
                values: vec.into_boxed_slice(),
            })
        } else {
            None
        }
    }
}
impl TryFrom<Box<[Value]>> for Matrix {
    type Error = ();

    fn try_from(values: Box<[Value]>) -> Result<Self, Self::Error> {
        let n = sqrt(values.len());
        if values.len() == n * n {
            Ok(Self { n, values })
        } else {
            Err(())
        }
    }
}
impl TryFrom<Vec<Value>> for Matrix {
    type Error = ();

    fn try_from(value: Vec<Value>) -> Result<Self, Self::Error> {
        Self::from_vec(value).ok_or(())
    }
}
impl Index<usize> for Matrix {
    type Output = Value;

    fn index(&self, index: usize) -> &Self::Output {
        &self.values[index]
    }
}
impl IndexMut<usize> for Matrix {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.values[index]
    }
}

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct Task {
    n: usize,
    matrices: Vec<Matrix>,
}
impl Task {
    pub fn from_vec(matrices: Vec<Matrix>) -> Option<Self> {
        let first = matrices.first()?;
        let n = first.n();
        for matrix in &matrices {
            if matrix.n() != n {
                return None;
            }
        }

        Some(Self { n, matrices })
    }

    pub const fn n(&self) -> usize {
        self.n
    }

    pub fn matrices(&self) -> &[Matrix] {
        self.matrices.as_slice()
    }
}

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct Solution(pub Vec<Matrix>);
