use rayon::prelude::*;

use crate::{
    task::{Matrix, Solution, Task},
    types::ZERO,
};

pub fn multiply(a: &Matrix, b: &Matrix) -> Matrix {
    assert!(a.n() == b.n(), "matrices should have the same dimensions");
    let n = a.n();

    let a = a.as_slice();
    let b = b.as_slice();

    let mut c = vec![ZERO; n * n];

    for row in 0..n {
        for column in 0..n {
            let mut value = 0.;
            for k in 0..n {
                value += a[k * n + row] * b[column * n + k];
            }
            c[column * n + row] = value;
        }
    }

    c.into_boxed_slice()
        .try_into()
        .unwrap_or_else(|_| panic!("n = {n}"))
}

pub fn solve(task: &Task) -> Solution {
    let n = task.matrices().len();

    Solution(
        (0..n)
            .map(|index| {
                multiply_all(task.matrices().iter().cycle().skip(index).take(n).cloned())
                    .expect("This is unrechable when `n` is zero")
            })
            .collect(),
    )
}

pub fn solve_par(task: &Task) -> Solution {
    let n = task.matrices().len();

    Solution(
        (0..n)
            .into_par_iter()
            .map(|index| {
                multiply_all(task.matrices().iter().cycle().skip(index).take(n).cloned())
                    .expect("This is unrechable when `n` is zero")
            })
            .collect(),
    )
}

pub fn multiply_all(matrices: impl Iterator<Item = Matrix>) -> Option<Matrix> {
    matrices.reduce(|l, r| multiply(&l, &r))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_2x2() {
        let a = Matrix::from_vec(vec![1., 3., 2., 4.]).unwrap();
        let b = Matrix::from_vec(vec![5., 7., 6., 8.]).unwrap();

        assert_eq!(
            multiply(&a, &b),
            Matrix::from_vec(vec![19., 43., 22., 50.]).unwrap(),
        );
    }
}
