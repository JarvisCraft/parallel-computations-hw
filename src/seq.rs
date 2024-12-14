use rayon::prelude::*;

use crate::{
    task::{Matrix, Solution, Task},
    types::ZERO,
};

pub fn multiply<const N: usize>(a: &Matrix<N>, b: &Matrix<N>) -> Matrix<N> {
    let a = a.as_slice();
    let b = b.as_slice();
    let mut c = vec![ZERO; N * N];

    for row in 0..N {
        for column in 0..N {
            let mut value = 0.;
            for k in 0..N {
                value += a[k * N + row] * b[column * N + k];
            }
            c[column * N + row] = value;
        }
    }

    c.into_boxed_slice().try_into().unwrap()
}

pub fn solve<const N: usize>(task: Task<N>) -> Solution<N> {
    let n = task.0.len();
    Solution(
        (0..n)
            .into_par_iter()
            .map(|index| {
                task.0
                    .iter()
                    .cycle()
                    .skip(index)
                    .take(n)
                    .cloned()
                    .reduce(|l, r| multiply(&l, &r))
                    .expect("This is unrechable when `n` is zero")
            })
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_2x2() {
        let a = Matrix::<2>::from_vec(vec![1., 3., 2., 4.]).unwrap();
        let b = Matrix::<2>::from_vec(vec![5., 7., 6., 8.]).unwrap();

        assert_eq!(
            multiply(&a, &b),
            Matrix::from_vec(vec![19., 43., 22., 50.]).unwrap(),
        );
    }
}
