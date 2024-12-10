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
    if n == 0 {
        return Solution(vec![]);
    }

    Solution(
        (0..n)
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
