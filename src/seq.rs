use crate::{task::Matrix, types::ZERO};

pub fn multiply<const N: usize>(a: Matrix<N>, b: Matrix<N>) -> Matrix<N> {
    let a = a.into_inner();
    let b = b.into_inner();
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
